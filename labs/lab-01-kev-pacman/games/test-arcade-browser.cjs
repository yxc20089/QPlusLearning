// CPU controller test: actual native engine, mock notebook transport/DOM.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const listeners={},messages=[];
let drawFrame;
function Audio(){this.paused=true;this.load=this.pause=this.addEventListener=this.removeEventListener=()=>{};this.play=()=>Promise.resolve();}
const parent={postMessage(message){messages.push(message);}};
const context={Audio,parent,document:{},localStorage:{},console:{log(){},error(){}},
 window:{addEventListener(name,fn){(listeners[name]??=[]).push(fn);},location:{hash:''}},
 setTimeout(){return 1;},clearTimeout(){},setInterval(){},clearInterval(){},
 requestAnimationFrame(fn){drawFrame=fn;return 1;},cancelAnimationFrame(){}};
vm.createContext(context);
context.LAB_ACTION_VERSION=Number(process.argv[4]||1);
let engine=fs.readFileSync(process.argv[2],'utf8'),end=engine.lastIndexOf('})();');
engine=engine.slice(0,end)+fs.readFileSync(process.argv[3],'utf8')+'\n'+engine.slice(end);
vm.runInContext(engine,context);context.labArcade.headless();
listeners.load.at(-1)();
const handler=listeners.message.at(-1);
const send=(type,data,extra={})=>handler({source:parent,data:{channel:'kev-classic-pacman',type,data,...extra}});
const callback=()=>messages.filter(m=>m.type==='callback').at(-1);
const identity={checkpoint:'/checkpoints/kev-4b-skills',checkpoint_name:'kev-4b-skills',stage:'skills',lora_rank:16};
let now=0;
async function reply(message,data,error){await send('callback-result',data,{id:message.id,error});await new Promise(resolve=>setImmediate(resolve));}
async function runToDecision(){
 for(let i=0;i<500;i++){now+=1000/60;drawFrame(now);await Promise.resolve();if(callback()?.callback==='pacman.decide')return callback();}
 throw new Error('No player decision at play boundary');
}
(async()=>{
 await send('command','kev');
 const start=send('command','start');
 assert.equal(callback().callback,'pacman.model');await reply(callback(),identity);await start;
 const decision=await runToDecision();
 assert.equal(decision.argument.engine_revision,'7407174c1d6a38be8cd230577489e39e0873145b');
 const frozen=JSON.stringify(context.labArcade.observe());
 for(let i=0;i<30;i++){now+=1000/60;drawFrame(now);}
 assert.equal(JSON.stringify(context.labArcade.observe()),frozen,'simulation must wait for inference');
 await reply(decision,{answers:{move:{choice:'left',probabilities:{left:1,right:0}}},active_checkpoint:identity});
 for(let i=0;i<12;i++){now+=1000/60;drawFrame(now);await Promise.resolve();}
 assert.equal(context.labArcade.observe().turn,1);
 const firstState=context.labArcade.observe();
 let reverseState;
 if(context.LAB_ACTION_VERSION===2){
  await reply(callback(),{answers:{move:{choice:'right',probabilities:Object.fromEntries(callback().argument.legal_moves.map(d=>[d,d==='right'?1:0]))}},active_checkpoint:identity});
  for(let i=0;i<12;i++){now+=1000/60;drawFrame(now);await Promise.resolve();}
  reverseState=context.labArcade.observe();
  assert.equal(reverseState.turn,2);
 }
 assert.ok(messages.some(m=>m.type==='decision'&&m.data.active_checkpoint.checkpoint===identity.checkpoint));
 const stale=callback();
 assert.equal(stale.callback,'pacman.decide');await send('command','reset');
 const reset=JSON.stringify(context.labArcade.observe());
 await reply(stale,{answers:{move:{choice:'left',probabilities:{left:1,right:0}}},active_checkpoint:identity});
 assert.equal(JSON.stringify(context.labArcade.observe()),reset,'reset must reject stale inference');
 const restart=send('command','start');await reply(callback(),identity);await restart;
 const failure=await runToDecision();await reply(failure,null,'fixture API failed');
 const stopped=JSON.stringify(context.labArcade.observe());
 for(let i=0;i<30;i++){now+=1000/60;drawFrame(now);}
 assert.equal(JSON.stringify(context.labArcade.observe()),stopped);
 assert.ok(messages.some(m=>m.type==='error'&&m.data.includes('fixture API failed')));
 process.stdout.write(JSON.stringify({status:'Browser/native parity, adapter transport, wait, reset and API-error pause passed.',firstState,reverseState}));
})().catch(error=>{console.error(error);process.exitCode=1;});
