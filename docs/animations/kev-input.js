/* Teaching view of the pinned Kev encoder: spans, markers and hidden readouts. */
(function(global){
  'use strict';
  const C={ink:'#111827',muted:'#5e6e80',teal:'#167985',orange:'#b36b17',gray:'#dce4e8',light:'#f5f7f9',tealFill:'#eff8f8',orangeFill:'#fff4e4'};
  const spans=[{id:'state',x:36,w:151},{id:'question',x:195,w:108},{id:'left',x:311,w:95},{id:'up',x:414,w:95},{id:'right',x:517,w:95},{id:'decide',x:620,w:64}];
  const rects=spans.map((v,i)=>({...v,y:136,h:54,fill:i===0?C.light:i===1?C.tealFill:C.orangeFill,stroke:i<2?C.gray:C.orange}));
  rects.push({id:'backbone',x:36,y:215,w:648,h:40,fill:C.tealFill,stroke:C.teal});
  spans.forEach((v,i)=>rects.push({...v,id:'hidden-'+v.id,y:279,h:31,fill:i<2?C.light:C.orangeFill,stroke:i<2?C.gray:C.orange}));
  const lines=spans.flatMap(v=>[[v.x+v.w/2,190,v.x+v.w/2,213,C.teal,true],[v.x+v.w/2,255,v.x+v.w/2,277,C.teal,true]]);
  const texts=[
    {id:'title',x:36,y:25,w:648,h:42,size:30,bold:true,text:'From board state to hidden vectors'},
    {id:'subtitle',x:36,y:76,w:648,h:25,size:17,text:'State + instruction + legal moves → one structured input'},
    {id:'state-label',x:36,y:112,w:151,h:22,size:15,bold:true,text:'State s'},
    {id:'question-label',x:195,y:112,w:108,h:22,size:15,bold:true,text:'Instruction u'},
    {id:'options-label',x:311,y:112,w:301,h:22,size:15,bold:true,color:C.orange,text:'Ordered options A'},
    {id:'decide-label',x:620,y:114,w:64,h:20,size:13,bold:true,text:'Decide'},
    {id:'state',x:42,y:144,w:139,h:40,size:13,mono:true,text:'<state>\nMaze, ghosts, ...'},
    {id:'question',x:201,y:144,w:96,h:40,size:12,mono:true,text:'<q>\nChoose a move'},
    {id:'left',x:317,y:143,w:83,h:43,size:12,mono:true,text:'<opt> Left\n</opt>'},
    {id:'up',x:420,y:143,w:83,h:43,size:12,mono:true,text:'<opt> Up\n</opt>'},
    {id:'right',x:523,y:143,w:83,h:43,size:12,mono:true,text:'<opt> Right\n</opt>'},
    {id:'decide',x:623,y:151,w:58,h:30,size:12,mono:true,center:true,text:'<decide>'},
    {id:'backbone-label',x:36,y:221,w:648,h:29,size:17,bold:true,color:C.teal,center:true,text:'Qwen3.5-4B backbone + LoRA'},
    {id:'hidden-state',x:36,y:284,w:151,h:23,size:17,color:C.muted,center:true,text:'…'},
    {id:'hidden-question',x:195,y:284,w:108,h:23,size:17,color:C.muted,center:true,text:'…'},
    {id:'hidden-left',x:311,y:282,w:95,h:26,size:18,formula:true,center:true,text:'hend,1',subscript:[1,6]},
    {id:'hidden-up',x:414,y:282,w:95,h:26,size:18,formula:true,center:true,text:'hend,2',subscript:[1,6]},
    {id:'hidden-right',x:517,y:282,w:95,h:26,size:18,formula:true,center:true,text:'hend,3',subscript:[1,6]},
    {id:'hidden-decide',x:620,y:282,w:64,h:26,size:18,formula:true,center:true,text:'hdecide',subscript:[1,7]},
    {id:'readout-caption',x:36,y:315,w:648,h:23,size:13,color:C.muted,text:'The pointer head reads each option ending and the final decision vector'},
    {id:'input-formula',x:36,y:341,w:310,h:30,size:20,formula:true,text:'x = encode(s, u, A)'},
    {id:'hidden-formula',x:390,y:341,w:165,h:30,size:20,formula:true,text:'H ∈ ℝL×d',superscript:[5,8]},
    {id:'dimensions',x:555,y:345,w:129,h:25,size:12,color:C.muted,text:'L tokens × d features'},
    {id:'footer',x:36,y:383,w:615,h:17,size:9,color:C.muted,text:'Schematic token groups. Readable marker names. Open animation controls.'},
    {id:'number',x:665,y:383,w:19,h:17,size:9,color:C.muted,text:'2'},
  ];
  const stages=[
    {title:'1. Describe the current board',text:'s is the board state. In the lab it includes the maze, pellets and power pellets, Pac-Man and ghost information, timers and recent movement. The state text begins with a boundary token.'},
    {title:'2. State the decision',text:'u is the instruction. It tells Kev what to choose. The question marker separates it from the board description.'},
    {title:'3. Supply the legal options',text:'A is an ordered list of moves. Each option has an opening and closing marker. The illustration uses Left, Up and Right; actual lab requests include the legal moves and their descriptions.'},
    {title:'4. End with the decision marker',text:'The decide marker comes after every option. Its hidden vector can use the state, instruction and all choices. With this causal configuration, later option endings can also use earlier options.'},
    {title:'5. Compute a hidden vector at every token',text:'Qwen processes x, the encoded input sequence. H contains L hidden vectors of width d: one row per token, one column per feature. The boxes group text spans and do not represent literal token counts.'},
    {title:'6. Read the vectors used by the pointer head',text:'The head reads h at each option closing marker and at the final decide marker. The rest of H supplies their context. This is position-based readout, rather than an average over all token vectors.'},
  ];
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
  function highlight(r,t,color=C.orange){return `<rect x="${r.x-3}" y="${r.y-3}" width="${r.w+6}" height="${r.h+6}" rx="3" fill="none" stroke="${color}" stroke-width="${1.2+Math.sin(t*Math.PI)*1.4}"/>`;}
  function overlay(stage,t){
    if(stage<2)return highlight(rects[stage],t,C.teal);
    if(stage===2)return rects.slice(2,5).map(r=>highlight(r,t)).join('');
    if(stage===3)return highlight(rects[5],t);
    if(stage===4)return highlight(rects[6],t,C.teal);
    return rects.slice(9).map(r=>highlight(r,t)).join('');
  }
  function content(v,line){
    const range=v.subscript||v.superscript;
    if(!range)return esc(line);
    return esc(line.slice(0,range[0]))+`<tspan baseline-shift="${v.subscript?'sub':'super'}" font-size="${v.size*.7}">${esc(line.slice(...range))}</tspan>`+esc(line.slice(range[1]));
  }
  function diagram(){
    let s=`<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="${C.teal}"/></marker></defs>`;
    s+=rects.map(r=>`<rect x="${r.x}" y="${r.y}" width="${r.w}" height="${r.h}" rx="3" fill="${r.fill}" stroke="${r.stroke}"/>`).join('');
    s+=lines.map(([x1,y1,x2,y2,c])=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${c}" stroke-width="1.2" marker-end="url(#arrow)"/>`).join('');
    s+=texts.map(v=>`<text x="${v.center?v.x+v.w/2:v.x}" y="${v.y+v.size}" font-family="${v.formula?'Georgia':v.mono?'Roboto Mono, monospace':'Arial'}" font-size="${v.size}" font-weight="${v.bold?'700':'400'}" fill="${v.color||C.ink}"${v.center?' text-anchor="middle"':''}>${v.text.split('\n').map((line,i)=>`<tspan x="${v.center?v.x+v.w/2:v.x}" dy="${i?v.size*1.3:0}">${content(v,line)}</tspan>`).join('')}</text>`).join('');
    return s;
  }
  const api={C,rects,lines,texts,stages,overlay,diagram};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  global.KevInput=api;
  if(typeof document==='undefined')return;
  const svg=document.querySelector('#input');if(!svg)return;
  svg.innerHTML=diagram()+'<g id="flow" aria-hidden="true"></g>';
  let stage=0,progress=0,playing=false,previous=null;
  const buttons=[...document.querySelectorAll('[data-stage]')];
  function render(){
    document.querySelector('#flow').innerHTML=overlay(stage,progress);
    document.querySelector('#stage-title').textContent=stages[stage].title;
    document.querySelector('#explanation').textContent=stages[stage].text;
    document.querySelector('#play').textContent=playing?'Pause':'Play';
    document.querySelector('#play').setAttribute('aria-pressed',String(playing));
    buttons.forEach((b,i)=>b.setAttribute('aria-current',stage===i?'step':'false'));
    document.querySelector('#previous').disabled=stage===0;document.querySelector('#next').disabled=stage===stages.length-1;
  }
  function choose(i){stage=i;progress=0;playing=false;previous=null;render();}
  buttons.forEach(b=>b.addEventListener('click',()=>choose(Number(b.dataset.stage))));
  document.querySelector('#previous').addEventListener('click',()=>choose(Math.max(0,stage-1)));
  document.querySelector('#next').addEventListener('click',()=>choose(Math.min(stages.length-1,stage+1)));
  document.querySelector('#reset').addEventListener('click',()=>choose(0));
  document.querySelector('#play').addEventListener('click',()=>{if(!playing&&stage===stages.length-1&&progress===1){stage=0;progress=0;}playing=!playing;previous=null;render();});
  function tick(now){if(playing){if(previous!==null)progress+=(now-previous)/2500;if(progress>=1){progress=0;stage++;if(stage===stages.length){stage--;progress=1;playing=false;}}render();}previous=now;requestAnimationFrame(tick);}
  document.addEventListener('visibilitychange',()=>{previous=null;});render();requestAnimationFrame(tick);
})(globalThis);
