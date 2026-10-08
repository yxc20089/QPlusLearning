/* Option softmax and deterministic Choice selection from pinned Kev. */
(function(global){
  'use strict';
  const C={ink:'#111827',muted:'#5e6e80',teal:'#167985',orange:'#b36b17',gray:'#dce4e8',light:'#f5f7f9',tealFill:'#eff8f8',orangeFill:'#fff4e4'};
  const options=['Left','Up','Right'],scores=[1,0,2];
  function probabilities(T=1){const scaled=scores.map(z=>z/T),m=Math.max(...scaled),e=scaled.map(z=>Math.exp(z-m)),s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s);}
  const rects=[
    {id:'temperature',x:526,y:123,w:157,h:49,fill:C.tealFill,stroke:C.teal},
    {id:'scores',x:36,y:237,w:186,h:91,fill:C.light,stroke:C.gray},
    {id:'probabilities',x:280,y:237,w:246,h:91,fill:C.tealFill,stroke:C.gray},
    {id:'chosen',x:581,y:251,w:103,h:62,fill:C.orangeFill,stroke:C.orange}
  ];
  const lines=[[245,153,486,153,C.ink,false],[224,282,275,282,C.muted,true],[529,282,576,282,C.muted,true]];
  const texts=[
    {id:'title',x:36,y:25,w:648,h:42,size:30,bold:true,text:'Softmax and action selection'},
    {id:'subtitle',x:36,y:76,w:648,h:25,size:17,text:'Softmax gives option probabilities. Argmax chooses the largest.'},
    {id:'probability-formula',x:36,y:134,w:194,h:42,size:25,formula:true,text:'p(ai | s) =',subs:[[3,4]]},
    {id:'numerator',x:254,y:111,w:223,h:37,size:25,formula:true,center:true,text:'exp(zi / T)',subs:[[5,6]]},
    {id:'sum',x:248,y:158,w:43,h:42,size:32,formula:true,center:true,text:'Σ'},
    {id:'sum-upper',x:252,y:154,w:35,h:20,size:12,formula:true,center:true,text:'K'},
    {id:'sum-lower',x:246,y:193,w:47,h:20,size:12,formula:true,center:true,text:'j = 1'},
    {id:'denominator',x:300,y:166,w:186,h:35,size:25,formula:true,text:'exp(zj / T)',subs:[[5,6]]},
    {id:'temperature',x:526,y:134,w:157,h:34,size:23,formula:true,color:C.teal,center:true,text:'T = 1'},
    {id:'temperature-caption',x:514,y:180,w:181,h:23,size:13,color:C.muted,center:true,text:'Our fresh lab checkpoint'},
    {id:'scores-heading',x:36,y:214,w:186,h:22,size:15,bold:true,color:C.muted,text:'Raw scores'},
    {id:'probabilities-heading',x:280,y:214,w:246,h:22,size:15,bold:true,color:C.teal,text:'Option probabilities'},
    {id:'chosen-heading',x:565,y:214,w:119,h:22,size:15,bold:true,color:C.orange,center:true,text:'Chosen move'},
    {id:'chosen',x:581,y:268,w:103,h:35,size:23,bold:true,color:C.orange,center:true,text:'Right'},
    {id:'argmax',x:36,y:345,w:403,h:33,size:22,formula:true,text:'i* = argmaxi p(ai | s)',subs:[[11,12],[16,17]],supers:[[1,2]]},
    {id:'action-formula',x:449,y:345,w:235,h:33,size:22,formula:true,text:'a* = ai*',subs:[[6,8]],supers:[[1,2]]},
    {id:'footer',x:36,y:383,w:615,h:17,size:9,color:C.muted,text:'Illustrative scores, not a live prediction. Open animation controls.'},
    {id:'number',x:665,y:383,w:19,h:17,size:9,color:C.muted,text:'5'}
  ];
  options.forEach((label,i)=>{
    texts.push({id:'option-'+i,x:49,y:244+i*26,w:102,h:25,size:16,text:label});
    texts.push({id:'score-'+i,x:156,y:244+i*26,w:51,h:25,size:16,formula:true,center:true,text:scores[i].toFixed(1)});
  });
  function dynamicTexts(T=1){const ps=probabilities(T);return texts.map(v=>v.id==='temperature'?{...v,text:'T = '+T}:v.id==='temperature-caption'&&T!==1?{...v,text:'Illustrative temperature'}:v).concat(ps.map((p,i)=>({id:'percent-'+i,x:457,y:244+i*26,w:58,h:25,size:15,color:i===2?C.orange:C.teal,center:true,text:(p*100).toFixed(1)+'%'})));}
  function bars(T=1){return probabilities(T).flatMap((p,i)=>[
    {id:'bar-background-'+i,x:293,y:249+i*26,w:155,h:13,fill:'#dbe8e9',stroke:'#dbe8e9'},
    {id:'bar-'+i,x:293,y:249+i*26,w:155*p,h:13,fill:i===2?C.orange:C.teal,stroke:i===2?C.orange:C.teal}
  ]);}
  const stages=[
    {title:'1. Begin with the pointer scores',text:'The pointer head supplies one raw score z_i per option. Here Left, Up and Right have illustrative scores 1, 0 and 2. The scores themselves need not add up to one.'},
    {title:'2. Apply softmax over the supplied options',text:'Divide each score by temperature T, exponentiate it, and divide by the sum of the exponentials for all K options. The resulting option probabilities are positive and add up to one.'},
    {title:'3. Compare the option probabilities',text:'At T = 1, these scores give Left 24.5%, Up 9.0% and Right 66.5%. A probability describes the model’s preference among these choices. It is not a probability of surviving or winning the game.'},
    {title:'4. Choose the largest probability',text:'Kev’s Choice API takes the largest unrounded probability. It does not sample a move. Right wins this example. A positive temperature can sharpen or flatten the probabilities, but cannot change the ordering of fixed scores.'}
  ];
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
  const highlight=(r,t,c=C.orange)=>`<rect x="${r.x-3}" y="${r.y-3}" width="${r.w+6}" height="${r.h+6}" rx="3" fill="none" stroke="${c}" stroke-width="${1.2+Math.sin(t*Math.PI)*1.4}"/>`;
  function overlay(stage,t){
    if(stage===0)return highlight(rects[1],t,C.muted);
    if(stage===1)return highlight({x:32,y:109,w:460,h:99},t,C.teal);
    if(stage===2)return highlight(rects[2],t,C.teal);
    return highlight(rects[3],t,C.orange)+highlight({x:286,y:299,w:231,h:27},t,C.orange);
  }
  function rich(v){let p=0,s='';const ranges=[...(v.subs||[]).map(([a,b])=>[a,b,'sub']),...(v.supers||[]).map(([a,b])=>[a,b,'super'])].sort((a,b)=>a[0]-b[0]);for(const [a,b,shift] of ranges){s+=esc(v.text.slice(p,a))+`<tspan baseline-shift="${shift}" font-size="${v.size*.7}">${esc(v.text.slice(a,b))}</tspan>`;p=b;}return s+esc(v.text.slice(p));}
  function diagram(T=1){
    let s=`<defs><marker id="arrow-muted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="${C.muted}"/></marker></defs>`;
    s+=[...rects,...bars(T)].map(r=>`<rect x="${r.x}" y="${r.y}" width="${r.w}" height="${r.h}" rx="${r.id.startsWith('bar')?0:3}" fill="${r.fill}" stroke="${r.stroke}"/>`).join('');
    s+=lines.map(([x1,y1,x2,y2,c,arrow])=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${c}" stroke-width="1.2"${arrow?' marker-end="url(#arrow-muted)"':''}/>`).join('');
    s+=dynamicTexts(T).map(v=>`<text x="${v.center?v.x+v.w/2:v.x}" y="${v.y+v.size}" font-family="${v.formula?'Georgia':'Arial'}" font-size="${v.size}" font-weight="${v.bold?'700':'400'}" fill="${v.color||C.ink}"${v.center?' text-anchor="middle"':''}>${rich(v)}</text>`).join('');
    return s;
  }
  const api={C,options,scores,probabilities,rects,lines,texts,dynamicTexts,bars,stages,overlay,diagram};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  global.KevSoftmax=api;
  if(typeof document==='undefined')return;
  const svg=document.querySelector('#softmax');if(!svg)return;
  let stage=0,progress=0,playing=false,previous=null,T=1;
  const buttons=[...document.querySelectorAll('[data-stage]')],temperatures=[...document.querySelectorAll('[data-temperature]')];
  function render(){svg.innerHTML=diagram(T)+`<g aria-hidden="true">${overlay(stage,progress)}</g>`;document.querySelector('#stage-title').textContent=stages[stage].title;document.querySelector('#explanation').textContent=stage===2?`At T = ${T}, Left is ${(100*probabilities(T)[0]).toFixed(1)}%, Up is ${(100*probabilities(T)[1]).toFixed(1)}% and Right is ${(100*probabilities(T)[2]).toFixed(1)}%. These probabilities sum to one. They express a preference among the supplied options, not a chance of surviving or winning.`:stages[stage].text;document.querySelector('#play').textContent=playing?'Pause':'Play';document.querySelector('#play').setAttribute('aria-pressed',String(playing));buttons.forEach((b,i)=>b.setAttribute('aria-current',stage===i?'step':'false'));temperatures.forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.temperature)===T)));document.querySelector('#previous').disabled=stage===0;document.querySelector('#next').disabled=stage===stages.length-1;}
  function choose(i){stage=i;progress=0;playing=false;previous=null;render();}
  buttons.forEach(b=>b.addEventListener('click',()=>choose(Number(b.dataset.stage))));
  temperatures.forEach(b=>b.addEventListener('click',()=>{T=Number(b.dataset.temperature);choose(2);}));
  document.querySelector('#previous').addEventListener('click',()=>choose(Math.max(0,stage-1)));
  document.querySelector('#next').addEventListener('click',()=>choose(Math.min(stages.length-1,stage+1)));
  document.querySelector('#reset').addEventListener('click',()=>{T=1;choose(0);});
  document.querySelector('#play').addEventListener('click',()=>{if(!playing&&stage===stages.length-1&&progress===1){stage=0;progress=0;}playing=!playing;previous=null;render();});
  function tick(now){if(playing){if(previous!==null)progress+=(now-previous)/2500;if(progress>=1){progress=0;stage++;if(stage===stages.length){stage=0;}}render();}previous=now;requestAnimationFrame(tick);}
  document.addEventListener('visibilitychange',()=>{previous=null;});render();requestAnimationFrame(tick);
})(globalThis);
