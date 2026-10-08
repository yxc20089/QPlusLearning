/* The lab's pinned general curriculum and completed native task lineage. */
(function(global){
  'use strict';
  const C={ink:'#111827',muted:'#5e6e80',teal:'#167985',orange:'#b36b17',gray:'#dce4e8',tealFill:'#eff8f8',orangeFill:'#fff4e4'};
  const rects=[
    {id:'initial',x:36,y:129,w:150,h:48,fill:C.tealFill,stroke:C.teal},
    {id:'dates',x:202,y:129,w:150,h:48,fill:C.tealFill,stroke:C.teal},
    {id:'documents',x:368,y:129,w:150,h:48,fill:C.tealFill,stroke:C.teal},
    {id:'skills',x:534,y:129,w:150,h:48,fill:C.tealFill,stroke:C.teal},
    {id:'baseline',x:36,y:233,w:150,h:48,fill:C.tealFill,stroke:C.teal},
    {id:'v2',x:286,y:233,w:150,h:48,fill:C.orangeFill,stroke:C.orange},
    {id:'v3',x:534,y:233,w:150,h:48,fill:C.orangeFill,stroke:C.orange}
  ];
  const lines=[[189,153,199,153,C.muted,true],[355,153,365,153,C.muted,true],[521,153,531,153,C.muted,true],[190,257,281,257,C.muted,true],[440,257,529,257,C.muted,true]];
  const phi='φchild0 = φparent',data='Dtrain = Dnew ⊎ Dreplay';
  const texts=[
    {id:'title',x:36,y:25,w:648,h:42,size:30,bold:true,text:'A curriculum, then task checkpoints'},
    {id:'subtitle',x:36,y:76,w:648,h:25,size:17,text:'Each new stage loads its parent’s LoRA and pointer head.'},
    {id:'general-label',x:36,y:108,w:648,h:20,size:12,color:C.muted,text:'General decision curriculum — prework'},
    {id:'initial-label',x:36,y:140,w:150,h:27,size:17,center:true,text:'Initial decisions'},
    {id:'dates-label',x:202,y:140,w:150,h:27,size:17,center:true,text:'Dates + evidence'},
    {id:'documents-label',x:368,y:140,w:150,h:27,size:17,center:true,text:'Documents'},
    {id:'skills-label',x:534,y:140,w:150,h:27,size:17,center:true,text:'Skills + tools'},
    {id:'initial-count',x:36,y:183,w:150,h:19,size:11,color:C.muted,center:true,text:'12,576 requests × 2 epochs'},
    {id:'dates-count',x:202,y:183,w:150,h:19,size:11,color:C.muted,center:true,text:'1,425 + 2,000 replay'},
    {id:'documents-count',x:368,y:183,w:150,h:19,size:11,color:C.muted,center:true,text:'5,219 + 2,000 replay'},
    {id:'skills-count',x:534,y:183,w:150,h:19,size:11,color:C.muted,center:true,text:'11,320 + 4,000 replay'},
    {id:'task-label',x:36,y:211,w:648,h:20,size:12,color:C.muted,text:'Skills becomes the baseline for task adaptation'},
    {id:'baseline-label',x:36,y:244,w:150,h:27,size:17,center:true,text:'Skills baseline'},
    {id:'v2-label',x:286,y:244,w:150,h:27,size:17,center:true,text:'Pac-Man v2'},
    {id:'v3-label',x:534,y:244,w:150,h:27,size:17,center:true,text:'Pac-Man v3'},
    {id:'baseline-caption',x:36,y:287,w:162,h:20,size:11,color:C.muted,text:'No Pac-Man labels yet'},
    {id:'v2-caption',x:286,y:287,w:188,h:20,size:11,color:C.muted,text:'Qualified teacher labels'},
    {id:'v3-caption',x:520,y:287,w:178,h:20,size:11,color:C.muted,center:true,text:'Corrections on v2 states'},
    {id:'phi',x:36,y:321,w:276,h:31,size:22,formula:true,text:phi,subs:[[1,6],[11,17]],supers:[[6,7]]},
    {id:'data',x:340,y:321,w:344,h:31,size:22,formula:true,text:data,subs:[[1,6],[10,13],[17,23]]},
    {id:'handoff-caption',x:36,y:357,w:648,h:19,size:12,color:C.muted,text:'New stage: fresh optimizer. Parent checkpoint stays saved.'},
    {id:'footer',x:36,y:383,w:615,h:17,size:9,color:C.muted,text:'Same fixed Qwen base. Counts name requests; labels are not game wins. Open animation controls.'},
    {id:'number',x:665,y:383,w:19,h:17,size:9,color:C.muted,text:'7'}
  ];
  const stages=[
    {title:'1. Start from pretrained Qwen',text:'The Qwen3.5-4B-Base language weights already exist and stay fixed. Initial decision training creates a fresh LoRA adapter and pointer head, then trains them on 12,576 generic requests for two epochs. This is decision-head/adaptor training, not pretraining the language model again.'},
    {title:'2. Add dates and missing-evidence cases',text:'The dates stage loads the initial LoRA and head, then trains on 1,425 new requests and 2,000 generic replay requests for one epoch. The examples provide day counts and cases where deciding evidence is absent. They do not make Qwen reliably subtract arbitrary dates.'},
    {title:'3. Add document decisions',text:'The document stage starts from dates and learns from 5,219 labeled real consumer-finance complaints, mixed with 2,000 generic replay requests. It saves a separate documents checkpoint after one epoch.'},
    {title:'4. Add skills and developer tools',text:'The skills stage starts from documents. Our pinned lab file concatenates 6,000 hard-v1 training requests and 5,320 devtools-v1 training requests, then adds 4,000 generic replay requests. The completed Skills adapter is the general baseline; it has no Pac-Man task labels.'},
    {title:'5. Adapt Skills to Pac-Man',text:'Native v2 starts from the completed Skills LoRA/head. It learns from 4,096 qualified teacher requests plus 2,000 generic replay requests. A new optimizer and schedule train this stage for one epoch. Its parent stays saved and playable.'},
    {title:'6. Correct the learner’s own states',text:'Native v3 starts from native v2. Its 4,096 task requests mix 2,048 original v2 requests and 2,048 verified correction/recovery requests, plus 2,000 generic replay requests. The teacher labels v2-visited states and verified winning continuations. A changed checkpoint is an experiment, not proof of better gameplay.'},
    {title:'7. Understand the handoff',text:'φ denotes every trainable LoRA and pointer-head parameter. Superscript zero means the child’s starting values: a new stage copies its parent’s parameters, then updates them on D_train, the new requests together with replay requests. Warm-starting a new stage creates a fresh optimizer. Recovery resumes the interrupted stage with its optimizer and step position.'}
  ];
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
  const hl=(r,t,c=C.teal)=>`<rect x="${r.x-3}" y="${r.y-3}" width="${r.w+6}" height="${r.h+6}" rx="3" fill="none" stroke="${c}" stroke-width="${1.2+Math.sin(t*Math.PI)*1.4}"/>`;
  function overlay(stage,t){if(stage<4)return hl(rects[stage],t);if(stage===4)return hl(rects[4],t)+hl(rects[5],t,C.orange);if(stage===5)return hl(rects[5],t,C.orange)+hl(rects[6],t,C.orange);return hl({x:33,y:317,w:654,h:56},t,C.teal);}
  function rich(v){const spans=[...(v.subs||[]).map(([a,b])=>({a,b,shift:'sub'})),...(v.supers||[]).map(([a,b])=>({a,b,shift:'super'}))].sort((a,b)=>a.a-b.a);let p=0,s='';for(const q of spans){s+=esc(v.text.slice(p,q.a))+`<tspan baseline-shift="${q.shift}" font-size="${v.size*.7}">${esc(v.text.slice(q.a,q.b))}</tspan>`;p=q.b;}return s+esc(v.text.slice(p));}
  function diagram(){return `<defs><marker id="arrow-muted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0L10 5L0 10Z" fill="${C.muted}"/></marker></defs>`+rects.map(r=>`<rect x="${r.x}" y="${r.y}" width="${r.w}" height="${r.h}" rx="3" fill="${r.fill}" stroke="${r.stroke}"/>`).join('')+lines.map(([x1,y1,x2,y2,c,arrow])=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${c}" stroke-width="1.2"${arrow?' marker-end="url(#arrow-muted)"':''}/>`).join('')+texts.map(v=>`<text x="${v.center?v.x+v.w/2:v.x}" y="${v.y+v.size}" font-family="${v.formula?'Georgia':'Arial'}" font-size="${v.size}" font-weight="${v.bold?'700':'400'}" fill="${v.color||C.ink}"${v.center?' text-anchor="middle"':''}>${rich(v)}</text>`).join('');}
  const api={C,rects,lines,texts,stages,overlay,diagram};if(typeof module!=='undefined'&&module.exports)module.exports=api;global.KevCurriculum=api;if(typeof document==='undefined')return;const svg=document.querySelector('#curriculum');if(!svg)return;
  let stage=0,progress=0,playing=false,last=null;const buttons=[...document.querySelectorAll('[data-stage]')];
  function render(){svg.innerHTML=diagram()+`<g aria-hidden="true">${overlay(stage,progress)}</g>`;document.querySelector('#stage-title').textContent=stages[stage].title;document.querySelector('#explanation').textContent=stages[stage].text;document.querySelector('#play').textContent=playing?'Pause':'Play';document.querySelector('#play').setAttribute('aria-pressed',String(playing));buttons.forEach((b,i)=>b.setAttribute('aria-current',stage===i?'step':'false'));document.querySelector('#previous').disabled=stage===0;document.querySelector('#next').disabled=stage===stages.length-1;}
  function choose(i){stage=i;progress=0;playing=false;last=null;render();}buttons.forEach(b=>b.addEventListener('click',()=>choose(Number(b.dataset.stage))));document.querySelector('#previous').addEventListener('click',()=>choose(Math.max(0,stage-1)));document.querySelector('#next').addEventListener('click',()=>choose(Math.min(stages.length-1,stage+1)));document.querySelector('#reset').addEventListener('click',()=>choose(0));document.querySelector('#play').addEventListener('click',()=>{playing=!playing;last=null;render();});function tick(now){if(playing){if(last!==null)progress+=(now-last)/2500;if(progress>=1){progress=0;stage=(stage+1)%stages.length;}render();}last=now;requestAnimationFrame(tick);}document.addEventListener('visibilitychange',()=>{last=null;});render();requestAnimationFrame(tick);
})(globalThis);
