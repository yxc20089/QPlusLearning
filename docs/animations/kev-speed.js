/* Pinned Kev inference structure versus ordinary autoregressive text output. */
(function(global){
  'use strict';
  const C={ink:'#111827',muted:'#5e6e80',teal:'#167985',orange:'#b36b17',gray:'#dce4e8',light:'#f5f7f9',tealFill:'#eff8f8',orangeFill:'#fff4e4'};
  const rects=[
    {id:'state',x:120,y:115,w:115,h:50,fill:C.tealFill,stroke:C.teal},
    {id:'branch',x:265,y:115,w:132,h:50,fill:C.tealFill,stroke:C.teal},
    {id:'pointer',x:430,y:115,w:130,h:50,fill:C.orangeFill,stroke:C.orange},
    {id:'action',x:590,y:115,w:94,h:50,fill:C.orangeFill,stroke:C.orange},
    {id:'prompt',x:120,y:211,w:146,h:50,fill:C.tealFill,stroke:C.teal}
  ];
  const lines=[[237,140,260,140,C.muted,true],[400,140,425,140,C.muted,true],[562,140,585,140,C.muted,true]];
  const pointerFormula='RKev ≈ (K + 1) d dp + K dp';
  const texts=[
    {id:'title',x:36,y:25,w:648,h:42,size:30,bold:true,text:'Why Kev avoids the answer-token loop'},
    {id:'subtitle',x:36,y:76,w:648,h:25,size:17,text:'The backbone reads the input. The pointer head returns K scores.'},
    {id:'kev-label',x:36,y:128,w:71,h:27,size:16,bold:true,color:C.teal,text:'Kev'},
    {id:'state',x:120,y:131,w:115,h:27,size:16,center:true,text:'Read state'},
    {id:'branch-line1',x:265,y:120,w:132,h:24,size:15,center:true,text:'Question +'},
    {id:'branch-line2',x:265,y:139,w:132,h:24,size:15,center:true,text:'all options'},
    {id:'pointer-line1',x:430,y:120,w:130,h:24,size:15,center:true,text:'K option'},
    {id:'pointer-line2',x:430,y:139,w:130,h:24,size:15,center:true,text:'scores'},
    {id:'action',x:590,y:130,w:94,h:30,size:19,bold:true,color:C.orange,center:true,text:'Right'},
    {id:'kev-caption',x:120,y:173,w:298,h:21,size:13,color:C.muted,text:'Cold Qwen3.5: state pass + question row'},
    {id:'output-caption',x:430,y:173,w:254,h:21,size:13,color:C.muted,text:'No generated answer tokens'},
    {id:'text-label',x:36,y:216,w:74,h:42,size:16,bold:true,color:C.teal,text:'Text\noutput'},
    {id:'prompt',x:120,y:227,w:146,h:27,size:16,center:true,text:'Read prompt'},
    {id:'prefill-caption',x:291,y:269,w:90,h:21,size:13,color:C.muted,center:true,text:'From prefill'},
    {id:'decode-caption',x:382,y:269,w:302,h:21,size:13,color:C.orange,center:true,text:'M − 1 extra sequential decode passes'},
    {id:'pointer-cost',x:36,y:308,w:453,h:33,size:23,formula:true,text:pointerFormula,subs:[[1,4],...Array.from(pointerFormula.matchAll(/dp/g),m=>[m.index+1,m.index+2])]},
    {id:'text-cost',x:36,y:344,w:453,h:33,size:23,formula:true,text:'Rtext ≈ M d V',subs:[[1,5]]},
    {id:'cost-caption',x:510,y:308,w:174,h:23,size:13,color:C.muted,text:'Readout multiplications'},
    {id:'dimension-caption',x:510,y:331,w:174,h:22,size:13,color:C.muted,text:'K choices, V vocabulary'},
    {id:'footer',x:36,y:383,w:615,h:17,size:9,color:C.muted,text:'Illustrative M = 4 answer. Counts exclude backbone work. Open animation controls.'},
    {id:'number',x:665,y:383,w:19,h:17,size:9,color:C.muted,text:'6'}
  ];
  function tokens(M=4){const w=M<=4?68:40,start=M<=4?302:286,step=M<=4?93:50;return Array.from({length:M},(_,i)=>({id:'token-'+(i+1),x:start+step*i,y:211,w,h:50,fill:i===0?C.tealFill:C.orangeFill,stroke:i===0?C.teal:C.orange}));}
  function dynamicTexts(M=4){const ts=tokens(M);return texts.map(v=>v.id==='decode-caption'?{...v,text:M===1?'M = 1: no extra decode pass':`M = ${M}: ${M-1} extra decode passes`}:v.id==='prefill-caption'?{...v,x:ts[0].x+ts[0].w/2-v.w/2}:v.id==='footer'?{...v,text:`Illustrative M = ${M} answer. Counts exclude backbone work. Open animation controls.`}:v).concat(ts.map((r,i)=>({id:'token-label-'+(i+1),x:r.x,y:225,w:r.w,h:31,size:M<=4?23:19,formula:true,center:true,color:i===0?C.teal:C.orange,text:'y'+(i+1),subs:[[1,String(i+1).length+1]]})));}
  function dynamicLines(M=4){const ts=tokens(M);return lines.concat([[268,236,ts[0].x-5,236,C.muted,true]],ts.slice(1).map((r,i)=>[ts[i].x+ts[i].w+2,236,r.x-5,236,C.muted,true]));}
  const stages=[
    {title:'1. Read the state',text:'For a new state, Qwen3.5 serving computes the state prefix. This is real backbone work. The drawing shows the cold single-question path used for a Pac-Man move, without claiming the whole request is one backbone call.'},
    {title:'2. Read the question and every option',text:'One question row continues from that state. It contains the instruction and all K options. The backbone reads all of them in this row; it does not run a separate backbone pass for each option.'},
    {title:'3. Return the decision directly',text:'The pointer head projects the decide and option readouts, produces K scores, and softmax plus argmax selects Right in this illustration. Kev generates no answer tokens. The API can still serialize this decision as JSON.'},
    {title:'4. Compare with text generation',text:'A conventional decoder reads its prompt in a prefill pass. That pass provides the distribution for the first answer token, y_1. M is the total number of answer tokens, and y_m means token number m.'},
    {title:'5. Follow the sequential output',text:'Each later answer token needs the previous generated tokens. An M-token answer therefore adds M−1 sequential decode passes after prefill. The illustration uses M = 4, so three extra passes follow the first token.'},
    {title:'6. Compare the readouts',text:'R counts approximate readout multiplications after hidden states are available. The pointer performs one d×d_p query projection, K key projections and K dot products. A conventional dense vocabulary head performs d×V multiplications for each of M tokens. Backbone work, biases, scaling and normalization are outside these sketches; this is not a GPU timing benchmark.'}
  ];
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
  const highlight=(r,t,c=C.orange)=>`<rect x="${r.x-3}" y="${r.y-3}" width="${r.w+6}" height="${r.h+6}" rx="3" fill="none" stroke="${c}" stroke-width="${1.2+Math.sin(t*Math.PI)*1.4}"/>`;
  function overlay(stage,t,M=4){const ts=tokens(M);if(stage===0)return highlight(rects[0],t,C.teal);if(stage===1)return highlight(rects[1],t,C.teal);if(stage===2)return highlight(rects[2],t,C.orange)+highlight(rects[3],t,C.orange);if(stage===3)return highlight(rects[4],t,C.teal)+highlight(ts[0],t,C.teal);if(stage===4){const i=M===1?0:1+Math.min(M-2,Math.floor(t*(M-1)));return highlight(ts[i],t,M===1?C.teal:C.orange);}return highlight({x:33,y:305,w:657,h:69},t,C.teal);}
  function rich(v){let p=0,s='';for(const [a,b] of v.subs||[]){s+=esc(v.text.slice(p,a))+`<tspan baseline-shift="sub" font-size="${v.size*.7}">${esc(v.text.slice(a,b))}</tspan>`;p=b;}return s+esc(v.text.slice(p));}
  function diagram(M=4){let s=`<defs><marker id="arrow-muted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="${C.muted}"/></marker></defs>`;s+=[...rects,...tokens(M)].map(r=>`<rect x="${r.x}" y="${r.y}" width="${r.w}" height="${r.h}" rx="3" fill="${r.fill}" stroke="${r.stroke}"/>`).join('');s+=dynamicLines(M).map(([x1,y1,x2,y2,c,arrow])=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${c}" stroke-width="1.2"${arrow?' marker-end="url(#arrow-muted)"':''}/>`).join('');s+=dynamicTexts(M).map(v=>v.text.split('\n').map((line,i)=>`<text x="${v.center?v.x+v.w/2:v.x}" y="${v.y+v.size+i*v.size*1.2}" font-family="${v.formula?'Georgia':'Arial'}" font-size="${v.size}" font-weight="${v.bold?'700':'400'}" fill="${v.color||C.ink}"${v.center?' text-anchor="middle"':''}>${rich({...v,text:line})}</text>`).join('')).join('');return s;}
  const api={C,rects,lines,texts,tokens,dynamicTexts,dynamicLines,stages,overlay,diagram};if(typeof module!=='undefined'&&module.exports)module.exports=api;global.KevSpeed=api;if(typeof document==='undefined')return;const svg=document.querySelector('#speed');if(!svg)return;
  let stage=0,progress=0,playing=false,previous=null,M=4;const buttons=[...document.querySelectorAll('[data-stage]')],lengths=[...document.querySelectorAll('[data-length]')];
  function render(){svg.innerHTML=diagram(M)+`<g aria-hidden="true">${overlay(stage,progress,M)}</g>`;document.querySelector('#stage-title').textContent=stages[stage].title;document.querySelector('#explanation').textContent=stage===4?(M===1?'With M = 1, prefill provides the single answer token. There is no extra decode pass. A constrained single-token classification baseline therefore narrows this comparison. A custom classification head can also avoid a dense vocabulary readout.':`With M = ${M}, the first answer token comes from prefill. The remaining ${M-1} tokens require ${M-1} extra sequential decode passes. These passes can reuse the decoder cache, but each later token still depends on earlier generated tokens.`):stages[stage].text;document.querySelector('#play').textContent=playing?'Pause':'Play';document.querySelector('#play').setAttribute('aria-pressed',String(playing));buttons.forEach((b,i)=>b.setAttribute('aria-current',stage===i?'step':'false'));lengths.forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.length)===M)));document.querySelector('#previous').disabled=stage===0;document.querySelector('#next').disabled=stage===stages.length-1;}
  function choose(i){stage=i;progress=0;playing=false;previous=null;render();}buttons.forEach(b=>b.addEventListener('click',()=>choose(Number(b.dataset.stage))));lengths.forEach(b=>b.addEventListener('click',()=>{M=Number(b.dataset.length);choose(4);}));document.querySelector('#previous').addEventListener('click',()=>choose(Math.max(0,stage-1)));document.querySelector('#next').addEventListener('click',()=>choose(Math.min(stages.length-1,stage+1)));document.querySelector('#reset').addEventListener('click',()=>{M=4;choose(0);});document.querySelector('#play').addEventListener('click',()=>{playing=!playing;previous=null;render();});function tick(now){if(playing){if(previous!==null)progress+=(now-previous)/2500;if(progress>=1){progress=0;stage=(stage+1)%stages.length;}render();}previous=now;requestAnimationFrame(tick);}document.addEventListener('visibilitychange',()=>{previous=null;});render();requestAnimationFrame(tick);
})(globalThis);
