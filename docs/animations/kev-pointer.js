/* Exact affine projections and scaled dot product from pinned Kev PointerHead. */
(function(global){
  'use strict';
  const C={ink:'#111827',muted:'#5e6e80',teal:'#167985',orange:'#b36b17',gray:'#dce4e8',light:'#f5f7f9',tealFill:'#eff8f8',orangeFill:'#fff4e4'};
  const rects=[
    {id:'decide',x:36,y:130,w:105,h:40,fill:C.tealFill,stroke:C.teal},
    {id:'option',x:36,y:230,w:105,h:40,fill:C.orangeFill,stroke:C.orange},
    {id:'query',x:177,y:124,w:225,h:52,fill:C.tealFill,stroke:C.teal},
    {id:'key',x:177,y:224,w:225,h:52,fill:C.orangeFill,stroke:C.orange},
    {id:'scale',x:576,y:179,w:52,h:36,fill:C.light,stroke:C.muted}
  ];
  const circles=[{id:'dot',x:509,y:177,w:40,h:40,fill:'#ffffff',stroke:C.muted}];
  const lines=[
    [141,150,175,150,C.teal,true],[402,150,433,150,C.teal,true],
    [470,150,529,150,C.teal,false],[529,150,529,175,C.teal,true],
    [141,250,175,250,C.orange,true],[402,250,433,250,C.orange,true],
    [470,250,529,250,C.orange,false],[529,250,529,219,C.orange,true],
    [551,197,574,197,C.muted,true],[628,197,655,197,C.muted,true]
  ];
  const texts=[
    {id:'title',x:36,y:25,w:648,h:42,size:30,bold:true,text:'The pointer head scores every option'},
    {id:'subtitle',x:36,y:76,w:648,h:25,size:17,text:'Two learned linear projections, then a scaled dot product'},
    {id:'decide-label',x:36,y:106,w:142,h:21,size:13,bold:true,color:C.teal,text:'Final <decide> vector'},
    {id:'option-label',x:36,y:204,w:164,h:21,size:13,bold:true,color:C.orange,text:'Option i: closing </opt>'},
    {id:'decide',x:36,y:131,w:105,h:35,size:23,formula:true,color:C.teal,center:true,text:'hdecide',subs:[[1,7]]},
    {id:'option',x:36,y:231,w:105,h:35,size:23,formula:true,color:C.orange,center:true,text:'hend,i',subs:[[1,6]]},
    {id:'query',x:177,y:135,w:225,h:34,size:20,formula:true,color:C.teal,center:true,text:'q = Wq hdecide + bq',subs:[[5,6],[8,14],[18,19]]},
    {id:'key',x:177,y:235,w:225,h:34,size:20,formula:true,color:C.orange,center:true,text:'ki = Wk hend,i + bk',subs:[[1,2],[6,7],[9,14],[18,19]]},
    {id:'q',x:436,y:131,w:36,h:34,size:24,formula:true,color:C.teal,center:true,text:'q'},
    {id:'k',x:436,y:231,w:36,h:34,size:24,formula:true,color:C.orange,center:true,text:'ki',subs:[[1,2]]},
    {id:'query-dim',x:177,y:180,w:225,h:20,size:13,color:C.muted,center:true,text:'2,560 features → 256 features'},
    {id:'shared',x:177,y:280,w:225,h:21,size:13,color:C.muted,center:true,text:'The same key layer for every option'},
    {id:'dot',x:509,y:172,w:40,h:41,size:30,formula:true,center:true,text:'·'},
    {id:'scale',x:576,y:181,w:52,h:30,size:17,formula:true,center:true,text:'÷√dp',subs:[[3,4]]},
    {id:'scale-label',x:570,y:219,w:63,h:20,size:13,color:C.muted,center:true,text:'÷16 here'},
    {id:'z',x:658,y:181,w:29,h:36,size:24,formula:true,text:'zi',subs:[[1,2]]},
    {id:'score-label',x:641,y:219,w:54,h:20,size:13,color:C.muted,center:true,text:'Score'},
    {id:'score-formula',x:36,y:310,w:318,h:34,size:24,formula:true,text:'zi = qᵀki / √dp',subs:[[1,2],[8,9],[14,15]]},
    {id:'repeat',x:365,y:316,w:319,h:26,size:17,text:'K legal options → K raw scores'},
    {id:'config',x:36,y:349,w:648,h:23,size:14,text:'d = 2,560     dp = 256     √dp = 16     Both projection layers train',subs:[[15,16],[29,30]]},
    {id:'footer',x:36,y:383,w:615,h:17,size:9,color:C.muted,text:'Exact Kev pointer head. i indexes one option. Open animation controls.'},
    {id:'number',x:665,y:383,w:19,h:17,size:9,color:C.muted,text:'4'}
  ];
  const stages=[
    {title:'1. Pick the two readouts',text:'h_decide is the contextual vector at the final decide marker. h_end,i is the contextual vector at option i’s closing marker. Both come from the same encoded state, question and candidate sequence.'},
    {title:'2. Project the decision vector',text:'The learned affine layer q = W_q h_decide + b_q produces one query vector. W_q is its weight matrix and b_q is its bias. That same query is reused for every option in this question.'},
    {title:'3. Project an option vector',text:'The learned affine layer k_i = W_k h_end,i + b_k produces one key vector per option. W_k and b_k are shared across the options. Our Qwen3.5-4B readouts have 2,560 features; each projected vector has 256.'},
    {title:'4. Compare matching features',text:'The dot product qᵀk_i multiplies matching query and key entries, then adds those products. Kev uses this ordinary dot product; it does not normalize the vectors into a cosine similarity.'},
    {title:'5. Apply the fixed scale',text:'The score is z_i = qᵀk_i / √d_p. d_p is the pointer width, 256 here, so √d_p is 16. This is a raw score, before the probability calculation on the next slide.'},
    {title:'6. Score the supplied options',text:'Repeat the same key projection and scaled comparison for each of the K legal options. That produces K raw scores, such as one each for Left, Up and Right. Both pointer projection layers train together with the LoRA adapters.'}
  ];
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
  const highlight=(r,t,c=C.orange)=>`<rect x="${r.x-3}" y="${r.y-3}" width="${r.w+6}" height="${r.h+6}" rx="3" fill="none" stroke="${c}" stroke-width="${1.2+Math.sin(t*Math.PI)*1.4}"/>`;
  function overlay(stage,t){
    if(stage===0)return highlight(rects[0],t,C.teal)+highlight(rects[1],t,C.orange);
    if(stage===1)return highlight(rects[2],t,C.teal);
    if(stage===2)return highlight(rects[3],t,C.orange);
    if(stage===3)return `<ellipse cx="529" cy="197" rx="24" ry="24" fill="none" stroke="${C.teal}" stroke-width="${1.2+Math.sin(t*Math.PI)*1.4}"/>`;
    if(stage===4)return highlight(rects[4],t,C.teal);
    return highlight(rects[1],t,C.orange)+highlight(rects[3],t,C.orange)+highlight({x:652,y:177,w:38,h:38},t,C.teal);
  }
  function rich(v){let p=0,s='';for(const [a,b] of v.subs||[]){s+=esc(v.text.slice(p,a))+`<tspan baseline-shift="sub" font-size="${v.size*.7}">${esc(v.text.slice(a,b))}</tspan>`;p=b;}return s+esc(v.text.slice(p));}
  function diagram(){
    let s=`<defs>${[['teal',C.teal],['orange',C.orange],['muted',C.muted]].map(([name,c])=>`<marker id="arrow-${name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="${c}"/></marker>`).join('')}</defs>`;
    s+=rects.map(r=>`<rect x="${r.x}" y="${r.y}" width="${r.w}" height="${r.h}" rx="3" fill="${r.fill}" stroke="${r.stroke}"/>`).join('');
    s+=circles.map(r=>`<ellipse cx="${r.x+r.w/2}" cy="${r.y+r.h/2}" rx="${r.w/2}" ry="${r.h/2}" fill="${r.fill}" stroke="${r.stroke}"/>`).join('');
    s+=lines.map(([x1,y1,x2,y2,c,arrow])=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${c}" stroke-width="1.2"${arrow?` marker-end="url(#arrow-${c===C.orange?'orange':c===C.teal?'teal':'muted'})"`:''}/>`).join('');
    s+=texts.map(v=>`<text x="${v.center?v.x+v.w/2:v.x}" y="${v.y+v.size}" font-family="${v.formula?'Georgia':'Arial'}" font-size="${v.size}" font-weight="${v.bold?'700':'400'}" fill="${v.color||C.ink}"${v.center?' text-anchor="middle"':''}>${rich(v)}</text>`).join('');
    return s;
  }
  const api={C,rects,circles,lines,texts,stages,overlay,diagram};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  global.KevPointer=api;
  if(typeof document==='undefined')return;
  const svg=document.querySelector('#pointer');if(!svg)return;
  svg.innerHTML=diagram()+'<g id="flow" aria-hidden="true"></g>';
  let stage=0,progress=0,playing=false,previous=null;
  const buttons=[...document.querySelectorAll('[data-stage]')];
  function render(){document.querySelector('#flow').innerHTML=overlay(stage,progress);document.querySelector('#stage-title').textContent=stages[stage].title;document.querySelector('#explanation').textContent=stages[stage].text;document.querySelector('#play').textContent=playing?'Pause':'Play';document.querySelector('#play').setAttribute('aria-pressed',String(playing));buttons.forEach((b,i)=>b.setAttribute('aria-current',stage===i?'step':'false'));document.querySelector('#previous').disabled=stage===0;document.querySelector('#next').disabled=stage===stages.length-1;}
  function choose(i){stage=i;progress=0;playing=false;previous=null;render();}
  buttons.forEach(b=>b.addEventListener('click',()=>choose(Number(b.dataset.stage))));
  document.querySelector('#previous').addEventListener('click',()=>choose(Math.max(0,stage-1)));
  document.querySelector('#next').addEventListener('click',()=>choose(Math.min(stages.length-1,stage+1)));
  document.querySelector('#reset').addEventListener('click',()=>choose(0));
  document.querySelector('#play').addEventListener('click',()=>{if(!playing&&stage===stages.length-1&&progress===1){stage=0;progress=0;}playing=!playing;previous=null;render();});
  function tick(now){if(playing){if(previous!==null)progress+=(now-previous)/2500;if(progress>=1){progress=0;stage++;if(stage===stages.length){stage--;progress=1;playing=false;}}render();}previous=now;requestAnimationFrame(tick);}
  document.addEventListener('visibilitychange',()=>{previous=null;});render();requestAnimationFrame(tick);
})(globalThis);
