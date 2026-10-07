/* One adapted projection from the pinned Kev rank-16 LoRA configuration. */
(function(global){
  'use strict';
  const C={ink:'#111827',muted:'#5e6e80',teal:'#167985',orange:'#b36b17',gray:'#dce4e8',light:'#f5f7f9',tealFill:'#eff8f8',orangeFill:'#fff4e4'};
  const rects=[
    {id:'base',x:192,y:114,w:320,h:57,fill:C.tealFill,stroke:C.teal},
    {id:'a',x:176,y:218,w:106,h:54,fill:C.orangeFill,stroke:C.orange},
    {id:'b',x:328,y:218,w:106,h:54,fill:C.orangeFill,stroke:C.orange},
    {id:'scale',x:466,y:218,w:84,h:54,fill:C.orangeFill,stroke:C.orange}
  ];
  const circles=[{id:'sum',x:602,y:173,w:38,h:38,fill:'white',stroke:C.muted}];
  const lines=[
    [87,192,137,192,C.teal,false],[137,142.5,137,245,C.teal,false],
    [137,142.5,190,142.5,C.teal,true],[512,142.5,621,142.5,C.teal,false],[621,142.5,621,171,C.teal,true],
    [137,245,174,245,C.orange,true],[282,245,326,245,C.orange,true],[434,245,464,245,C.orange,true],
    [550,245,621,245,C.orange,false],[621,245,621,213,C.orange,true],[640,192,675,192,C.teal,true]
  ];
  const texts=[
    {id:'title',x:36,y:25,w:648,h:42,size:30,bold:true,text:'LoRA inside Kev'},
    {id:'subtitle',x:36,y:76,w:648,h:25,size:17,text:'Small learned matrices adapt each selected projection'},
    {id:'v',x:56,y:175,w:30,h:30,size:24,formula:true,text:'v'},
    {id:'y',x:677,y:175,w:25,h:30,size:24,formula:true,text:'y'},
    {id:'base',x:192,y:119,w:320,h:27,size:20,bold:true,color:C.teal,center:true,text:'Base W₀ stays fixed'},
    {id:'base-dim',x:192,y:147,w:320,h:20,size:14,formula:true,color:C.muted,center:true,text:'dout × din',subs:[[1,4],[8,10]]},
    {id:'a',x:176,y:219,w:106,h:27,size:22,formula:true,color:C.orange,center:true,text:'A'},
    {id:'a-dim',x:176,y:249,w:106,h:20,size:14,formula:true,color:C.orange,center:true,text:'r × din',subs:[[5,7]]},
    {id:'b',x:328,y:219,w:106,h:27,size:22,formula:true,color:C.orange,center:true,text:'B'},
    {id:'b-dim',x:328,y:249,w:106,h:20,size:14,formula:true,color:C.orange,center:true,text:'dout × r',subs:[[1,4]]},
    {id:'scale',x:466,y:230,w:84,h:28,size:20,formula:true,color:C.orange,center:true,text:'α/r'},
    {id:'train-label',x:176,y:191,w:374,h:23,size:15,bold:true,color:C.orange,text:'LoRA matrices A and B train'},
    {id:'sum',x:602,y:174,w:38,h:35,size:25,formula:true,center:true,text:'+'},
    {id:'projection-formula',x:36,y:286,w:390,h:30,size:23,formula:true,text:'y = W₀v + (α/r)BAv'},
    {id:'update-formula',x:459,y:286,w:225,h:30,size:22,formula:true,text:'ΔW = (α/r)BA'},
    {id:'config',x:36,y:322,w:648,h:24,size:16,text:'r = 16     α = 32     α/r = 2     LoRA dropout = 0.05'},
    {id:'trainable',x:36,y:348,w:648,h:22,size:14,text:'33.8M trainable parameters across LoRA and the pointer head'},
    {id:'symbols',x:36,y:369,w:648,h:18,size:12,color:C.muted,text:'v: layer input    y: layer output    din / dout: input / output widths',subs:[[38,40],[44,47]]},
    {id:'footer',x:36,y:391,w:615,h:13,size:9,color:C.muted,text:'One adapted projection. Kev uses all targets. Open animation controls.'},
    {id:'number',x:665,y:391,w:19,h:13,size:9,color:C.muted,text:'3'}
  ];
  const stages=[
    {title:'1. Follow a layer input',text:'v is an activation entering one selected projection inside Qwen. It is different from x, the complete token sequence on the previous slide. y is the output of this projection.'},
    {title:'2. Keep the pretrained projection',text:'W₀ maps d_in input features to d_out output features. Its weights stay fixed during adapter training.'},
    {title:'3. Learn a small intermediate space',text:'A has r rows and d_in columns. It maps the input to r features. In this lab, r is 16.'},
    {title:'4. Map back to the output width',text:'B has d_out rows and r columns. It maps those r features to the output width. The learned matrices produce BAv.'},
    {title:'5. Scale and add the learned contribution',text:'The LoRA contribution is multiplied by α/r, which is 32/16 = 2 here. Adding it to W₀v gives y. The corresponding weight update is ΔW = (α/r)BA.'},
    {title:'6. Train adapters across the backbone',text:'Kev trains these matrices across attention, DeltaNet and feed-forward projections, together with the pointer head. The lab has about 33.8 million trainable parameters. Backpropagation still runs through the fixed backbone.'}
  ];
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
  const highlight=(r,t,c=C.orange)=>`<rect x="${r.x-3}" y="${r.y-3}" width="${r.w+6}" height="${r.h+6}" rx="3" fill="none" stroke="${c}" stroke-width="${1.2+Math.sin(t*Math.PI)*1.4}"/>`;
  function overlay(stage,t){
    if(stage===0)return `<circle cx="71" cy="192" r="21" fill="none" stroke="${C.teal}" stroke-width="${1.2+Math.sin(t*Math.PI)*1.4}"/>`;
    if(stage<4)return highlight(rects[stage-1],t,stage===1?C.teal:C.orange);
    if(stage===4)return highlight(rects[3],t)+highlight(circles[0],t,C.teal);
    return rects.slice(1,3).map(r=>highlight(r,t)).join('');
  }
  function rich(v){let p=0,s='';for(const [a,b] of v.subs||[]){s+=esc(v.text.slice(p,a))+`<tspan baseline-shift="sub" font-size="${v.size*.7}">${esc(v.text.slice(a,b))}</tspan>`;p=b;}return s+esc(v.text.slice(p));}
  function diagram(){
    let s=`<defs><marker id="arrow-teal" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="${C.teal}"/></marker><marker id="arrow-orange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="${C.orange}"/></marker></defs>`;
    s+=rects.map(r=>`<rect x="${r.x}" y="${r.y}" width="${r.w}" height="${r.h}" rx="3" fill="${r.fill}" stroke="${r.stroke}"/>`).join('');
    s+=circles.map(r=>`<ellipse cx="${r.x+r.w/2}" cy="${r.y+r.h/2}" rx="${r.w/2}" ry="${r.h/2}" fill="${r.fill}" stroke="${r.stroke}"/>`).join('');
    s+=lines.map(([x1,y1,x2,y2,c,arrow])=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${c}" stroke-width="1.2"${arrow?` marker-end="url(#arrow-${c===C.orange?'orange':'teal'})"`:''}/>`).join('');
    s+=texts.map(v=>`<text x="${v.center?v.x+v.w/2:v.x}" y="${v.y+v.size}" font-family="${v.formula?'Georgia':'Arial'}" font-size="${v.size}" font-weight="${v.bold?'700':'400'}" fill="${v.color||C.ink}"${v.center?' text-anchor="middle"':''}>${rich(v)}</text>`).join('');
    return s;
  }
  const api={C,rects,circles,lines,texts,stages,overlay,diagram};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  global.KevLora=api;
  if(typeof document==='undefined')return;
  const svg=document.querySelector('#lora');if(!svg)return;
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
