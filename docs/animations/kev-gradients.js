/* Pinned Kev choice-loss gradient. Probabilities illustrate one label, not a model run. */
(function(global){
  'use strict';
  const C={ink:'#111827',muted:'#5e6e80',teal:'#167985',orange:'#b36b17',tealFill:'#eff8f8',orangeFill:'#fff4e4'};
  const rects=[
    {id:'backbone',x:104,y:188,w:238,h:94,fill:C.tealFill,stroke:C.teal},
    {id:'head',x:386,y:188,w:156,h:94,fill:C.orangeFill,stroke:C.orange},
    {id:'loss',x:588,y:188,w:96,h:94,fill:'#ffffff',stroke:'#b5c4cc'}
  ];
  const lines=[
    [75,225,104,225,C.teal,true],[342,225,386,225,C.teal,true],[542,225,588,225,C.teal,true],
    [636,282,636,300,C.orange,false],[636,300,464,300,C.orange,false],[464,300,464,282,C.orange,true],
    [386,265,342,265,C.orange,true]
  ];
  function math(parts){let text='',subs=[];for(const part of parts){if(typeof part==='string')text+=part;else{const start=text.length;text+=part.sub;subs.push([start,text.length]);}}return {text,subs};}
  const texts=[
    {id:'title',x:36,y:25,w:648,h:42,size:30,bold:true,text:'How gradients update Kev'},
    {id:'subtitle',x:36,y:76,w:648,h:24,size:16,text:'The loss flows backward. LoRA and the pointer head learn.'},
    {id:'score-gradient',x:36,y:112,w:304,h:34,size:25,formula:true,...math(['∂ℒ/∂z',{sub:'i'},' = p',{sub:'i'},' − t',{sub:'i'}])},
    {id:'gradient-values',x:357,y:127,w:327,h:19,size:14.5,color:C.orange,text:'Up +0.35   Left −0.80   Right +0.45'},
    {id:'target',x:36,y:151,w:648,h:18,size:12.5,color:C.muted,...math(['t',{sub:'i'},' = 1 for the teacher action, 0 otherwise. Values are illustrative.'])},
    {id:'input',x:52,y:211,w:23,h:31,size:25,formula:true,text:'x'},
    {id:'backbone-title',x:116,y:197,w:214,h:21,size:14,bold:true,text:'Qwen backbone'},
    {id:'base',x:116,y:224,w:214,h:22,size:16,color:C.teal,text:'W₀ stays fixed'},
    {id:'lora',x:116,y:250,w:214,h:23,size:16,color:C.orange,...math(['LoRA A',{sub:'ℓ'},', B',{sub:'ℓ'},' train'])},
    {id:'hidden',x:351,y:197,w:25,h:27,size:20,formula:true,text:'H'},
    {id:'head-title',x:398,y:197,w:132,h:21,size:14,bold:true,color:C.orange,text:'Pointer head'},
    {id:'head-query',x:398,y:223,w:132,h:22,size:16,formula:true,...math(['W',{sub:'q'},', b',{sub:'q'}])},
    {id:'head-key',x:398,y:249,w:132,h:22,size:16,formula:true,...math(['W',{sub:'k'},', b',{sub:'k'}])},
    {id:'scores',x:555,y:197,w:25,h:27,size:20,formula:true,text:'z'},
    {id:'loss-title',x:594,y:197,w:84,h:18,size:11.5,bold:true,text:'Cross-entropy'},
    {id:'loss-symbol',x:598,y:220,w:76,h:29,size:23,formula:true,center:true,text:'ℒ'},
    {id:'teacher',x:598,y:254,w:76,h:18,size:11.5,text:'teacher: Left'},
    {id:'frozen-gradient',x:104,y:304,w:272,h:20,size:12.5,color:C.muted,text:'Gradients pass through frozen layers'},
    {id:'backward',x:386,y:304,w:298,h:20,size:12.5,color:C.orange,text:'Backpropagation'},
    {id:'trainable',x:36,y:326,w:648,h:30,size:22,formula:true,...math(['φ = {A',{sub:'ℓ'},', B',{sub:'ℓ'},', W',{sub:'q'},', b',{sub:'q'},', W',{sub:'k'},', b',{sub:'k'},'}'])},
    {id:'optimizer',x:36,y:358,w:648,h:18,size:12.5,text:'Batch 4 × accum 2     Clip norm 1.0     Fused AdamW     peak lr 2e−5'},
    {id:'footer',x:36,y:383,w:632,h:14,size:9,color:C.muted,text:'Single-choice gradient. Frozen base. Open animation controls.'},
    {id:'number',x:678,y:383,w:20,h:14,size:9,color:C.muted,text:'12'}
  ];
  const stages=[
    {title:'1. Differentiate the action scores',text:'For the same Left target as slide 11, the one-choice cross-entropy derivative is ∂ℒ/∂z_i = p_i − t_i. z_i is a raw score, p_i its softmax probability, and t_i a one-hot target: one for Left and zero for other legal actions. The illustrative gradient is +0.35 for Up, −0.80 for Left and +0.45 for Right. A downhill direction in score space favors Left. These are neither measured checkpoint predictions nor parameter updates.'},
    {title:'2. Backpropagate through the pointer head',text:'The orange return path enters the pointer head. Autograd differentiates the dot products and both affine projections. W_q and b_q are the query projection’s matrix and bias; W_k and b_k belong to the key projection. These parameters learn, and gradients also reach the decide-marker and option-marker hidden vectors inside H. The pointer head is separate from Qwen’s attention projections.'},
    {title:'3. Pass through the fixed backbone',text:'A fixed weight can still transmit an activation gradient. W₀ keeps its pretrained values while the chain rule carries the loss through the backbone to earlier adapters. This schematic backbone includes many layers, not one detached feature extractor. Gradient checkpointing recomputes selected activations during backward to save memory; it does not prevent LoRA from learning.'},
    {title:'4. Differentiate every selected LoRA',text:'A_ℓ and B_ℓ are the two LoRA matrices in adapted projection ℓ, pronounced “ell.” The lab uses rank 16 and scaling α/r = 32/16 = 2. Backpropagation differentiates their low-rank contribution and accumulates gradients for both matrices. The all-targets setting includes attention, DeltaNet and feed-forward projections. The pretrained base matrices and original token embeddings remain fixed in this lab.'},
    {title:'5. Accumulate, clip and update φ',text:'Phi, φ, collects all adapter matrices across ℓ together with the pointer-head matrices and biases. Our full eight-request update combines two microbatches of four, with source-record weighting in Kev. The global trainable gradient norm is clipped to 1.0. Fused AdamW then updates only φ, and OneCycleLR advances the learning-rate schedule with a configured maximum of 2e−5. The new v4 stage starts from completed v3 weights with a fresh optimizer; recovery resumes v4’s own optimizer state. Shared weights and mixed requests mean no individual action probability is guaranteed to improve at every update.'}
  ];
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
  const highlight=(r,t,c=C.orange)=>`<rect x="${r.x-3}" y="${r.y-3}" width="${r.w+6}" height="${r.h+6}" rx="3" fill="none" stroke="${c}" stroke-width="${1.2+Math.sin(t*Math.PI)*1.4}"/>`;
  function overlay(stage,t){return [()=>highlight({x:33,y:110,w:653,h:57},t),()=>highlight(rects[1],t),()=>highlight(rects[0],t,C.teal),()=>highlight({x:113,y:246,w:218,h:30},t),()=>highlight({x:33,y:326,w:653,h:49},t)][stage]();}
  function distribution(left){const up=(1-left)*.4375,right=(1-left)*.5625;return {up,left,right,gradient:[up,left-1,right],loss:-Math.log(left)};}
  const signed=x=>(x<0?'−':'+')+Math.abs(x).toFixed(2);
  function rich(v){let p=0,s='';for(const [a,b] of v.subs||[]){s+=esc(v.text.slice(p,a))+`<tspan baseline-shift="sub" font-size="${v.size*.7}">${esc(v.text.slice(a,b))}</tspan>`;p=b;}return s+esc(v.text.slice(p));}
  function diagram(left=.2){const d=distribution(left);return rects.map(r=>`<rect x="${r.x}" y="${r.y}" width="${r.w}" height="${r.h}" fill="${r.fill}" stroke="${r.stroke}" stroke-width=".6"/>`).join('')+lines.map(([x1,y1,x2,y2,c,arrow])=>{const dist=Math.hypot(x2-x1,y2-y1),ux=(x2-x1)/dist,uy=(y2-y1)/dist,bx=x2-ux*4,by=y2-uy*4;return `<path d="M${x1} ${y1}L${x2} ${y2}" fill="none" stroke="${c}" stroke-width="1.3"/>${arrow?`<polygon points="${x2},${y2} ${bx-uy*2},${by+ux*2} ${bx+uy*2},${by-ux*2}" fill="${c}"/>`:''}`;}).join('')+texts.map(v=>{if(v.id==='gradient-values')v={...v,text:`Up ${signed(d.gradient[0])}   Left ${signed(d.gradient[1])}   Right ${signed(d.gradient[2])}`};return `<text xml:space="preserve" x="${v.center?v.x+v.w/2:v.x}" y="${v.y+v.size}" font-family="${v.formula?'Georgia':'Arial'}" font-size="${v.size}" font-weight="${v.bold?'700':'400'}" fill="${v.color||C.ink}"${v.center?' text-anchor="middle"':''}>${rich(v)}</text>`;}).join('');}
  const api={C,rects,lines,texts,stages,distribution,overlay,diagram};if(typeof module!=='undefined'&&module.exports)module.exports=api;global.KevGradients=api;if(typeof document==='undefined')return;const svg=document.querySelector('#gradients');if(!svg)return;
  let stage=0,progress=0,playing=false,last=null,left=.2;const buttons=[...document.querySelectorAll('[data-stage]')];
  function render(){const d=distribution(left);svg.innerHTML=diagram(left)+`<g aria-hidden="true">${overlay(stage,progress)}</g>`;document.querySelector('#stage-title').textContent=stages[stage].title;document.querySelector('#explanation').textContent=stages[stage].text;document.querySelector('#play').textContent=playing?'Pause':'Play';document.querySelector('#play').setAttribute('aria-pressed',String(playing));buttons.forEach((b,i)=>b.setAttribute('aria-current',stage===i?'step':'false'));document.querySelector('#previous').disabled=stage===0;document.querySelector('#next').disabled=stage===stages.length-1;document.querySelector('#probability-value').textContent=left.toFixed(2);document.querySelector('#readout').textContent=`Probabilities: Up ${d.up.toFixed(4)}, Left ${d.left.toFixed(4)}, Right ${d.right.toFixed(4)}. Score gradients: ${d.gradient.map(x=>(x>=0?'+':'')+x.toFixed(4)).join(', ')}. Gradient sum ${d.gradient.reduce((s,x)=>s+x,0).toFixed(4)}. CE ${d.loss.toFixed(4)}.`;document.querySelectorAll('[data-probability]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.probability)===left)));}
  function choose(i){stage=i;progress=0;playing=false;last=null;render();}buttons.forEach(b=>b.addEventListener('click',()=>choose(Number(b.dataset.stage))));document.querySelector('#previous').addEventListener('click',()=>choose(Math.max(0,stage-1)));document.querySelector('#next').addEventListener('click',()=>choose(Math.min(stages.length-1,stage+1)));document.querySelector('#reset').addEventListener('click',()=>{left=.2;document.querySelector('#probability').value='0.2';choose(0);});document.querySelector('#play').addEventListener('click',()=>{playing=!playing;last=null;render();});document.querySelector('#probability').addEventListener('input',e=>{left=Number(e.target.value);choose(0);});document.querySelectorAll('[data-probability]').forEach(b=>b.addEventListener('click',()=>{left=Number(b.dataset.probability);document.querySelector('#probability').value=String(left);choose(0);}));
  function tick(now){if(playing){if(last!==null)progress+=(now-last)/2500;if(progress>=1){progress=0;stage=(stage+1)%stages.length;}render();}last=now;requestAnimationFrame(tick);}document.addEventListener('visibilitychange',()=>{last=null;});render();requestAnimationFrame(tick);
})(globalThis);
