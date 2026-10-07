/* Teaching schematic of the pinned Kev inference architecture, not a model run. */
(function (global) {
  'use strict';
  const C = {ink:'#111827', muted:'#5e6e80', teal:'#167985', orange:'#b36b17',
    tealFill:'#eff8f8', orangeFill:'#fff4e4', gray:'#dce4e8', light:'#f5f7f9'};
  const rects = [
    {id:'input',x:36,y:142,w:120,h:140,fill:C.light,stroke:C.gray},
    {id:'backbone',x:186,y:139,w:228,h:143,fill:'#fbfdfd',stroke:C.gray},
    {id:'lora',x:242,y:152,w:104,h:45,fill:C.orangeFill,stroke:C.orange},
    {id:'base',x:242,y:219,w:104,h:45,fill:C.tealFill,stroke:C.teal},
    {id:'head',x:436,y:157,w:119,h:116,fill:C.orangeFill,stroke:C.orange},
  ];
  const lines = [
    [156,215,222,215,C.teal,true], [222,215,222,174,C.orange,false],
    [222,174,242,174,C.orange,true], [222,215,222,241,C.teal,false],
    [222,241,242,241,C.teal,true], [346,174,389,174,C.orange,false],
    [389,174,389,205,C.orange,true], [346,241,389,241,C.teal,false],
    [389,241,389,225,C.teal,true], [399,215,436,215,C.teal,true],
    [555,215,579,215,C.orange,true],
  ];
  const texts = [
    {id:'title',x:36,y:25,w:648,h:50,size:30,bold:true,text:'Kev architecture'},
    {id:'subtitle',x:36,y:76,w:648,h:28,size:17,text:'One board, supplied moves, direct probabilities'},
    {id:'input-title',x:36,y:112,w:120,h:25,size:17,bold:true,text:'Input x'},
    {id:'model-title',x:186,y:112,w:228,h:25,size:17,bold:true,text:'Qwen3.5-4B-Base'},
    {id:'head-title',x:436,y:112,w:130,h:25,size:17,bold:true,text:'Pointer head'},
    {id:'output-title',x:584,y:112,w:108,h:25,size:17,bold:true,text:'Softmax p'},
    {id:'state',x:45,y:155,w:103,h:94,size:13,text:'Maze + ghosts\nTimers + history\nChoose a move\nLeft / Up / Right'},
    {id:'tokens',x:45,y:254,w:103,h:21,size:13,color:C.muted,text:'Input token IDs'},
    {id:'lora-label',x:246,y:160,w:96,h:34,size:13,bold:true,color:C.orange,center:true,text:'LoRA ΔW\ntrainable'},
    {id:'base-label',x:246,y:227,w:96,h:34,size:13,bold:true,color:C.teal,center:true,text:'Base W₀\nfixed'},
    {id:'hidden',x:414,y:190,w:20,h:24,size:16,bold:true,color:C.teal,text:'H'},
    {id:'head-input',x:446,y:175,w:99,h:48,size:13,center:true,text:'Decision +\noption vectors'},
    {id:'head-score',x:443,y:239,w:105,h:24,size:13,bold:true,color:C.orange,center:true,text:'Learned scores'},
    {id:'score',x:560,y:190,w:18,h:24,size:16,bold:true,color:C.orange,text:'z'},
    {id:'left',x:584,y:154,w:108,h:22,size:13,bold:true,color:C.teal,text:'Left  0.67'},
    {id:'up',x:584,y:198,w:108,h:22,size:13,text:'Up  0.24'},
    {id:'right',x:584,y:242,w:108,h:22,size:13,text:'Right  0.09'},
    {id:'execute',x:584,y:282,w:108,h:23,size:15,bold:true,color:C.teal,text:'Execute Left'},
    {id:'repeated',x:186,y:285,w:228,h:18,size:12,color:C.muted,text:'Adapted projections across layers'},
    {id:'encoder-formula',x:36,y:306,w:355,h:32,size:22,formula:true,text:'H = f(x; W₀ + ΔW)'},
    {id:'output-formula',x:436,y:306,w:248,h:32,size:22,formula:true,text:'p = softmax(z)'},
    {id:'symbols',x:36,y:343,w:648,h:36,size:13,color:C.muted,text:'x: input tokens   H: hidden vectors   W₀: fixed base weights\nΔW: LoRA updates   z: option scores   p: move probabilities'},
    {id:'footer',x:36,y:383,w:615,h:17,size:9,color:C.muted,text:'Illustrative probabilities. Open animation controls.'},
    {id:'number',x:665,y:383,w:19,h:17,size:9,color:C.muted,text:'1'},
  ];
  const stages = [
    {title:'1. Supply the decision input',text:'The game supplies the current maze, ghost state, timers, movement history and legal moves. These become input token IDs, x. Kev chooses only among the supplied options.'},
    {title:'2. Compute contextual hidden vectors',text:'Qwen processes x using its fixed base weights W₀ and learned LoRA updates ΔW. H collects the resulting hidden vectors. This branch drawing summarizes adaptations inside multiple layers, not one adapter added after the whole model.'},
    {title:'3. Score the supplied options',text:'The trainable pointer head reads the final decision marker and the closing marker of each option from H. Its two projections produce one score per supplied move. The vector z contains these scores.'},
    {title:'4. Convert scores into probabilities',text:'Softmax converts z into p, a probability distribution over the supplied moves. The example uses logits [2, 1, 0], giving approximately [0.67, 0.24, 0.09]. These are illustrative values, not measured model output or survival odds.'},
    {title:'5. Execute the selected label',text:'The application returns and executes Left, the option with the largest probability in this example. The next board becomes a new input. Kev does not generate the action label or response JSON token by token.'},
  ];
  function esc(s) { return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c])); }
  function point(path,t) {
    const lengths=path.slice(1).map((p,i)=>Math.hypot(p[0]-path[i][0],p[1]-path[i][1]));
    let d=Math.max(0,Math.min(1,t))*lengths.reduce((a,b)=>a+b,0);
    for(let i=0;i<lengths.length;i++) {
      if(d<=lengths[i]){const f=lengths[i]?d/lengths[i]:0;return [path[i][0]+f*(path[i+1][0]-path[i][0]),path[i][1]+f*(path[i+1][1]-path[i][1])];}
      d-=lengths[i];
    }
    return path.at(-1);
  }
  function dot(path,t,color) { const [x,y]=point(path,t); return `<circle cx="${x}" cy="${y}" r="4" fill="${color}"/>`; }
  function overlay(stage,t) {
    t=Math.max(0,Math.min(1,t));
    if(stage===0)return dot([[156,215],[222,215]],t,C.teal);
    if(stage===1)return dot([[222,215],[222,174],[242,174],[346,174],[389,174],[389,215],[436,215]],t,C.orange)+dot([[222,215],[222,241],[242,241],[346,241],[389,241],[389,215],[436,215]],t,C.teal);
    if(stage===2)return `<rect x="439" y="160" width="113" height="110" rx="5" fill="none" stroke="${C.orange}" stroke-width="${1.3+Math.sin(t*Math.PI)*1.5}"/>`;
    if(stage===3)return dot([[555,215],[577,215]],t,C.orange);
    return `<rect x="581" y="149" width="108" height="47" rx="3" fill="none" stroke="${C.teal}" stroke-width="${1.3+Math.sin(t*Math.PI)*1.5}"/>`;
  }
  function diagram() {
    const defs=`<defs><marker id="teal-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="${C.teal}"/></marker><marker id="orange-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="${C.orange}"/></marker></defs>`;
    let s=defs+rects.map(r=>`<rect id="${r.id}" x="${r.x}" y="${r.y}" width="${r.w}" height="${r.h}" rx="5" fill="${r.fill}" stroke="${r.stroke}" stroke-width="1"/>`).join('');
    s+=lines.map(([x1,y1,x2,y2,c,a])=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${c}" stroke-width="1.4"${a?` marker-end="url(#${c===C.teal?'teal':'orange'}-arrow)"`:''}/>`).join('');
    s+=`<circle cx="389" cy="215" r="10" fill="white" stroke="${C.muted}"/><text x="389" y="220" text-anchor="middle" font-family="Arial" font-size="17" fill="${C.ink}">+</text>`;
    for(const [y,w] of [[181,67],[225,24],[269,9]])s+=`<rect x="584" y="${y}" width="100" height="7" fill="${C.light}"/><rect x="584" y="${y}" width="${w}" height="7" fill="${C.teal}"/>`;
    s+=texts.map(v=>`<text id="text-${v.id}" x="${v.center?v.x+v.w/2:v.x}" y="${v.y+v.size}" font-family="${v.formula?'Georgia':'Arial'}" font-size="${v.size}" font-weight="${v.bold?'700':'400'}" fill="${v.color||C.ink}"${v.center?' text-anchor="middle"':''}>${v.text.split('\n').map((line,i)=>`<tspan x="${v.center?v.x+v.w/2:v.x}" dy="${i?v.size*1.3:0}">${esc(line)}</tspan>`).join('')}</text>`).join('');
    return s;
  }
  const api={C,rects,lines,texts,stages,overlay,diagram,point};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  global.KevArchitecture=api;
  if(typeof document==='undefined')return;
  const svg=document.querySelector('#architecture');
  if(!svg)return;
  svg.innerHTML=diagram()+'<g id="flow" aria-hidden="true"></g>';
  let stage=0,progress=0,playing=false,previous=null;
  const buttons=[...document.querySelectorAll('[data-stage]')];
  function render() {
    document.querySelector('#flow').innerHTML=overlay(stage,progress);
    document.querySelector('#stage-title').textContent=stages[stage].title;
    document.querySelector('#explanation').textContent=stages[stage].text;
    document.querySelector('#play').textContent=playing?'Pause':'Play';
    document.querySelector('#play').setAttribute('aria-pressed',String(playing));
    buttons.forEach((b,i)=>b.setAttribute('aria-current',i===stage?'step':'false'));
    document.querySelector('#previous').disabled=stage===0;
    document.querySelector('#next').disabled=stage===stages.length-1;
  }
  function choose(i){stage=i;progress=0;playing=false;previous=null;render();}
  buttons.forEach(b=>b.addEventListener('click',()=>choose(Number(b.dataset.stage))));
  document.querySelector('#play').addEventListener('click',()=>{playing=!playing;previous=null;render();});
  document.querySelector('#reset').addEventListener('click',()=>choose(0));
  document.querySelector('#previous').addEventListener('click',()=>choose(Math.max(0,stage-1)));
  document.querySelector('#next').addEventListener('click',()=>choose(Math.min(stages.length-1,stage+1)));
  function tick(now){
    if(playing){
      if(previous!==null)progress+=(now-previous)/3000;
      if(progress>=1){progress=0;stage++;if(stage===stages.length){stage=stages.length-1;playing=false;progress=1;}}
      render();
    }
    previous=now;requestAnimationFrame(tick);
  }
  document.addEventListener('visibilitychange',()=>{previous=null;});
  render();requestAnimationFrame(tick);
})(typeof globalThis==='undefined'?this:globalThis);
