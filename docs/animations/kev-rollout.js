/* Recorded native-game evidence. The animation explains a protocol; it runs no model. */
(function(global){
  'use strict';
  const evidence = /*EVIDENCE*/ {"recorded_date":"2026-10-07","scope":"20 reference starts, already used for diagnosis; not a blind test. No v4 outcomes.","source_revision":"b5aa381","source_sha256":{"ghost-evasion-evidence.json":"10752800ee3a63e1eed28f37b9e21813d7d8df3f89f8e517962b2f366610393a","pellet-efficiency-evidence.json":"1eb80cda3e69ddfe4b119850b9992d6e1e838ca4fb4f03e31df0688e203c9bb5","teacher-reference.json":"c3c06bb5a5de1cca6559da62206a1314aefa17c544c97bd7171b22903014c888"},"controllers":{"native-v2":{"games":20,"clears":0,"avoidable":52,"pellets":2639,"frames":56471,"life_losses":60,"longest_dry":131,"checkpoint_sha256":"c100872bd94f9869807ed33bdb4617cc38867abf61c551a6ac507e8f38106d33"},"native-v3":{"games":20,"clears":0,"avoidable":22,"pellets":2647,"frames":76493,"life_losses":60,"longest_dry":273,"checkpoint_sha256":"c7c5c8326f74e86e161e05730db5004efdfd3b2351c6f2f9478581cfc13550b0"},"teacher":{"games":20,"clears":20,"avoidable":0,"pellets":4880,"frames":65865,"life_losses":1,"longest_dry":112}}};
  const C={ink:'#111827',muted:'#5e6e80',teal:'#167985',orange:'#b36b17',tealFill:'#eff8f8',orangeFill:'#fff4e4'};
  const rects=[
    {id:'observation',x:36,y:132,w:164,h:78,fill:C.tealFill,stroke:C.teal},
    {id:'policy',x:244,y:132,w:209,h:78,fill:C.orangeFill,stroke:C.orange},
    {id:'game',x:496,y:132,w:188,h:78,fill:C.tealFill,stroke:C.teal}
  ];
  const lines=[
    [200,171,244,171,C.teal,true],[453,171,496,171,C.teal,true],
    [590,210,590,234,C.teal,false],[590,234,118,234,C.teal,false],[118,234,118,210,C.teal,true]
  ];
  function math(parts){let text='',subs=[];for(const part of parts){if(typeof part==='string')text+=part;else{const start=text.length;text+=part.sub;subs.push([start,text.length]);}}return {text,subs};}
  const texts=[
    {id:'title',x:36,y:25,w:648,h:42,size:30,bold:true,text:'Does the learned policy survive?'},
    {id:'subtitle',x:36,y:76,w:648,h:24,size:16,text:'Run complete native games. Diagnose mistakes in saved traces.'},
    {id:'starts',x:36,y:108,w:648,h:18,size:12.5,color:C.muted,text:'Same 20 starts: 5 seeds × levels 1, 2, 3, 5'},
    {id:'observation-title',x:48,y:141,w:140,h:18,size:13.5,bold:true,color:C.teal,text:'Observe full game'},
    {id:'observation-formula',x:48,y:166,w:140,h:27,size:19,formula:true,...math(['s',{sub:'t'},' = O(g',{sub:'t'},')'])},
    {id:'observation-detail',x:48,y:194,w:140,h:13,size:10.5,color:C.muted,text:'Full maze + ghosts + timers'},
    {id:'policy-title',x:256,y:141,w:185,h:18,size:13.5,bold:true,color:C.orange,text:'Kev selects one legal move'},
    {id:'policy-formula',x:256,y:166,w:185,h:27,size:20,formula:true,...math(['a',{sub:'t'},' = π',{sub:'φ'},'(s',{sub:'t'},')'])},
    {id:'policy-detail',x:256,y:194,w:185,h:13,size:10.5,color:C.muted,text:'Largest probability; no teacher override'},
    {id:'game-title',x:508,y:141,w:164,h:18,size:13.5,bold:true,color:C.teal,text:'Native game advances'},
    {id:'game-formula',x:508,y:166,w:164,h:27,size:18,formula:true,...math(['g',{sub:'t+1'},' = F(g',{sub:'t'},', a',{sub:'t'},')'])},
    {id:'game-detail',x:508,y:194,w:164,h:13,size:10.5,color:C.muted,text:'4 ghosts · pellets · power timers'},
    {id:'repeat',x:224,y:216,w:356,h:14,size:11.5,color:C.muted,text:'Repeat until clear, game-over or watchdog'},
    {id:'clear-formula',x:36,y:259,w:302,h:29,size:20,formula:true,...math(['C = (1/N) ∑',{sub:'j'},' 1[clear',{sub:'j'},']'])},
    {id:'clear-key',x:36,y:290,w:302,h:18,size:12,text:'N = games; count genuine level clears only'},
    {id:'efficiency-formula',x:36,y:315,w:302,h:29,size:20,formula:true,...math(['E = 1000 P / n',{sub:'f'}])},
    {id:'efficiency-key',x:36,y:346,w:302,h:18,size:12,...math(['P: pellets; n',{sub:'f'},': active native frames'])},
    {id:'diagnostics',x:36,y:365,w:302,h:17,size:12,color:C.muted,text:'Inspect deaths, dry spells and loops.'},
    {id:'archive-caption',x:356,y:367,w:328,h:13,size:10.5,color:C.muted,text:'Recorded v3 / teacher. V4 must be measured.'},
    {id:'footer',x:36,y:383,w:632,h:14,size:9,color:C.muted,text:'Reference suite, not blind test. Open rollout controls.'},
    {id:'number',x:678,y:383,w:20,h:14,size:9,color:C.muted,text:'13'}
  ];
  const table={x:356,y:248,w:328,h:115.2,widths:[174,76,78],rowHeight:28.8};
  function tableRows(name='native-v3'){
    const s=evidence.controllers[name],t=evidence.controllers.teacher;
    return [['Metric',name==='native-v3'?'Native-v3':'Native-v2','Teacher'],
      ['Clears',`${s.clears}/${s.games}`,`${t.clears}/${t.games}`],
      ['Avoidable deaths',String(s.avoidable),String(t.avoidable)],
      ['Pellets / 1k frames',(1000*s.pellets/s.frames).toFixed(2),(1000*t.pellets/t.frames).toFixed(2)]];
  }
  const stages=[
    {title:'1. Pair the starting games',text:'Use the same 20 seed/level pairs for every controller. Keep the native engine, full-state observation format, adjacent-tile-entry actions and watchdog limits identical. Log the active checkpoint hash. These starts were already used to diagnose v2 and v3, so this is a reference suite, not a blind generalization test.'},
    {title:'2. Observe the whole current game',text:'g_t is the native engine snapshot at decision t. O is the observation function and s_t is what Kev actually receives: the full maze, pellets and power pellets, each ghost’s identity and movement, native clocks and recent movement history. Kev does not receive future teacher rollouts or hidden future randomness.'},
    {title:'3. Let the learned policy choose',text:'π_φ, pronounced “pi sub phi,” denotes Kev with the current LoRA and pointer-head weights. It selects the largest probability among the legal moves supplied for this observation. a_t is the selected direction. The instruction and legal option list are implicit in this compact notation. A diagnostic teacher or immediate-risk check never substitutes a safer action for Kev.'},
    {title:'4. Advance the actual native engine',text:'F is the native game transition: execute a_t with unchanged physics, ghost controllers and timers, then return g_(t+1). Decision t+1 can take several native frames; it is not one frame. Observe again after every move. Continue after an intermediate life loss while lives remain. Stop at the first level clear, native game-over or a watchdog exit. Caps and stalls are reported as incomplete failures, never wins.'},
    {title:'5. Measure safety and food progress together',text:'C is clear rate: sum the indicators of a genuine level clear over games j and divide by N. E is food efficiency: 1,000 times total pellets P divided by active native frames n_f. The recorded v3 and teacher results below come from the same 20 starts. They are not v4 results. Single-state agreement and CE diagnose imitation; full-game clears, life losses, pellet efficiency, dry intervals and power-expiry failures determine whether gameplay improves.'}
  ];
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
  function rich(v){let p=0,s='';for(const [a,b] of v.subs||[]){s+=esc(v.text.slice(p,a))+`<tspan baseline-shift="sub" font-size="${v.size*.7}">${esc(v.text.slice(a,b))}</tspan>`;p=b;}return s+esc(v.text.slice(p));}
  function highlight(r,t,c=C.orange){return `<rect x="${r.x-3}" y="${r.y-3}" width="${r.w+6}" height="${r.h+6}" rx="3" fill="none" stroke="${c}" stroke-width="${1.2+Math.sin(t*Math.PI)*1.4}"/>`;}
  function overlay(stage,t){return [()=>highlight({x:33,y:108,w:653,h:17},t),()=>highlight(rects[0],t,C.teal),()=>highlight(rects[1],t),()=>highlight(rects[2],t,C.teal),()=>highlight({x:33,y:245,w:653,h:112},t)][stage]();}
  function diagram(name='native-v3'){
    return rects.map(r=>`<rect x="${r.x}" y="${r.y}" width="${r.w}" height="${r.h}" fill="${r.fill}" stroke="${r.stroke}" stroke-width=".6"/>`).join('')+
      lines.map(([x1,y1,x2,y2,c,arrow])=>{const d=Math.hypot(x2-x1,y2-y1),ux=(x2-x1)/d,uy=(y2-y1)/d,bx=x2-ux*4,by=y2-uy*4;return `<path d="M${x1} ${y1}L${x2} ${y2}" fill="none" stroke="${c}" stroke-width="1.2"/>${arrow?`<polygon points="${x2},${y2} ${bx-uy*2},${by+ux*2} ${bx+uy*2},${by-ux*2}" fill="${c}"/>`:''}`;}).join('')+
      texts.map(v=>{if(v.id==='archive-caption'&&name==='native-v2')v={...v,text:'Recorded v2 / teacher. V4 must be measured.'};return `<text xml:space="preserve" x="${v.x}" y="${v.y+v.size}" font-family="${v.formula?'Georgia':'Arial'}" font-size="${v.size}" font-weight="${v.bold?'700':'400'}" fill="${v.color||C.ink}">${rich(v)}</text>`;}).join('')+
      tableRows(name).map((row,ri)=>{let x=table.x;return row.map((value,ci)=>{const w=table.widths[ci],y=table.y+ri*table.rowHeight;const cell=`<rect x="${x}" y="${y}" width="${w}" height="${table.rowHeight}" fill="${ri===0?C.tealFill:'#ffffff'}" stroke="#b5c4cc" stroke-width=".4"/><text x="${x+6}" y="${y+16}" font-family="Arial" font-size="12" font-weight="${ri===0?700:400}" fill="${C.ink}">${esc(value)}</text>`;x+=w;return cell;}).join('');}).join('');
  }
  const api={C,rects,lines,texts,table,tableRows,stages,evidence,overlay,diagram};if(typeof module!=='undefined'&&module.exports)module.exports=api;global.KevRollout=api;if(typeof document==='undefined')return;const svg=document.querySelector('#rollout');if(!svg)return;
  let stage=0,progress=0,playing=false,last=null,controller='native-v3';
  function render(){svg.innerHTML=diagram(controller)+`<g aria-hidden="true">${overlay(stage,progress)}</g>`;document.querySelector('#stage-title').textContent=stages[stage].title;document.querySelector('#explanation').textContent=stages[stage].text;document.querySelector('#play').textContent=playing?'Pause':'Play';document.querySelector('#play').setAttribute('aria-pressed',String(playing));document.querySelectorAll('[data-stage]').forEach((b,i)=>b.setAttribute('aria-current',stage===i?'step':'false'));document.querySelector('#previous').disabled=stage===0;document.querySelector('#next').disabled=stage===stages.length-1;document.querySelectorAll('[data-controller]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.controller===controller)));const s=evidence.controllers[controller];document.querySelector('#readout').textContent=`${controller}: C = ${s.clears}/${s.games} = ${(100*s.clears/s.games).toFixed(0)}%. E = 1,000 × ${s.pellets.toLocaleString('en-US')} / ${s.frames.toLocaleString('en-US')} = ${(1000*s.pellets/s.frames).toFixed(2)} pellets per 1,000 active native frames. ${s.life_losses} life losses; longest pellet-free interval ${s.longest_dry} decisions. Teacher: 20/20 clears, 1 life loss, 74.09 pellets per 1,000 frames; longest dry interval 112 decisions.`;}
  function choose(i){stage=i;progress=0;playing=false;last=null;render();}
  document.querySelectorAll('[data-stage]').forEach(b=>b.addEventListener('click',()=>choose(Number(b.dataset.stage))));document.querySelector('#previous').addEventListener('click',()=>choose(Math.max(0,stage-1)));document.querySelector('#next').addEventListener('click',()=>choose(Math.min(stages.length-1,stage+1)));document.querySelector('#reset').addEventListener('click',()=>{controller='native-v3';choose(0);});document.querySelector('#play').addEventListener('click',()=>{playing=!playing;last=null;render();});document.querySelectorAll('[data-controller]').forEach(b=>b.addEventListener('click',()=>{controller=b.dataset.controller;choose(4);}));
  function tick(now){if(playing){if(last!==null)progress+=(now-last)/2500;if(progress>=1){progress=0;stage=(stage+1)%stages.length;}render();}last=now;requestAnimationFrame(tick);}document.addEventListener('visibilitychange',()=>{last=null;});render();requestAnimationFrame(tick);
})(globalThis);
