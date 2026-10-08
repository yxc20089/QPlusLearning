/* Fixed native teacher qualification evidence. Controls illustrate frozen gates. */
(function(global){
  'use strict';
  const C={ink:'#111827',muted:'#5e6e80',teal:'#167985',orange:'#b36b17',tealFill:'#eff8f8',orangeFill:'#fff4e4'};
  const levels=[1,2,3,5], seeds=[91009,92021,93031,94033,95047];
  const evidence={version:'native-teacher-qualification-v3',games:20,clears:20,life_losses:0,avoidable:0,loop_streak:0,dry:67,invalid:0,verified_replays:20,replay_sha256:'bb21bcbf33ab5332c35a4f6022f8e4a1a4f70ef1546c0e5c3218142fc8d520c2'};
  const rects=[
    ...levels.flatMap((level,r)=>seeds.map((seed,c)=>({id:`game_${r}_${c}`,x:95+c*34,y:141+r*34,w:22,h:22,type:'ELLIPSE',fill:C.tealFill,stroke:C.teal}))),
    {id:'complete',x:36,y:330,w:168,h:35,fill:'#ffffff',stroke:C.muted},
    {id:'replay',x:276,y:330,w:168,h:35,fill:C.orangeFill,stroke:C.orange},
    {id:'admit',x:516,y:330,w:168,h:35,fill:C.tealFill,stroke:C.teal}
  ];
  const lines=[[204,347.5,274,347.5,C.muted,true],[444,347.5,514,347.5,C.teal,true]];
  const table={x:308,y:121,w:376,h:169,widths:[218,61,97],rows:[
    ['Frozen criterion','Observed','Limit'],
    ['Total life losses','0','≤ 4'],
    ['Avoidable immediate deaths','0','0'],
    ['Longest dry-cycle streak','0','≤ 7 / game'],
    ['Longest spell without a pellet','67','≤ 128 / game'],
    ['Invalid tile transitions','0','0']
  ]};
  const texts=[
    {id:'title',x:36,y:25,w:648,h:42,size:30,bold:true,text:'Teacher qualification'},
    {id:'subtitle',x:36,y:76,w:648,h:25,size:17,text:'Complete native games qualify the teacher before we use its labels.'},
    {id:'suite-label',x:36,y:111,w:258,h:21,size:15,bold:true,color:C.teal,text:'5 held-out seeds × 4 levels'},
    ...levels.map((l,r)=>({id:`level_${l}`,x:36,y:143+r*34,w:55,h:21,size:12,text:`Level ${l}`})),
    ...levels.flatMap((l,r)=>seeds.map((s,c)=>({id:`clear_${r}_${c}`,x:100+c*34,y:144+r*34,w:15,h:18,size:12,bold:true,color:C.teal,text:'✓'}))),
    {id:'wins',x:36,y:276,w:242,h:24,size:19,bold:true,color:C.teal,text:'20 / 20 levels cleared'},
    {id:'formula',x:36,y:303,w:265,h:24,size:19,formula:true,text:'Cℓ = wℓ / nℓ = 1',subs:[[1,2],[6,7],[11,12]]},
    {id:'replay-count',x:308,y:301,w:376,h:22,size:14,bold:true,color:C.orange,text:'20 exact native replays verified'},
    {id:'complete-label',x:48,y:338,w:144,h:20,size:12,text:'Run to clear or game over'},
    {id:'replay-label',x:292,y:338,w:139,h:20,size:12,bold:true,color:C.orange,text:'Replay stored decisions'},
    {id:'admit-label',x:528,y:338,w:144,h:20,size:12,bold:true,color:C.teal,text:'Teacher labels allowed'},
    {id:'footer',x:36,y:383,w:615,h:17,size:9,color:C.muted,text:'Fixed finite suite. Empirical evidence, no universal guarantee. Open animation controls.'},
    {id:'number',x:665,y:383,w:19,h:17,size:9,color:C.muted,text:'10'}
  ];
  const stages=[
    {title:'1. Fix the suite and gates first',text:'The qualification suite contains every pair of five held-out seeds and levels 1, 2, 3 and 5, exactly once. That makes 20 native games. Development seeds, qualification seeds and the separate student benchmark seeds stay out of training. We freeze the teacher and its gates before seeing these results.'},
    {title:'2. Finish each native game',text:'The same arcade engine starts from each declared board and advances all four ghosts, native clocks and pixel collisions. A run continues until the first level clears or the game ends after its lives. Intermediate deaths continue the game. Watchdog exits count as incomplete, never as wins. Every green circle here represents an observed complete level clear.'},
    {title:'3. Check survival and progress',text:'Clearing is required at every level. The frozen suite allows at most four total life losses and zero avoidable immediate deaths. Every game must also stay within seven consecutive dry-cycle decisions and 128 consecutive decisions without a pellet. Nonterminal moves must enter exactly one adjacent tile. The observed suite had zero life losses, zero loop streak and a longest dry spell of 67 decisions.'},
    {title:'4. Read the clear-rate equation',text:'C subscript ℓ is the clear rate at level ℓ. w subscript ℓ is the number of wins, meaning complete level clears. n subscript ℓ counts the games at that level. Here n is five for each of levels 1, 2, 3 and 5. Each level cleared five out of five games, so every C equals one. A large average score cannot compensate for a failed level or a failed safety gate.'},
    {title:'5. Replay the evidence',text:'Qualification checks the exact declared suite, frozen options and current source hashes. Then it cold-replays all stored teacher choices through the native engine and checks every state hash and the recomputed metrics. These 20 replays passed. Replaying stored decisions verifies the recorded evidence. It does not run the teacher search again, and a manually edited approved flag cannot authorize training.'},
    {title:'6. Admit a qualified teacher',text:'The frozen teacher passes this finite suite. Its labels may now support dataset generation. A positive training continuation must itself finish the level with zero life loss and pass the loop and stall gates under verified native replay. That is stricter than the suite allowance of four total losses. Passing 20 games gives empirical evidence, not a proof of universal survival or global optimality. The student must later pass its own separate complete-game benchmark. No v4 training result appears here.'}
  ];
  const cases=[
    {id:'observed',label:'Observed suite',kind:'Recorded qualification result',values:{...evidence},message:'PASS: all 20 declared games cleared, and every fixed gate passed.'},
    {id:'brief',label:'Brief repeat',kind:'Hypothetical teaching example',values:{...evidence,loop_streak:7},message:'PASS at the loop boundary: seven dry-cycle decisions are allowed. Other gates remain required.'},
    {id:'loop',label:'Sustained loop',kind:'Hypothetical teaching example',values:{...evidence,loop_streak:8},message:'REJECT: a dry-cycle streak of eight exceeds the fixed limit of seven.'},
    {id:'stall',label:'Pellet stall',kind:'Hypothetical teaching example',values:{...evidence,dry:129},message:'REJECT: 129 consecutive decisions without a pellet exceed the fixed limit of 128.'},
    {id:'death',label:'Avoidable death',kind:'Hypothetical teaching example',values:{...evidence,avoidable:1},message:'REJECT: one avoidable immediate death violates the zero-tolerance gate.'},
    {id:'incomplete',label:'Incomplete game',kind:'Hypothetical teaching example',values:{...evidence,clears:19},message:'REJECT: one unfinished or uncleared level violates the required clear rate of one at every level.'}
  ];
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
  const hl=(r,t,c=C.teal)=>`<rect x="${r.x-3}" y="${r.y-3}" width="${r.w+6}" height="${r.h+6}" rx="3" fill="none" stroke="${c}" stroke-width="${1.2+Math.sin(t*Math.PI)*1.4}"/>`;
  function overlay(stage,t){if(stage===0)return hl({x:36,y:109,w:258,h:158},t);if(stage===1)return rects.slice(0,20).map((r,i)=>i<=Math.floor(t*19)?hl(r,t):'').join('');if(stage===2)return hl(table,t,C.orange);if(stage===3)return hl({x:34,y:279,w:261,h:48},t);if(stage===4)return hl(rects[21],t,C.orange);return hl(rects[22],t);}
  function rich(v){let p=0,s='';for(const [a,b] of v.subs||[]){s+=esc(v.text.slice(p,a))+`<tspan baseline-shift="sub" font-size="${v.size*.7}">${esc(v.text.slice(a,b))}</tspan>`;p=b;}return s+esc(v.text.slice(p));}
  function diagram(){return rects.map(r=>r.type==='ELLIPSE'?`<ellipse cx="${r.x+r.w/2}" cy="${r.y+r.h/2}" rx="${r.w/2}" ry="${r.h/2}" fill="${r.fill}" stroke="${r.stroke}"/>`:`<rect x="${r.x}" y="${r.y}" width="${r.w}" height="${r.h}" fill="${r.fill}" stroke="${r.stroke}"/>`).join('')+lines.map(([x1,y1,x2,y2,c,arrow])=>`<path d="M${x1} ${y1}L${x2} ${y2}" fill="none" stroke="${c}" stroke-width="1.2"/>${arrow?`<polygon points="${x2},${y2} ${x2-5},${y2-3} ${x2-5},${y2+3}" fill="${c}"/>`:''}`).join('')+table.rows.map((row,i)=>{let x=table.x;return row.map((cell,j)=>{const out=`<rect x="${x}" y="${table.y+i*table.h/6}" width="${table.widths[j]}" height="${table.h/6}" fill="${i?'white':C.tealFill}" stroke="#d7e2e5"/><text x="${x+6}" y="${table.y+i*table.h/6+17}" font-family="Arial" font-size="${i?11:10}" font-weight="${i?'400':'700'}" fill="${C.ink}">${esc(cell)}</text>`;x+=table.widths[j];return out;}).join('');}).join('')+texts.map(v=>`<text x="${v.x}" y="${v.y+v.size}" font-family="${v.formula?'Georgia':'Arial'}" font-size="${v.size}" font-weight="${v.bold?'700':'400'}" fill="${v.color||C.ink}">${rich(v)}</text>`).join('');}
  const api={C,levels,seeds,evidence,rects,lines,table,texts,stages,cases,overlay,diagram};if(typeof module!=='undefined'&&module.exports)module.exports=api;global.KevQualification=api;if(typeof document==='undefined')return;const svg=document.querySelector('#qualification');if(!svg)return;
  let stage=0,progress=0,playing=false,last=null,caseId='observed';const buttons=[...document.querySelectorAll('[data-stage]')];
  function inspect(){const c=cases.find(c=>c.id===caseId),v=c.values;document.querySelector('#case-kind').textContent=c.kind;document.querySelector('#verdict').textContent=c.message;document.querySelector('#case-metrics').textContent=JSON.stringify({cleared_games:v.clears,total_games:v.games,total_life_losses:v.life_losses,avoidable_immediate_deaths:v.avoidable,longest_dry_cycle_streak:v.loop_streak,longest_no_pellet_spell:v.dry,invalid_tile_transitions:v.invalid},null,2);document.querySelectorAll('[data-case]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.case===caseId)));}
  function render(){svg.innerHTML=diagram()+`<g aria-hidden="true">${overlay(stage,progress)}</g>`;document.querySelector('#stage-title').textContent=stages[stage].title;document.querySelector('#explanation').textContent=stages[stage].text;document.querySelector('#play').textContent=playing?'Pause':'Play';document.querySelector('#play').setAttribute('aria-pressed',String(playing));buttons.forEach((b,i)=>b.setAttribute('aria-current',stage===i?'step':'false'));document.querySelector('#previous').disabled=stage===0;document.querySelector('#next').disabled=stage===stages.length-1;}
  function choose(i){stage=i;progress=0;playing=false;last=null;render();}buttons.forEach(b=>b.addEventListener('click',()=>choose(Number(b.dataset.stage))));document.querySelector('#previous').addEventListener('click',()=>choose(Math.max(0,stage-1)));document.querySelector('#next').addEventListener('click',()=>choose(Math.min(stages.length-1,stage+1)));document.querySelector('#reset').addEventListener('click',()=>{caseId='observed';inspect();choose(0);});document.querySelector('#play').addEventListener('click',()=>{playing=!playing;last=null;render();});document.querySelectorAll('[data-case]').forEach(b=>b.addEventListener('click',()=>{caseId=b.dataset.case;inspect();choose(2);}));function tick(now){if(playing){if(last!==null)progress+=(now-last)/2500;if(progress>=1){progress=0;stage=(stage+1)%stages.length;}render();}last=now;requestAnimationFrame(tick);}document.addEventListener('visibilitychange',()=>{last=null;});inspect();render();requestAnimationFrame(tick);
})(globalThis);
