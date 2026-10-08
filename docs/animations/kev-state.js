/* Exact v2 observation fields; a fresh native maze excerpt, not a learner trace. */
(function(global){
  'use strict';
  const C={ink:'#111827',muted:'#5e6e80',teal:'#167985',orange:'#b36b17',tealFill:'#eff8f8',orangeFill:'#fff4e4',maze:'#0b1020',wall:'#334be8',pellet:'#f4c4ad'};
  const maze=['############','#...........','#.####.#####','#o####.#####','#.####.#####','#...........','#.####.##.##','#.####.##.##','#......##...','######.#####','######.#####','######.##   '];
  const rects=[{id:'maze',x:36,y:124,w:192,h:192,fill:C.maze,stroke:C.maze},{id:'actors',x:268,y:116,w:416,h:93,fill:C.tealFill,stroke:C.teal},{id:'timers',x:268,y:217,w:416,h:62,fill:C.orangeFill,stroke:C.orange},{id:'history',x:268,y:287,w:416,h:61,fill:C.tealFill,stroke:C.teal}];
  maze.forEach((row,r)=>[...row].forEach((tile,c)=>{if(tile==='#')rects.push({id:`wall-${r}-${c}`,x:36+c*16+1,y:124+r*16+1,w:14,h:14,fill:C.wall,stroke:C.wall});else if(tile==='.'||tile==='o'){const n=tile==='o'?9:3;rects.push({id:`food-${r}-${c}`,x:36+c*16+(16-n)/2,y:124+r*16+(16-n)/2,w:n,h:n,fill:C.pellet,stroke:C.pellet,type:'ELLIPSE'});}}));
  const lines=[];
  const texts=[
    {id:'title',x:36,y:25,w:648,h:42,size:30,bold:true,text:'The game state Kev actually receives'},
    {id:'subtitle',x:36,y:76,w:648,h:25,size:17,text:'A structured board, moving actors, native timers and memory.'},
    {id:'map-label',x:36,y:99,w:232,h:24,size:10,color:C.muted,text:'Display crop only\nKev receives the full maze'},
    {id:'legend',x:36,y:322,w:212,h:24,size:12,color:C.muted,text:'. pellet · o power pellet'},
    {id:'actors-title',x:280,y:123,w:392,h:22,size:16,bold:true,color:C.teal,text:'Player + four distinct ghosts'},
    {id:'blinky',x:280,y:145,w:82,h:18,size:12,bold:true,color:'#c02929',text:'Blinky'},
    {id:'pinky',x:378,y:145,w:82,h:18,size:12,bold:true,color:'#a63568',text:'Pinky'},
    {id:'inky',x:476,y:145,w:82,h:18,size:12,bold:true,color:'#087d92',text:'Inky'},
    {id:'clyde',x:574,y:145,w:82,h:18,size:12,bold:true,color:'#aa6313',text:'Clyde'},
    {id:'actor-fields1',x:280,y:163,w:392,h:18,size:12,text:'row/column · heading · pixel_offset · speed_phase'},
    {id:'actor-fields2',x:280,y:181,w:392,h:18,size:12,text:'Ghosts: mode · frightened · next/pending turns'},
    {id:'timers-title',x:280,y:224,w:392,h:22,size:16,bold:true,color:C.orange,text:'Native clocks'},
    {id:'timers1',x:280,y:246,w:392,h:18,size:12,text:'timing.power.remaining_frames'},
    {id:'timers2',x:280,y:262,w:392,h:18,size:11,color:C.muted,text:'chase/scatter · release · Elroy · fruit · eating pauses'},
    {id:'history-title',x:280,y:293,w:392,h:22,size:16,bold:true,color:C.teal,text:'Movement and pellet history'},
    {id:'history1',x:280,y:315,w:392,h:18,size:12,text:'recent_positions (up to 64) · destination_visits'},
    {id:'history2',x:280,y:331,w:392,h:18,size:11,color:C.muted,text:'decisions_since_last_pellet · visits_to_current_tile'},
    {id:'observation',x:36,y:353,w:240,h:28,size:22,formula:true,text:'st = O(gt)',subs:[[1,2],[8,9]]},
    {id:'actions',x:340,y:353,w:230,h:28,size:22,formula:true,text:'at ∈ At',subs:[[1,2],[6,7]]},
    {id:'actions-caption',x:500,y:357,w:184,h:22,size:12,color:C.muted,text:'Wall-legal directions'},
    {id:'footer',x:36,y:383,w:615,h:17,size:9,color:C.muted,text:'Teacher forecasts and labels stay outside this input. Open animation controls.'},
    {id:'number',x:665,y:383,w:19,h:17,size:9,color:C.muted,text:'8'}
  ];
  const ghosts=[
    {name:'blinky',row:14,column:13,heading:'left',mode:'outside',frightened:false,pixel_offset:{x:4,y:0},speed_phase:0,next_heading:'left',pending_reverse:false,pending_release:false},
    {name:'pinky',row:17,column:13,heading:'down',mode:'pacing_home',frightened:false,pixel_offset:{x:4,y:0},speed_phase:0,next_heading:'down',pending_reverse:false,pending_release:false},
    {name:'inky',row:17,column:11,heading:'up',mode:'pacing_home',frightened:false,pixel_offset:{x:4,y:0},speed_phase:0,next_heading:'up',pending_reverse:false,pending_release:false},
    {name:'clyde',row:17,column:15,heading:'up',mode:'pacing_home',frightened:false,pixel_offset:{x:4,y:0},speed_phase:0,next_heading:'up',pending_reverse:false,pending_release:false}
  ];
  const stages=[
    {title:'1. Read the maze and food',text:'The maze is an array of native tile strings. A dot is a 10-point pellet; o is a 50-point power pellet; a space is empty floor. Consuming food changes that tile to a space. This is the upper-left native map excerpt, rows 3–14 and columns 0–11. Only this display is cropped. Kev receives every row and column of the full maze, together with the current actors, timers and history.'},
    {title:'2. Read the player’s exact motion',text:'Player position includes row and column, heading and pixel_offset relative to the tile center. The input also includes speed_phase, eat_pause_frames, stopped, next_heading and queued_heading. Two actors can occupy the same tile while being at different pixels or movement phases.'},
    {title:'3. Keep the ghosts distinct',text:'Every ghost has a name, tile position, heading, mode and frightened flag. Its pixel_offset, speed_phase, next_heading, pending_reverse and pending_release are also visible. The four names identify different native targeting rules. At this fresh starting state, Blinky is outside while the others are pacing in the house; native release rules bring them out.'},
    {title:'4. Read the remaining power time',text:'The input supplies timing.power.remaining_frames as well as global frightened and each ghost’s frightened flag. An uneaten o shows where a power pellet remains; a consumed power pellet disappears from the maze. The timer matters because power can expire during a multi-frame move. Ghost-eating pauses can pause that clock, and some ghost flags can remain true briefly when the clock reaches zero. Read timer, per-ghost mode and pixel phase together. Chase/scatter, release, Elroy, fruit and eating-pause state are visible too.'},
    {title:'5. Recognize loops and dry decisions',text:'The native-v2 observation keeps up to 64 recent positions, including headings, plus visits_to_current_tile and a destination_visits count for every legal direction. decisions_since_last_pellet counts decisions without a new pellet. These describe what has already happened; they do not supply a teacher-recommended route. A new life resets the recent-position/dry history.'},
    {title:'6. Choose a direction, then simulate frames',text:'s_t is the observation at decision t, O is the observation function, and g_t is the complete native game state. Some internals, including the private random generator, stay outside s_t. a_t is the chosen direction and A_t is the supplied set of wall-legal directions. Legal does not mean safe from ghosts. Native-v2 simulates complete frames until entry to the adjacent target tile or a terminal event; the model does not receive the teacher’s future rollout or label.'}
  ];
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
  const hl=(r,t,c=C.teal)=>`<rect x="${r.x-3}" y="${r.y-3}" width="${r.w+6}" height="${r.h+6}" rx="3" fill="none" stroke="${c}" stroke-width="${1.2+Math.sin(t*Math.PI)*1.4}"/>`;
  function overlay(stage,t){if(stage===0)return hl(rects[0],t);if(stage===1||stage===2)return hl(rects[1],t);if(stage===3)return hl(rects[2],t,C.orange)+hl({x:55.5,y:175.5,w:9,h:9},t,C.orange);if(stage===4)return hl(rects[3],t);return hl({x:33,y:349,w:654,h:27},t);}
  function rich(v){let p=0,s='';for(const [a,b] of v.subs||[]){s+=esc(v.text.slice(p,a))+`<tspan baseline-shift="sub" font-size="${v.size*.7}">${esc(v.text.slice(a,b))}</tspan>`;p=b;}return s+esc(v.text.slice(p));}
  function diagram(){return rects.map(r=>r.type==='ELLIPSE'?`<ellipse cx="${r.x+r.w/2}" cy="${r.y+r.h/2}" rx="${r.w/2}" ry="${r.h/2}" fill="${r.fill}"/>`:`<rect x="${r.x}" y="${r.y}" width="${r.w}" height="${r.h}" fill="${r.fill}" stroke="${r.stroke}"/>`).join('')+texts.map(v=>v.text.split('\n').map((line,i)=>`<text x="${v.x}" y="${v.y+v.size+i*v.size*1.1}" font-family="${v.formula?'Georgia':'Arial'}" font-size="${v.size}" font-weight="${v.bold?'700':'400'}" fill="${v.color||C.ink}">${rich({...v,text:line,subs:i?[]:v.subs})}</text>`).join('')).join('');}
  const api={C,rects,lines,texts,stages,ghosts,overlay,diagram};if(typeof module!=='undefined'&&module.exports)module.exports=api;global.KevState=api;if(typeof document==='undefined')return;const svg=document.querySelector('#state');if(!svg)return;
  let stage=0,progress=0,playing=false,last=null;const buttons=[...document.querySelectorAll('[data-stage]')];
  function render(){svg.innerHTML=diagram()+`<g aria-hidden="true">${overlay(stage,progress)}</g>`;document.querySelector('#stage-title').textContent=stages[stage].title;document.querySelector('#explanation').textContent=stages[stage].text;document.querySelector('#play').textContent=playing?'Pause':'Play';document.querySelector('#play').setAttribute('aria-pressed',String(playing));buttons.forEach((b,i)=>b.setAttribute('aria-current',stage===i?'step':'false'));document.querySelector('#previous').disabled=stage===0;document.querySelector('#next').disabled=stage===stages.length-1;}
  function choose(i){stage=i;progress=0;playing=false;last=null;render();}buttons.forEach(b=>b.addEventListener('click',()=>choose(Number(b.dataset.stage))));document.querySelector('#previous').addEventListener('click',()=>choose(Math.max(0,stage-1)));document.querySelector('#next').addEventListener('click',()=>choose(Math.min(stages.length-1,stage+1)));document.querySelector('#reset').addEventListener('click',()=>choose(0));document.querySelector('#play').addEventListener('click',()=>{playing=!playing;last=null;render();});document.querySelectorAll('[data-ghost]').forEach(b=>b.addEventListener('click',()=>{choose(2);document.querySelector('#ghost-fields').textContent=JSON.stringify(ghosts[Number(b.dataset.ghost)],null,2);}));document.querySelector('#ghost-fields').textContent=JSON.stringify(ghosts[0],null,2);function tick(now){if(playing){if(last!==null)progress+=(now-last)/2500;if(progress>=1){progress=0;stage=(stage+1)%stages.length;}render();}last=now;requestAnimationFrame(tick);}document.addEventListener('visibilitychange',()=>{last=null;});render();requestAnimationFrame(tick);
})(globalThis);
