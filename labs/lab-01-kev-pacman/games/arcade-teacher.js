// Native rollout MPC + CS188-style shortest-food routes. GPL-3.0.
// An independently implemented heuristic search, not a port of a competition
// agent, MCTS/UCT, a globally optimal policy, or a proof over every RNG outcome.
function createArcadeTeacher(api) {
    const names = ['up', 'left', 'down', 'right'], dy = [-1,0,1,0], dx = [0,-1,0,1];
    let edges, routes;
    function at(f) { return f.row*28+f.column; }
    function neighbor(tile, direction) { return (Math.floor(tile/28)+dy[direction])*28+(tile%28+dx[direction]+28)%28; }
    function graph(tiles) {
        edges = {};
        for (let i=0;i<tiles.length;i++) if (' .o'.includes(tiles[i]))
            edges[i] = names.map((_,d)=>neighbor(i,d)).filter(j=>j>=0 && j<tiles.length && ' .o'.includes(tiles[j]));
        routes = {};
        for (const key of Object.keys(edges)) {
            const origin=Number(key), queue=[origin], dist={[origin]:0};
            for (let head=0;head<queue.length;head++) for (const next of edges[queue[head]])
                if (dist[next]===undefined) { dist[next]=dist[queue[head]]+1;queue.push(next); }
            routes[origin]=dist;
        }
    }
    function moveRank(f, move, history, buffer) {
        const tile=at(f), danger=f.ghosts.filter(g=>g.danger).map(g=>g.row*28+g.column);
        const recent={}; history.forEach(t=>recent[t]=(recent[t]||0)+1);
            const next=neighbor(tile,names.indexOf(move)), distance=routes[next] || {};
            const separation=Math.min(99,...danger.map(g=>distance[g]===undefined?99:distance[g]));
            let food=999;
            for (let dot=0;dot<f.tiles.length;dot++) if (f.tiles[dot]==='.' || f.tiles[dot]==='o') {
                let cost=distance[dot]===undefined?999:distance[dot];
                // Penalize food targets behind dangerous ghosts, not all empty
                // corridor revisits. Frightened ghosts do not form this barrier.
                for (const g of danger) if ((routes[g][dot] || 0)<buffer)
                    cost += (buffer-(routes[g][dot] || 0))*2;
                food=Math.min(food,cost);
            }
            const hasFood=f.tiles[next]==='.' || f.tiles[next]==='o';
            return [separation<=1, separation<buffer,
                hasFood ? 0 : completesCycle(history,next),
                food+2*(recent[next]||0), -Math.min(separation,buffer+2), move!==f.heading];
    }
    function heuristic(f, legal, history, buffer) {
        return legal.map(move=>({move,rank:moveRank(f,move,history,buffer)})).sort((a,b)=>compareRank(a.rank,b.rank))[0].move;
    }
    function completesCycle(history, next) {
        const positions=history.concat(next);
        for(let period=2;period<=Math.min(64,Math.floor(positions.length/2));period++) {
            const tail=positions.slice(-period), previous=positions.slice(-2*period,-period);
            if(new Set(tail).size>=2 && tail.every((p,i)=>p===previous[i])) return true;
        }
        return false;
    }
    function compareRank(a,b) { for(let i=0;i<a.length;i++) if(a[i]!==b[i]) return a[i]-b[i]; return 0; }
    return function plan(options={}) {
        const normalHorizon=options.horizon_frames ?? 480, seeds=options.scenario_seeds || [11117,77717];
        const endgameHorizon=options.endgame_horizon_frames ?? normalHorizon;
        const dangerHorizon=options.danger_horizon_frames ?? normalHorizon;
        const buffers=options.buffers || [2,4,6];
        if(![normalHorizon,endgameHorizon,dangerHorizon].every(h=>Number.isInteger(h) && h>=16 && h<=3600) || !seeds.length || seeds.length>8 ||
            !seeds.every(s=>Number.isInteger(s) && s>0) || !buffers.length || buffers.length>8 ||
            !buffers.every(b=>Number.isInteger(b) && b>=1 && b<=16)) throw new Error('Invalid teacher search budget');
        const before=api.observe(), root=api.searchSave(-910000), base=api.searchFeatures();
        if (!edges) graph(base.tiles);
        const dangerDistance=Math.min(999,...base.ghosts.filter(g=>g.danger)
            .map(g=>routes[at(base)][g.row*28+g.column]??999));
        const horizon=Math.max(normalHorizon,base.pellets<=30?endgameHorizon:normalHorizon,
            dangerDistance<=6?dangerHorizon:normalHorizon);
        // Carry the actual dry-spell history into each rollout. Replanning from
        // an empty history can repeatedly promise the same future food while
        // reversing in the live game. The root also rewards the first route
        // step towards food instead of only counting food at a sliding horizon.
        const rootHistory=before.recent_positions.slice(-Math.min(64,(before.decisions_since_last_pellet||0)+1))
            .map(p=>p.row*28+p.column);
        let transitions=0;
        const candidates=[];
        try {
            for (const first of api.legalMoves()) {
                const policies=[];
                for (const buffer of buffers) {
                    const scenarios=[];
                    for (const seed of seeds) {
                        api.searchRestore(root); api.searchRandom(seed);
                        let status=api.step(first,true); transitions++;
                        let f=api.searchFeatures(), last=f.pellets, stale=0, maxStale=0;
                        let nextPelletFrames=f.pellets<base.pellets ? f.frames-base.frames : null;
                        let history=f.pellets<base.pellets ? [] : rootHistory.slice(), decisions=1;
                        const rootCycle=f.pellets===base.pellets && completesCycle(rootHistory,at(f));
                        history.push(at(f));
                        let cycleCount=rootCycle?1:0;
                        let firstCycleFrames=rootCycle ? f.frames-base.frames : null;
                        const immediateDeath=status==='life_lost';
                        while(!status && f.frames-base.frames<horizon && decisions<192) {
                            const snapshot=api.searchSave(-910001), legal=api.legalMoves();
                            // Exact one-action survival filter in rollout policy;
                            // still the teacher's simulation, never a Kev filter.
                            const safe=[];
                            for(const move of legal) {
                                api.searchRestore(snapshot);
                                const outcome=api.step(move,true);transitions++;
                                if(outcome!=='life_lost') safe.push(move);
                            }
                            api.searchRestore(snapshot);
                            const move=heuristic(f,safe.length?safe:legal,history,buffer);
                            status=api.step(move,true);transitions++;
                            f=api.searchFeatures(); decisions++;
                            if(nextPelletFrames===null && f.pellets<base.pellets) nextPelletFrames=f.frames-base.frames;
                            if(f.pellets<last) {history=[];stale=0;} else {
                                stale++;
                                if(completesCycle(history,at(f))) {
                                    cycleCount++;
                                    if(firstCycleFrames===null) firstCycleFrames=f.frames-base.frames;
                                }
                            }
                            maxStale=Math.max(maxStale,stale);
                            history.push(at(f));if(history.length>64)history.shift();last=f.pellets;
                        }
                        scenarios.push({immediate_death:immediateDeath, root_cycle:rootCycle,
                            no_pellet_cycles:cycleCount, survived:status!=='life_lost',
                            cleared:status==='level_cleared', pellets:base.pellets-f.pellets,
                            score:f.score-base.score, frames:f.frames-base.frames,
                            next_pellet_frames:nextPelletFrames,
                            first_cycle_frames:firstCycleFrames,
                            max_no_pellet_decisions:maxStale, outcome:status});
                    }
                    const deaths=scenarios.filter(s=>!s.survived).length;
                    // Exact immediate safety has priority. Among currently
                    // nonfatal moves, do not close an actual dry cycle merely
                    // because a limited rollout policy predicts a later death.
                    // The prediction is approximate and will be replanned.
                    const rank=[Math.max(...scenarios.map(s=>s.immediate_death?1:0)),
                        Math.max(...scenarios.map(s=>s.root_cycle?1:0)),
                        // An imminent forced oscillation matters more than a
                        // speculative cycle much later in a greedy rollout.
                        // Otherwise the agent can enter a two-tile trap now
                        // to avoid a larger forecast cycle far in the future.
                        Math.max(...scenarios.map(s=>s.first_cycle_frames===null ? 0 : Math.max(0,32-s.first_cycle_frames))),
                        deaths,
                        deaths ? -Math.min(...scenarios.map(s=>s.frames)) : 0,
                        // A native forecast to the next pellet is a progress
                        // potential. Static distance alone repeatedly promises
                        // food behind a ghost while the live agent detours.
                        Math.max(...scenarios.map(s=>s.next_pellet_frames ?? horizon+600)),
                        Math.max(...scenarios.map(s=>s.no_pellet_cycles)),
                        moveRank(base,first,rootHistory,buffer)[3],
                        -scenarios.filter(s=>s.cleared).length,
                        -scenarios.reduce((sum,s)=>sum+s.pellets,0)/scenarios.length,
                        Math.max(...scenarios.map(s=>s.max_no_pellet_decisions)),
                        scenarios.reduce((sum,s)=>sum+s.frames,0)/scenarios.length];
                    policies.push({buffer,rank,scenarios});
                }
                policies.sort((a,b)=>compareRank(a.rank,b.rank));
                candidates.push({action:first,...policies[0]});
            }
            candidates.sort((a,b)=>compareRank(a.rank,b.rank));
            return {choice:candidates[0].action,candidates,transitions,
                algorithm:'native-rollout-mpc-food-routing-v2',horizon_frames:horizon,scenario_seeds:seeds,buffers,
                information:'Current native state; independently resampled future RNG. No real episode RNG lookup.',
                optimality:'Approximate policy portfolio; qualification must use complete native games.'};
        } finally {
            api.searchRestore(root);
            if(JSON.stringify(api.observe())!==JSON.stringify(before)) throw new Error('Teacher altered the live simulation');
        }
    };
}
