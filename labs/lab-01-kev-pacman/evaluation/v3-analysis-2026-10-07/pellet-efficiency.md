# Native-v3 pellet efficiency versus native-v2

Native-v3 learns some recovery and reversal behavior, but pellet routing has not improved reliably. Across the same 20 full native games it collects only eight more pellets than v2 while taking 40% more decisions and 35% more simulation frames. Every game still ends in game-over. The main progress failure is long wandering through already cleared corridors; the strict repeated-cycle metric alone understates it.

## Evidence and scope

The source is the backed-up `comparison-v2-v3-20261007-190200-41f4f71f` session. Both reports use identical protocol, levels 1/2/3/5, five reserved seeds, adjacent-tile-entry actions and the v2 observation. Both identities say completed 762-step Pac-Man adapters on the same pinned Qwen3.5-4B backbone. All 40 trace summaries were recomputed and match the saved metrics after JSON key normalization; each trace retains a constant adapter identity. The session archive was hash verified by the root analysis.

Additional tests below are **CPU native-engine teacher branches**, not GPU model reruns or new training. Four selected learner prefixes matched the saved observations exactly. The teacher source hashes match its qualification receipt, and it uses the frozen qualification options. No benchmark state was added to training. Raw private trajectories remain outside this report.

## Paired whole-game results

| Measure | Native-v2 | Native-v3 |
| --- | ---: | ---: |
| Level clears / games | 0 / 20 | 0 / 20 |
| Game-overs | 20 | 20 |
| Pellets collected / 4,880 available | 2,639 (54.1%) | 2,647 (54.2%) |
| Decisions | 7,959 | 11,129 |
| Native simulation frames | 56,471 | 76,493 |
| Pellets per decision | 0.332 | 0.238 |
| Pellets per 1,000 native frames | 46.73 | 34.60 |
| Mean longest pellet-free interval | 84.65 decisions | 123.35 decisions |
| Maximum pellet-free interval | 131 decisions | 273 decisions |
| Pellet-free intervals of at least 32 decisions | 53 | 88 |
| Pellet-free intervals of at least 128 decisions | 2 | 8 |
| Decisions made after at least 32 dry decisions | 1,853 | 3,973 |
| Raw dry-cycle detections | 0 | 16 |
| Longest consecutive dry-cycle streak | 0 | 4 |
| Reversals | 13 (0.16% of decisions) | 248 (2.23%) |
| Reached lives collecting zero pellets | 5 / 60 | 9 / 60 |
| Power pellets collected / 80 available | 35 | 32 |
| Ghosts eaten | 9 | 16 |
| Fruit eaten | 3 | 6 |

Pellet/action efficiency falls 28.3%; pellet/frame efficiency falls 26.0%. V3 has worse pellet/frame efficiency in 18 of 20 paired cases. It collects more pellets in 11 cases, fewer in eight and ties one. Level 3 improves total pellets from 612 to 848, but frames rise from 11,323 to 21,530. Level 1 falls from 695 to 524 pellets, and level 5 from 650 to 562.

First-life pellets fall from 91.45 to 69.70 per game. Post-respawn pellets rise from 40.50 to 62.65: v3 does recover additional food on later lives, but this does not translate into better complete-game efficiency. Longer survival changes the distribution of states encountered. Aggregate efficiency therefore diagnoses the realized policy trajectory; it is not a controlled comparison at identical intermediate states. Decision and frame denominators must both remain visible because reversals can take fewer native frames than normal corridor moves.

V3's 16 cycle detections are all period-two reversals, and no consecutive streak reaches the teacher gate of eight. The larger problem is long, irregular travel without food, which can involve many distinct tiles and fail to repeat an identical period-2-to-64 sequence. Reporting only “loops = 16” would miss this.

## Representative exact-state failures

### V3 level 2, seed 50033: unnecessary empty-corridor detour

Turns 409–681 consume **no pellet for 273 decisions / 1,827 native frames**, with 123 pellets still on the board. The path visits 153 distinct tiles and makes 120 revisits. A dangerous ghost is more than six maze tiles away in 166 of those decisions; distance alone is only a proximity diagnostic, not a safety proof.

At turn 437, Pac-Man is at row 26, column 15, heading down. Right immediately enters a tile containing a pellet; left enters empty corridor. All three legal first actions survive the immediate transition. V3 chooses **left with probability 0.6263**, versus right 0.3592. The frozen teacher chooses **right**: its best right rollouts obtain the next pellet in five frames; its best left/up rollouts obtain no pellet within the 240-frame search horizon.

The exact native continuation makes the failure concrete:

| From identical native turn-437 state | Recorded v3 suffix | Frozen teacher continuation |
| --- | ---: | ---: |
| Next pellet | 246 actions / 1,650 frames | 1 action / 5 frames |
| Continuation actions | 595 | 236 |
| Continuation frames | 4,072 | 1,584 |
| Pellets consumed | 73 of 123 | All 123 |
| End | Final life lost; 50 pellets left | Level cleared; no deaths |
| Longest dry interval | Original long dry spell continues | 22 actions |

At turn 452, row 32/column 12, v3 again picks left (0.6193) into empty corridor. Right immediately eats a pellet, and the teacher again picks right. This is evidence of weak state-dependent junction selection and food-route commitment, rather than missing game physics.

### An immediate pellet skip is not automatically an error

At turn 572 of the same game and the same row 32/column 12 junction, right again eats a pellet safely during its first transition. The teacher nevertheless agrees with v3's **left** choice. Its right rollouts both die about 245 frames later; its left rollouts survive the search horizon. Thus the benchmark's “safe immediate pellet skip” is an opportunity counter, not a count of proven policy mistakes. A useful analysis must preserve the teacher's forward safety reasons.

### Post-respawn food route missed

At level 1, seed 50021, turn 100, row 23/column 18, v3 chooses left (0.7367) into empty corridor although right eats a pellet immediately. The teacher chooses right: the best right scenarios eat 15 pellets over about 247 frames; the best left scenarios eat one over about 242 frames. Recorded v3 waits another 56 decisions / 415 frames for its next pellet. This connects the improved post-respawn total with a remaining recovery-navigation weakness.

## Mechanics, observation and capability diagnosis

All 19,088 recorded decisions have zero nonterminal stationary or multi-tile transitions. No observation places Pac-Man on an unconsumed normal or power pellet. The pinned native player consumes a dot when its tile is entered; the controller's legal adjacent-entry boundary agrees with that mechanism. The traces therefore do not support a pellet-consumption or movement-endpoint bug. Pellets remain because the policy chooses other paths.

The observation includes the full remaining-pellet maze, player and ghost movement/timers, power-pellet locations, 64 recent positions, destination visit counts and decisions since the last pellet. Option criteria name each destination row and column. The demonstrated issue is using this information to rank the next route; it is not evidence that the raw API state omits pellets. Auditing the final encoder/rendered input is still appropriate before asserting perfect perception.

At junctions, v3 skips a surviving immediate food action 266 times versus 172 for v2: 14.4% of its 1,847 junction decisions versus 12.5% of v2's 1,378. Of those opportunities, the nearest dangerous ghost is farther than six maze tiles away in 215 versus 128. The teacher branch at turn 572 shows why these counts need lookahead or teacher adjudication. They cannot be relabeled blindly as “always take the dot.”

The evidence supports four remaining progress capabilities to test explicitly:

1. **Pellet grounding at junctions:** connect option destination coordinates to the current grid cell and distinguish a pellet from already cleared corridor.
2. **Route commitment through empty corridors:** choose a safe route toward remaining food and continue until progress; avoid repeatedly choosing a familiar left turn when food remains on the right.
3. **Dry-spell recovery:** use recent history, visit counts and the dry counter to change a failing route. V3 frequently continues roaming after 32 or more dry decisions.
4. **Useful post-respawn recovery:** v3 adds later-life progress on average, but nine reached lives eat nothing. Recovery must lead to food and eventual completion, not only prolong play.

Power behavior remains incomplete: v3 eats more ghosts but reaches fewer power pellets. This is a mixed realized outcome, not proof of deliberate improved pursuit. Neither adapter ever reaches 30-or-fewer remaining pellets, so these benchmark trajectories provide **no evidence of learned endgame cleanup**.

## Implications for the next experiment

The v3 recipe is a one-step teacher-label imitation round. It forces coverage of safety-critical reversals, respawn, stalls and power expiry, but has no explicit minimum for food-opportunity or endgame states. Its 24 roots per learner game include only a few late-maze roots when those states exist; the learner's failure to reach the endgame limits that source. Qualified teacher suffixes may supply such states, but their coverage should be counted before deciding it is sufficient.

Keep the reserved benchmark games as evaluation only. On fresh v3 learner games, collect teacher-adjudicated junction opportunities and full verified recovery suffixes, plus explicit food-grounding, long safe transit, dry-spell recovery and late-maze cohorts. Include justified food skips as well as corrected skips so the student learns safety-dependent progress. Evaluate an observation ablation that exposes derived per-action pellet/route and ghost-arrival features; it tests spatial grounding without using future episode RNG. A route target or route-commitment representation is another controlled candidate, not a demonstrated fix.

Promote pellet/frame and pellet/action efficiency, zero-food lives, dry-spell distribution and completed-maze counts to the comparison view. Keep no-pellet cycles separately. Qualify any stronger teacher or changed input protocol before distillation, and compare teacher at exact learner states as above. The current results do not justify another training run on the same mixture solely because the score rose.

The later [paired frozen-teacher reference](teacher-reference.md) uses the same
20 starting states and independently verifies every native trajectory. It clears
all 20 mazes, collects all 4,880 pellets, loses one life, and has zero raw dry
cycles. Its aggregate pellet/frame rate is 0.07409 versus v3's 0.03460. The
longest dry interval is 112 decisions, so even this stronger controller has
imperfect progress. These complete games strengthen the exact-state branch
evidence without turning the held-out traces into training data.
