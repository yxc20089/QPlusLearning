# Native-v3 ghost-evasion audit — 2026-10-07

**V3 substantially improves immediate safe retreats, but does not solve survival.** Both adapters lose 60 lives and clear 0 of the 20 held-out games. V3 makes fewer fatal choices when an immediately surviving move exists; much more often it reaches a state where every current legal action is already fatal. That terminal metric does not mean the earlier death was unavoidable.

The source is the verified Drive session `20261007-194339-69e9e579`, archive SHA256 `77e9efd81c41d076fe54757d5ce93639e7ae0e9dd33dbeeb77e00a97251825b9`, comparison run `comparison-v2-v3-20261007-190200-41f4f71f`. The recorded adapters are native-v2 checkpoint `c100872bd94f9869807ed33bdb4617cc38867abf61c551a6ac507e8f38106d33` and native-v3 checkpoint `c7c5c8326f74e86e161e05730db5004efdfd3b2351c6f2f9478581cfc13550b0`, each at 762 optimizer steps. This analysis ran the pinned native engine on CPU; it performed no model inference or GPU training.

## What changed

| Full-game safety measure | V2 | V3 |
|---|---:|---:|
| Decisions | 7,959 | 11,129 |
| Life losses | 60 | 60 |
| Fatal choice with an immediately safe alternative | 52 | 22 |
| States with both safe and fatal immediate actions | 174 | 365 |
| Fatal-choice rate within those states | 29.9% | 6.0% |
| Death at a state where all immediate actions are fatal | 8 | 38 |
| Safe reverse selected / available in mixed-risk states | 6 / 106 | 131 / 176 |
| Avoidable fatal choices assigned probability ≥ 0.9 | 29 / 52 | 11 / 22 |
| First-life pellets, mean | 91.45 | 69.70 |

The conditional rate and the reversal behavior support a real improvement in reactive safety. The complete-game result does not improve. Different policies visit different states, so the conditional rate is not a same-state treatment comparison and must be read together with losses, first-life progress and the exact teacher queries below.

Safety improvement is uneven. At level 1 the mixed-risk fatal rate rises from 15/71 (21.1%) to 8/27 (29.6%). Levels 2, 3 and 5 improve from 60.9%, 38.9% and 20.5% to 2.6%, 3.3% and 6.2%. Every level still loses all 15 lives. The five seeds do not make every decision independent: V3's 22 avoidable-death roots contain 14 distinct observable states, and several initial mistakes recur identically across seeds.

[The new paired teacher reference](teacher-reference.md) runs the frozen teacher on the same 20 reserved starts and independently verifies every trajectory. Its aggregate results complement the local counterfactuals below; the older qualification suite is separate.

## Exact same-state teacher evidence

All 11,129 V3 requests were matched against an exact native replay. At every one of the 22 avoidable-fatal roots, the frozen teacher selected an action that survived the immediate native transition. Twenty of those roots had an immediately safe reverse. The student's mean probability for the teacher action was only 0.139; every teacher-selected action had probability below 0.5.

The teacher was also queried 1, 4 and 8 decisions before each of the 38 deaths with no immediately safe final action, within the same life. It disagreed with V3 at 12/38, 9/38 and 21/38 roots respectively. All 114 teacher choices were immediately nonfatal. Those differences identify planning candidates; a short MPC forecast alone does not establish a complete recovery.

Four disagreements eight decisions before forced-death endpoints were then tested with **complete teacher continuations** from the exact V3 state and episode RNG. Selection chose one per level with both short teacher scenarios forecasting survival. The unchanged teacher cleared all four with no additional life loss, no avoidable death and no repeated pellet-free cycle:

| Starting state: level / seed / turn | Remaining pellets cleared | Teacher decisions | Longest pellet-free interval |
|---|---:|---:|---:|
| 1 / 50021 / 335 | 162 | 322 | 30 |
| 2 / 50021 / 127 | 185 | 350 | 31 |
| 3 / 50023 / 649 | 62 | 210 | 49 |
| 5 / 50023 / 264 | 126 | 360 | 44 |

These are selected case studies, not a new unbiased teacher clear-rate estimate. They establish that concrete failures leading into a terminal trap could be avoided by earlier teacher decisions, without relying on the short rollout forecast. The reserved states remain diagnostics only.

Examples:

* **Missed urgent junction retreat:** level 1, seed 50021, turn 77. V3 chooses `up` with probability 0.9665. Exact native transitions show both `up` and `left` fatal, while `right` survives. The teacher chooses `right`, which V3 assigns only 0.0330. Pinky's current tile, heading, next heading and pixel offset are already in the request. This same mistake occurs at the other level-1 initial starts.
* **Power expiry during movement:** level 3, seed 50047, turn 218. The request says frightened is active with only **3 frames remaining**. V3 chooses `right` with probability 0.9761 and loses a life during a **7-frame action**; `left` survives and is the teacher choice. Exact replay confirms power reaches zero and Blinky becomes dangerous during the action. Both ghost status and the remaining timer were visible. A current-distance diagnostic reports no dangerous outside ghost here, so that diagnostic alone misses the expiry hazard.
* **Failure before a trap becomes unavoidable:** level 2, every reserved seed, turn 127. Both immediate choices survive, but V3 chooses `up`; the teacher chooses `right`. Eight decisions later, at turn 135, every legal action is fatal. The student gives the teacher's earlier move probability 0.2536. The repeated sequence supports an anticipation problem rather than a missing final-turn escape.

## Information versus learned capability

[The native observation](../../games/arcade-engine.js) already supplies the complete pellet/power-pellet maze, four named ghosts, current headings, modes, frightened flags, pixel offsets, next headings, movement phases, pending reversals/releases, power/phase/release clocks and recent positions. V2 and V3 use the same observation and action protocol. The V2 exact token audit reports zero states exceeding 4,096 tokens. [V3's actual training-log audit](training-coverage.md) covers 1,524 batches: the maximum encoded row was 2,658 tokens, and all 6,096 requests completed once under the pinned trainer's strict encoder, with no filter/overflow. There is no observed silent training truncation. This does not establish a separate serving-tokenizer audit.

Kev must infer maze connectivity, collision timing and an escape route from this representation. Direction criteria currently describe the adjacent destination; they contain no explicit ghost-arrival time or future collision annotation. Visible fields do not prove that the model has learned the native dynamics.

[The training-coverage audit](training-coverage-evidence.json) identifies a plausible exposure gap: the 4,096 task-training rows include 408 active-power states with at most 60 frames remaining, but only **7** also have both immediately safe and fatal actions. Six of those seven are new teacher-suffix states; none is a new learner-owned root. Broad expiry coverage therefore did not ensure coverage of the observed “currently frightened, lethal before action completion” failure. This is a dataset mechanism to test, not proof that it is the sole cause.

[The teacher](../../games/arcade-teacher.js) executes the native ghost mechanisms in lookahead, tests one-action survival inside its continuation policy, and compares routes under two independently resampled future RNG scenarios. The fixed configuration uses 240 normal frames and 480 danger/endgame frames, with buffers 1, 2, 4 and 6. No equivalent teacher or safety filter replaces Kev's live choices. Its qualification is finite-suite evidence, not universal survival: even at 11 of the 22 avoidable-fatal roots, both of its short chosen-action rollout scenarios forecast a later death, despite the selected action being safe immediately. Those roots need earlier recovery analysis.

## What the next correction round should test

Keep V3 as the parent for a controlled correction experiment: it has acquired useful safe-reversal behavior. Collect **fresh V3-owned trajectories** and label earlier trap-entry decisions, not just death endpoints. The reserved benchmark roots examined here must remain evaluation-only.

Prioritize explicit cohorts for (1) urgent safe retreat, (2) two-ghost/corridor interception several moves ahead, (3) expiring power versus the movement duration, and (4) retreat followed by renewed pellet progress. Require teacher continuation evidence before admitting labels, while retaining rejected recoveries. The existing entire-recovery admission rule can otherwise exclude precisely the hard learner states we need to diagnose.

Report first-life survival/progress, all-actions-fatal endpoints, expiry failures and same-state teacher disagreement in addition to the immediate safe/fatal metric. The observed improvement in that metric accompanied worse first-life pellet progress and zero wins. A new objective should not be selected from snapshot accuracy or score alone.

Within-option cross-entropy already pushes against competing legal moves. This evidence does not justify replacing Kev with CLM. Root exposure, anticipatory labels, critical-case weighting and representation ablations are the more direct experiments; they remain hypotheses until a controlled training run and unchanged complete-game benchmark validate them.
