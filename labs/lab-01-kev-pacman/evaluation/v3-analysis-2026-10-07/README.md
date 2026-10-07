# Native-v3 versus v2: safer retreats, weaker food routing

V3 learns useful immediate ghost avoidance, but it still clears **0 of 20 full
native games**. It collects almost the same number of pellets as v2 while using
40% more decisions. The strongest next experiment is a focused correction round
from v3, preserving the learner's actual mistakes and earlier trap-entry states.
The current evidence does not establish that a larger LoRA rank or CLM loss is
needed.

Three parallel audits examined [ghost evasion](ghost-evasion.md),
[pellet efficiency](pellet-efficiency.md), and
[training coverage and loss](training-coverage.md). They use the completed GPU
experiment's saved outputs. Additional native-engine teacher continuations ran
on CPU; no student inference, training or game implementation was changed.
The frozen teacher was also evaluated on the exact same 20 starting states,
with every resulting trajectory independently replay-verified.

## Full-game comparison

The two adapters play the same 20 reserved starts: levels 1, 2, 3 and 5, each
with seeds 50021, 50023, 50033, 50047 and 50051. The native observation and
adjacent-tile-entry action protocol are unchanged. Neither student receives a
teacher safety override. Every recorded game ends in game-over, rather than
being stopped by the evaluation limit.

| Measure | Native-v2 | Native-v3 | Frozen teacher |
| --- | ---: | ---: | ---: |
| Mazes cleared | 0 / 20 | 0 / 20 | 20 / 20 |
| Lives lost | 60 | 60 | 1 |
| Pellets collected, of 4,880 available | 2,639 | 2,647 | 4,880 |
| Decisions | 7,959 | 11,129 | 9,121 |
| Native active simulation frames | 56,471 | 76,493 | 65,865 |
| Pellets per decision | 0.332 | 0.238 | 0.535 |
| Pellets per 1,000 native active frames | 46.73 | 34.60 | 74.09 |
| Fatal move with an immediately safe alternative | 52 | 22 | 0 |
| Death endpoint with all current actions fatal | 8 | 38 | 1 |
| Longest pellet-free interval, decisions | 131 | 273 | 112 |
| First-life pellets per game, mean | 91.45 | 69.70 | 243.15 |
| Raw pellet-free cycle decisions | 0 | 16 | 0 |

In states offering both safe and fatal immediate moves, the realized fatal-choice
rate falls from 29.9% to 6.0%. Safe reversals rise from 6/106 available
opportunities to 131/176. These policies visit different intermediate states,
so the rates diagnose behavior rather than prove a same-state causal effect.
Level-1 immediate safety worsens while the other levels improve. Overall
pellet/action efficiency falls 28.3%, and pellet/frame efficiency falls 26.0%.

The [new paired teacher reference](teacher-reference.md) closes the older
qualification suite's different-seed limitation. It loses one life at
level 5/seed 50047, then completes the maze. Its longest dry detour is 112
decisions despite eventual completion. The teacher is substantially stronger
on this finite suite, not universally safe or optimal. Its zero immediately
avoidable deaths do not prove that its one death was unavoidable from earlier
states. Four CPU workers completed generation and verification in 9.27 minutes;
those wall times are not model-inference speed measurements. Native active
frames consistently exclude ready/death animations for all three controllers.

## What remains missing

1. **Anticipate a trap before all exits become fatal.** V3's 38 terminal
   all-actions-fatal states do not make the earlier deaths unavoidable. At one
   repeated level-2 junction, v3 and the teacher have two immediately safe
   choices; v3 chooses up and reaches a fatal trap eight decisions later.
   The teacher chooses right and a complete continuation clears the remaining
   185 pellets without another death. Four selected pre-trap continuations,
   one per level, all clear without deaths. These establish recoverable cases,
   not a population recovery rate.
2. **Reason about power expiry during movement.** At level 3/seed 50047/turn
   218, the observation explicitly shows three frightened frames remaining.
   V3 chooses right with probability 0.9761; the native transition lasts seven
   frames, frightened mode expires, and Blinky collides with Pac-Man.
   Left survives and is the teacher's choice. The model must compare a current
   timer with what happens before its action finishes.
3. **Ground pellet routes and recover from dry spells.** From the exact
   level-2/seed-50033/turn-437 state, the teacher eats all 123 remaining pellets
   in 236 decisions and clears. The recorded student suffix takes 595
   decisions, eats 73 pellets and dies. Its next pellet takes 246 decisions,
   versus one for the teacher. The 273-decision dry spell involves 153 distinct
   tiles, so a strict repeated-loop metric misses much of the wandering.
4. **Turn a safe retreat into renewed food progress.** V3 reverses more and
   makes later-life progress, but first-life food declines and nine lives eat
   no pellets. Surviving the next transition is only one part of completing
   the maze. Neither student reaches the last 30 pellets, so this benchmark
   does not demonstrate learned endgame cleanup.

The traces support policy weaknesses rather than a pellet-consumption defect:
all 19,088 observations have no unconsumed pellet under Pac-Man, and no
nonterminal action has a stationary or multi-tile endpoint. The input supplies
the full pellet grid, ghost identities, headings, native offsets and timers,
plus recent positions and dry counters. Having those fields does not prove the
model uses them to infer connectivity or collision timing. An immediate food
skip is also not automatically wrong: at another sampled junction, the teacher
agrees with an empty-corridor move because the food route later dies.
Training logs show a largest encoded row of 2,658 tokens; the pinned strict
encoder completes every request without truncation or overflow. The evidence
does not support silently cropped training states as the cause.

## Why the existing correction round falls short

The completed run loads 496 v2 LoRA tensors and the pointer head, then trains
all 6,096 admitted requests exactly once over 762 optimizer steps. The log
records about 66.5 minutes and a 30.24 GiB peak. The mixture is:

| Source | Requests | Share |
| --- | ---: | ---: |
| Existing v2 expert states | 2,048 | 33.6% |
| New teacher-owned recovery suffix states | 1,914 | 31.4% |
| Exact learner-visited roots | 134 | 2.2% |
| Generic decision replay | 2,000 | 32.8% |

Only 71 requests are learner/teacher disagreement roots; just 26 are critical
disagreements. Teacher qualification accepts 77 training roots offering both
safe and fatal moves, but final sampling retains only 30. The sampler discards
useful verified roots while filling with teacher-owned suffixes. There are 408
task-training power-expiry states, but only seven also offer both safe and fatal
moves, and none of those seven is an exact new learner root. Broad behavior
quotas do not guarantee exposure to the joint timing hazard seen above.

**Whole trajectories are retained as evidence, but not all their steps are sent
to training.** The generator selects a fixed request mixture from v2-owned roots,
qualified teacher continuations and replay. Each Pac-Man training request is one
observed board/history with its legal moves and teacher action label. It is not
a full-game sequence with a loss backpropagated across the trajectory. A teacher
continuation can contain hard states, but most selected new records are already
on the teacher's recovery path; they do not replace the original learner error
or its lead-in. A complete successful continuation proves a candidate correction
is usable, not that every later ordinary step needs to enter the dataset.

For the next sampler, select verified disagreement roots and early trap/expiry/
food-routing lead-ins first, then representative recovery-to-progress steps.
Deduplicate long easy corridor stretches and keep a bounded ordinary routing/
generic replay subset for retention. Report distinct roots and critical
disagreements separately from suffix count. More total teacher steps alone
would not establish more coverage of the states the learner got wrong.

Training loss cannot yet distinguish underfitting from this distribution gap.
Logged CE covers each whole eight-request optimizer step; different shuffled
examples and mixed cohorts prevent per-root fit conclusions. The intended
learning-rate schedule completes and gradients are finite. No fixed-state
training/development fit check establishes a rank bottleneck. Ordinary
within-option cross-entropy already penalizes competing wrong actions when the
corrected state is present.

## Does the noisy CE curve imply rank-16 capacity is exhausted?

No such conclusion follows from the current curve. Its final raw CE of 0.1252
at step 762 matches the saved metric. Successive points average eight different
source requests, mixing Pac-Man, teacher suffixes and generic decision tasks.
This is one shuffled pass starting from an already trained v2 adapter, rather
than repeated measurements of a fixed set. TensorBoard smoothing does not turn
it into a validation curve. The first and last 100-step mean losses are 0.344
and 0.392; their different examples prevent a fixed-state learning conclusion.

Rank 16 limits each learned projection update, not the complete backbone's
representations or the total policy to sixteen parameters. The recipe has
about 33.8M trainable LoRA/head parameters distributed across selected
projections. This does not guarantee adequate capacity, but a noisy minibatch
curve cannot establish that capacity is the limiting factor.

The measured issue is **coverage and selection**, not demonstrated bad teacher
labels: qualified learner mistakes are sparse and discarded, whereas the
teacher succeeds on the paired complete-game reference. Label ambiguity between
equally useful safe routes remains a possible separate quality issue to audit.

| Fixed-state probe result | What to investigate next |
| --- | --- |
| Poor fit on corrected training states and development states | Optimization/updates first; then a matched rank-capacity ablation |
| Good fit on corrected training states, poor related development fit | Coverage, generalization and state representation |
| Good local development fit, poor full-game completion | Learner-state distribution, early escape decisions and route-to-progress sequences |

Log CE and teacher agreement separately for critical retreats, pre-trap states,
expiry crossings, food routing and generic replay. If rank is tested later,
compare 16 and 32 on identical data, initialization lineage and update budgets,
holding LoRA scaling and dropout constant as well as the head and targets.
For example, keeping alpha/r at two requires alpha 32 at rank 16 and alpha 64
at rank 32. Simply changing rank while leaving alpha fixed also changes the
update scale. No rank ablation was performed in this analysis.

## Offline scenarios versus learner-state collection

The 20 reserved games are the benchmark, not a training-data ceiling. We can
generate many more training episodes and forks on separate seeds. Also,
**offline storage and off-policy collection are different choices**: a frozen
v3 can produce its own trajectories, then the teacher can relabel them later
and the next adapter can train from saved files. That is learner-policy data
collection followed by ordinary supervised training, not PPO or another
on-policy reinforcement-learning update.

| Collection source | Strength for this lab | Limitation |
| --- | --- | --- |
| Targeted teacher/scripted native scenarios, off-policy | Deliberately cover rare expiry, interceptions, sparse food and recovery; generate on CPU without running Kev | The chosen scenarios may miss states and histories caused by the learner's own errors |
| Current v3 rollouts relabeled by the teacher | Captures the actual learner's traps, biases and dry spells | Requires learner inference and may repeatedly miss rare hazards or late-maze states |
| Hybrid aggregation | Combines controlled coverage with real learner mistakes | Needs explicit source/cohort budgets so teacher suffixes do not crowd out correction roots |

Collecting expert labels at states visited by the evolving learner is the core
idea of [DAgger](https://proceedings.mlr.press/v15/ross11a.html). An alternative
is to perturb expert actions so it demonstrates recovery from mistakes;
[DART](https://proceedings.mlr.press/v78/laskey17a.html) studies that approach
with learned noise levels. For Pac-Man, a controlled wrong-turn/delayed-retreat
generator would be **DART-inspired**, not an implementation of its optimized
noise algorithm. These papers motivate data collection; their performance
guarantees or robot timings do not establish results for this Kev experiment.

Recommended scenario families are crossing ghost routes before immediate
danger, currently edible ghosts whose power expires before tile entry,
safe-versus-unsafe food junctions, retreat followed by food progress,
post-respawn recovery, and the last few scattered pellets. Harvest coherent
native snapshots from training episodes, then execute legal action
perturbations through the actual engine. Do not fabricate inconsistent ghost
positions, timers, pellet maps or recent histories. Preserve several decisions
before the failure as well as the recovery. Include justified food skips and
routine food progress, rather than labeling every safe immediate pellet as
mandatory.

Use cheap cohort screening and deduplication before expensive full teacher
continuations. Admit labels only after the teacher meets the recovery gate;
retain failed evidence and move earlier in the prefix when a root is already
unrecoverable. Keep all forks from an episode in one partition and vary timing,
heading, ghost identity, level and remaining-food topology. Parallel CPU
generation removes the need for Kev inference on these off-policy cases, but
teacher verification still has a cost. A small learner-data round after each
adapter update checks whether the new policy creates different mistakes.

## Proposed next experiment

Keep v2 and v3 immutable. Use v3 as the main warm start so its learned retreats
are retained; include a matched v2 warm-start control if resources permit.

1. Audit v3 agreement on the actual existing training corrections and on fresh
   development states. This first separates failure to fit known examples from
   failure to generalize to learner trajectories.
2. Collect fresh v3 games on new training/development seeds. Ask the unchanged,
   qualified teacher at disagreements, stalled junctions and earlier trap-entry
   windows. Retain complete-recovery admission and rejected-branch evidence.
   Preserve every qualified critical/disagreement root before filling suffixes;
   use explicit quotas for anticipatory escapes, expiry crossing an action,
   safe food-route choices, retreat-to-progress sequences and late-maze cleanup.
   Related episodes and forks stay in the same data partition.
3. Keep rank 16 and the CE recipe fixed for the first data-selection experiment.
   A concrete starting budget keeps the 2,048 old expert requests and 2,000
   generic replay requests unchanged. Reserve at least 1,024 of the 2,048 new
   requests for distinct, verified v3 learner roots. A starting allocation for
   the other half is 512 targeted off-policy native roots and 512 qualified
   teacher recovery/progress states. That raises exact learner exposure from 2.2% to at
   least 16.8% of the same 6,096-request budget. Collect more training episodes
   if necessary rather than meeting the quota by copying the same roots.
   Preserve critical and disagreement coverage within this budget.
   Then test modest critical-case weighting and deterministic current-state
   features separately: destination food/power flags, BFS food distances,
   heading/escape summaries and power/action timing ingredients. Match training
   and live formatting. Do not put future episode RNG or teacher rollout
   outcomes into a plain learned-policy input.
4. Compare complete native games against v2, v3 and the frozen teacher. Report
   clears, first-life food, deaths, pellet/frame and pellet/action efficiency,
   dry-spell tails and the safety/expiry cohorts. Tune on fresh development
   games; the 500xx benchmark traces examined here remain evaluation-only.
   Promote a model only when safety and food progress both improve.

More epochs, a higher rank, soft action targets or a contrastive objective are
later controlled ablations if fixed-state fit measurements justify them.
No next training run was started by this analysis.

## Reproducibility and artifacts

Source backup: `session-20261007-194339-69e9e579.zip`, SHA256
`77e9efd81c41d076fe54757d5ce93639e7ae0e9dd33dbeeb77e00a97251825b9`.
The archive and extracted entries were verified; private raw trajectories and
weights remain outside this repository. Both adapters report 762 completed
steps on pinned `Qwen/Qwen3.5-4B-Base` revision
`1001bb4d826a52d1f399e183466143f4da7b741b`. Native source revision:
`7407174c1d6a38be8cd230577489e39e0873145b`.

Machine-readable evidence:
[ghosts](ghost-evasion-evidence.json),
[pellets](pellet-efficiency-evidence.json),
[training](training-coverage-evidence.json),
[paired teacher](teacher-reference.json).
