# Native-v3: training coverage and next experiment

V3 learned a useful local retreat behavior, but its correction round mostly
trained expert-owned recovery states rather than the states where the learner
made mistakes. That distinction is measurable: only **134 of 6,096 training
requests were exact learner-visited roots**, and only **71** disagreed with v2.
The first change should be stronger learner-disagreement and sequence coverage,
with explicit safety/progress intersections. Increasing LoRA rank or replacing
the loss with CLM is lower priority.

This audit uses the completed October 7 session restored from the verified Drive
archive. The private archive and raw trajectories remain outside the repository.
[Derived counts](training-coverage-evidence.json) are bound to the v3 training
JSONL hash `35474b35c6106da5578a0900fe8f9391c3d5350ce8cba68098ffe37e4964ef6b`.
These are CPU data/trace analyses; they are not a new GPU experiment.

## What changed in the completed experiment

The unchanged 20 native starts compare levels 1, 2, 3 and 5 with five reserved
seeds under observation/action protocol v2. Neither policy receives a safety
override. Both lose all 60 initial lives and clear zero mazes. V3 reduces
avoidable immediate deaths from 52 to 22; its conditional fatal-choice rate
falls from 52/174 = 29.9% to 22/365 = 6.0%. The opportunities arise on different
trajectories, so that rate is a useful diagnosis, not a paired causal estimate.
Reversals rise from 13 to 248. This agrees with the new critical-reversal labels.

The same comparison gives 2,639 versus 2,647 pellets, but 7,959 versus 11,129
decisions: approximately 0.332 versus 0.238 pellets per decision, a 28% decrease
in this aggregate efficiency measure. Mean first-life food falls from 91.45 to
69.70; second-life food rises from 22.45 to 46.55; third-life food falls from
18.05 to 16.10. The longest pellet-free interval rises from 131 to 273 decisions,
and raw dry-cycle detections rise from zero to 16. More time alive and more
retreats did not translate into maze completion or efficient food routing.

## Actual mixture, not the planned headline

V3 warm-starts v2's rank-16, all-target LoRA and 256-dimensional pointer head.
It creates a new optimizer/scheduler rather than resuming v2 optimizer progress.
The published lab recipe is one epoch at `2e-5`, microbatch 4 × accumulation 2,
BF16 autocast with FP32 stored weights, checkpointing and the pinned optimized
kernel stack. Actual stdout confirms that **496 v2 LoRA tensors and the pointer
head were loaded**, followed by 2,000 replay requests mixed with 4,096 v3 task
requests. The trainer declares 6,096 admitted requests. The final metric reaches
step 762/762, epoch 1.0 and `records_seen=6096`; its 1,524 batch records account
for 6,096 distinct source IDs, each exactly once, including every one of the
4,096 expected task IDs. There are no drop, truncation, rejection or overflow
messages. This provides direct log evidence that the full mixture was trained,
rather than an inference from the optimizer-step count alone.

The largest encoded row across all 1,524 batch records is 2,658 tokens, below
the 4,096 state-token ceiling. The pinned trainer calls the encoder with
`strict=True`; oversized states raise rather than silently truncate. Every
admitted request appears once and there is no overflow or filtering message.
These logs therefore do not support training-state truncation as an explanation
for v3's failures. The lengths are observed post-encoding batch maxima, not a
new independent tokenizer pass over the private dataset.

The completed status records 3,990.3 seconds elapsed (about 66.5 minutes), and
the final metric reports a 30.24 GiB peak. Stdout ends with the native-v3 save
acknowledgment. The checkpoint archive is larger than the connector's download
limit, so its stored `training_metrics.json`, `training_config.json` and parent
checkpoint hash were not independently inspected. The log names the intended
v2 warm-start path; it does not independently prove that parent's file hash.
The JSON evidence contains the stdout SHA256 and exact observed log fields.

| Training input | Requests | Share of 6,096 |
| --- | ---: | ---: |
| Unchanged canonical v2 expert examples | 2,048 | 33.6% |
| New teacher-owned recovery suffix states | 1,914 | 31.4% |
| Exact learner-visited roots | 134 | 2.2% |
| Generic decision-v7 replay | 2,000 | 32.8% |

The 2,048 new examples contain 512 immediate safe/fatal decisions and 289 critical
reverse labels. However, **482 of those 512 critical states and 262 of those 289
critical reversals belong to teacher-owned suffixes**. Only 30 exact learner
roots are critical, 27 have critical reverse labels, and 26 are critical
learner/teacher disagreements. Those 26 records represent 0.43% of the complete
training mixture. There are 71 learner disagreements overall (1.16%).

This is still a substantial safety-coverage increase over v2: its original 4,096
task examples contained only 208 immediate-critical states. The v3 result is
consistent with the added supervision helping local danger/retreat decisions.
It does not show that the student has learned to reach safe routes several
moves before an encounter becomes immediately fatal.

New task coverage is heavily post-respawn: 1,459/2,048 examples (71.2%). Only 589
new examples are first-life states. This is a plausible contributor to the
observed recovery gain alongside weaker first-life food routing, but one run
does not establish that causal explanation. Coverage cohorts overlap.

## Admission succeeds; final selection loses many useful roots

The generator replayed 576 teacher recovery branches: 384 training and 192
development. It accepted 362 training branches (94.3%) and 171 development
branches (89.1%). Rejected branches retain their traces and supply no labels.
The 43 rejected receipts list 33 incomplete mazes and 10 pellet stalls.

| Training root cohort | Attempted | Accepted | Finally selected as a training root |
| --- | ---: | ---: | ---: |
| Immediate safe/fatal opportunity | 85 | 77 | 30 |
| V2 chose an immediately fatal action with a safe alternative | 44 | 40 | 24 |
| Current pellet stall | 125 | 117 | 79 |
| Power about to expire (≤60 frames) | 55 | 55 | 39 |
| Post-respawn | 262 | 242 | 91 |

These categories overlap. All 134 selected training roots are accepted branches,
but there are 362 accepted training roots in total. The sampler's low root and
disagreement floors, followed by a large suffix pool, explain much of the
missing learner-state emphasis. The teacher qualification gate is not the main
reason that only 30 risk roots reached training: 77 such roots were already
qualified and available.

Root selection does include precursor states one, four, eight and sixteen
decisions before an avoidable death, within the same life. It takes at most
three roots from each cohort before filling a 24-root-per-game budget.
**Pre-trap exposure has no final coverage floor and its identity is not preserved
as a distinct coverage stratum.** A branch qualifies on complete recovery, but
the final dataset can retain distant suffix states while discarding its root.
Full recovery proof is valuable; it does not ensure that the imitation mixture
focuses on the observed failure mechanism.

## The power-expiry intersection is nearly absent

The training split has 408 frightened states with ≤60 frames remaining, so power
expiry is present. It has only **seven states that are both power-expiry and
immediate safe/fatal decisions**: six new teacher suffix states and one old
expert state. There are zero such exact learner roots.

Within the new half, 39 learner roots are broad power-expiry states, but all are
immediately noncritical. Four learner roots have ≤5 frames left; none is
critical. Teacher suffixes include 23 states with ≤5 frames left, four critical.
This does not cover the hazardous timing combination well: an edible ghost
becomes dangerous during the duration of the selected tile-entry action.

The ghost audit identifies this exact failure in the held-out v3 trajectory
(level 3, seed 50047, turn 218): three power frames remain, the student chooses
right with probability 0.9761 and dies, while left survives and the frozen
teacher selects left. The clock is present in the observation. Broad frightened
or expiry quotas should not be treated as sufficient exposure to this joint
timing hazard. Held-out trajectories remain diagnostic evidence only; do not
add them to a later training split.

## State information and teacher capability

The v2/v3 observation supplies the full remaining-pellet maze, named ghost
positions, headings, modes and frightened flags; native pixel offsets, speed
phases, pending headings/reversals/releases; current power, commander, release
and Elroy clocks; player alignment and eat pause; 64 recent positions,
pellet-free duration and destination visit counts. It also supplies the legal
directions and their adjacent coordinates. No explicit wait action exists for
either student or teacher.

The raw information is extensive, but **availability is not demonstrated use**.
The model still has to derive maze routes, collision timing and escape topology
from ASCII rows and numeric JSON. Per-action nearest-food distances, corridor
escape structure and relative heading/distance summaries are not currently
provided. Current-state BFS summaries would be an auditable representation
ablation. Teacher future trajectories or exact counterfactual collision results
should stay out of the input unless a separate planner-assisted policy is
explicitly defined and measured.

The instruction also asks the player to avoid “reversing without progress.” A
necessary retreat often consumes no pellet immediately. The labels override
this prior in v3, but an instruction ablation should phrase the exception
explicitly: survival first; a temporary retreat is allowed; then resume food
progress. This is a hypothesis to test with matched states, not proof of the
failure cause.

The teacher is native rollout MPC with cached BFS maze distances. For each legal
first move it tries four food-routing buffers (1, 2, 4, 6), each under two
independently resampled future RNG scenarios. The horizon is 240 frames normally
and 480 for nearby danger or the last 30 pellets. Its internal rollout policy
checks exact one-action survival, then ranks danger separation, dry cycles and
food routing. Root ranking prioritizes immediate safety and actual imminent
cycle avoidance before predicted deaths and next-pellet time. It replans after
every move. This finite policy portfolio is not globally optimal and does not
prove that every escape is found.

The archived qualified teacher cleared **20/20** full native games with zero
deaths, zero avoidable deaths and zero raw dry cycles. It collected all 4,880
pellets, all 80 power pellets and 44 ghosts; maximum pellet stall was 67
decisions. The suite took 511.5 CPU seconds. It uses the same levels and native
observation/action mechanics, but its actual qualification seeds are
91009/92021/93031/94033/95047, whereas the student comparison uses the reserved
500xx starts. This is strong finite-suite teacher evidence, not a paired
teacher-versus-v3 result. The subsequent
[paired teacher reference](teacher-reference.md) now runs the exact 20 reserved
model starts: 20/20 clears, one life lost, zero immediately avoidable deaths,
all 4,880 pellets, no raw dry cycles and maximum dry interval 112. All initial
states match both student traces and every teacher trajectory is independently
replay-verified. These new trajectories remain evaluation-only. They demonstrate
a strong finite-suite reference while retaining the teacher's loss and detour.

## Cross-entropy already supplies competing actions

Kev minimizes `−log p(a_teacher | state, legal actions)` with
`p_i = softmax(z)_i`. Its gradient for each action score is
`∂L/∂z_i = p_i − 1[i = a_teacher]`. Thus an overconfident dangerous competing
move already receives a strong penalty when its corrected state is included.
Every legal dangerous direction remains in the option set; no teacher safety
filter removes it from student training or inference. The pinned trainer also
shuffles legal option order during augmentation, even with none/distractor
probabilities set to zero.

The current loss does not distinguish a slightly suboptimal safe alternative
from a fatal alternative beyond the hard teacher label, and uniform example
weighting does not express safety severity. Better correction selection,
explicit critical-example weighting/stratified sampling, and properly validated
teacher multi-action targets are direct ablations within this architecture.
CLM-style cross-example contrastive learning is not necessary to make the
current correct-versus-dangerous action comparison. Changing architecture/loss
before testing these simpler mechanisms would confound the diagnosis.

## What the training loss does and does not establish

The monitor differences Kev's running loss accumulator at each optimizer step,
before its ten-step reset. Therefore each logged CE is the record-weighted mean
over **both accumulated microbatches (eight source requests)**, not just the last
microbatch. Each source request averages its questions first. The first 100
steps average 0.344 and the last 100 average 0.392; the full epoch averages
0.339. These are different, shuffled examples from a v2 warm-start mixture,
not repeated measurements of the same boards. They show no clear aggregate
downward trend and do not establish either memorization or insufficient
capacity. Only 30 steps contain eight Pac-Man requests, none contains eight
learner roots, and the 134 roots occur in 124 mixed steps. The logs cannot
attribute a batch's loss to its root, dangerous alternative, teacher suffix or
generic replay record; cohort-specific CE claims would be invalid.

The observed next-step learning rate peaks at approximately `2e-5` around step
76 and decays to near zero by completion, as the intended one-cycle schedule
requires. Global gradients are clipped at norm 1 on 683/762 steps (89.6%);
the median pre-clip norm is 6.23. All logged values are finite. Clipping is
frequent, but this alone proves neither divergence nor a rank bottleneck.
The run has no fixed-state training/development loss comparison or per-cohort
teacher-agreement evaluation, so **more updates or a higher rank currently has
no demonstrated empirical justification as the first fix**. Measure whether the
completed model can fit the actual corrected critical roots and fresh related
development states first. Low fit on those states would motivate an update/
capacity ablation; good fit accompanied by full-game failures would instead
support learner-distribution and multi-step progress corrections. The present
data-selection mismatch is directly measured, while underfitting remains an
unresolved hypothesis.

## Ranked next experiments

1. **Make the correction mixture root- and sequence-focused.** Start with the
   already accepted, training-only branches: preserve every verified
   learner/teacher disagreement root and all accepted critical roots before
   filling with suffix states. Collect fresh v3 learner games on new training
   and development seeds, then verify teacher corrections at the learner's
   preceding decision windows as well as the fatal/stall endpoint. Establish
   mandatory joint quotas for critical learner disagreements, pre-trap escapes,
   currently-frightened ghosts expiring during an action, and stalled junction
   food choices. Keep each episode and its related forks in one partition.
   Do not satisfy learner-state quotas with expert suffixes. Retain the full
   recovery gates and failed evidence. This addresses the measured bottleneck
   before generating a much larger expert-only dataset.
2. **Separate local safety learning from food routing.** Use the same data and
   initialization to compare ordinary CE with modest critical/disagreement
   sampling weight, while protecting ordinary first-life routing examples.
   Require improved full-game progress as well as lower local fatal rate.
   Tune on fresh development games; report first-life/respawn food, decisions
   and simulation frames per pellet, stall distributions, and full clears.
   A safety improvement with unchanged clears and worse food efficiency does
   not qualify as a complete behavior improvement. Keep an untouched old
   teacher-food subset and generic replay retention check.
3. **Test a current-state representation ablation.** Add deterministic local
   destination food/power flags, BFS distance to remaining food and dangerous
   ghosts, relative headings, current power deadline and action-duration
   ingredients. Train both variants with identical teacher labels and budgets.
   Validate the same formatting in training and live play, including deadline
   crossings and necessary retreats. Measure option-order invariance and
   critical-state teacher agreement before raising LoRA rank.
4. **Only then adjust training capacity/objective.** If the model cannot fit
   the selected critical development cases, compare more updates or a small
   rank increase under a fixed dataset. If it fits them but fails full games,
   prefer another learner-state aggregation round, longer-horizon escape
   targets or route-intent state over capacity changes. Consider soft targets
   for demonstrably equivalent safe teacher actions; do not convert the
   teacher's lexicographic rank tuple into probabilities without validation.
   A contrastive objective is a later controlled research option, not the
   prerequisite fix.

Continue from v3 for a focused next correction round unless development evidence
shows irrecoverable routing regression; keep v2 as the frozen comparison. A
controlled v2-versus-v3 warm-start ablation can isolate retained routing versus
new retreat knowledge. No training or implementation changes were started by
this audit.
