# Rebalance v4 around measured choices

The previous `pacman-native-v4-offline` export remains an immutable reference.
Its data and replay checks passed, but its sampling did not adequately represent
the behaviors being corrected. Nearby-ghost and low-power primary categories
accounted for 71.94% of its 3,660 targeted roots. Immediate safe/fatal choices
accounted for 4.48%, and actual chosen-action expiry near a ghost for 0.38%.
The 1,220 context windows contained 1,201 lead-ins and only 19 follow-throughs.
Primary categories also concealed overlapping food-routing states.

The new dataset is named `pacman-native-v4-balanced`. Keep the native engine,
observations, legal directions, qualified teacher and its options unchanged.
Generation runs locally on CPU. No student predictions are invented, and no
v4 GPU training has run. Continue from completed native-v3 with a fresh
optimizer and a separate checkpoint only after full admission passes.

## What qualifies as a behavior case

| Behavior | Required evidence |
| --- | --- |
| Immediate collision avoidance | Native same-state legal alternatives include a fatal action and a surviving teacher action; the teacher's complete positive continuation clears without losing a life. |
| Anticipatory escape | Both first moves immediately survive, but the forced alternative followed by the same frozen teacher causes extra life loss or fails to clear; the teacher-first branch clears without life loss. Preserve and independently replay both continuations. |
| Dangerous power-expiry crossing | Native transition measurements establish expiry in the hazardous alternative, rather than inferring it from wall-clock action duration. A safe teacher continuation must clear without life loss. |
| Escape followed by pellet progress | Exact same-life teacher actions establish escape from a threat and subsequent pellet collection within a declared bound. Preserve the sequence and its source indices. Respawn or a changed heading alone is insufficient. |
| Productive branch choice | A legal multi-option native state offers a teacher move that collects a pellet and another surviving move that collects none, in a non-opening/revisited/dry routing context. The positive full-game suffix wins without life loss. This is measured local progress, not proof that every safe alternative is globally harmful. |
| Sparse-pellet cleanup | A meaningful native route choice toward remaining pellets, with a complete winning positive suffix. A remaining-pellet count alone is insufficient. |

Keep all legal options in each model request. Labels, counterfactual outcomes,
future progress and teacher search stay in targets/evidence, outside the input.
Rejected teacher recoveries remain recorded and supply no positive labels.
Legal wrong turns and delayed retreats create coherent off-policy states;
do not edit board JSON into impossible states or train on reserved benchmark
trajectories.

## Sampling constraints

Keep 3,660 targeted roots, 1,220 nearby informative decisions and 912 prior
Pac-Man replay requests. Kev adds 304 generic replay requests: 6,096 total,
762 updates. Development retains 512 requests. Keep rank 16, all-module LoRA,
the pointer head, option cross-entropy and the previous execution recipe.

Raise immediate-critical roots to at least 512 training / 64 development.
Limit nearby-ghost/low-timer-only exposure roots to at most 15% of targeted
data. At least 640 training / 64 development context decisions must be measured
post-escape progress/follow-through, not arbitrary later timestamps. Preserve
bounded same-life sequences and link them to selected roots; do not duplicate
already selected requests to manufacture windows.

Assign exactly one primary training role per request while retaining all
eligibility tags and mixed-scenario counts. Enforce disjoint seed partitions,
exclude identical model inputs across splits, and cap level/seed families at
128 selected requests. Avoid selecting hundreds of consecutive decisions from
one successful game. Development must cover the admitted behavior roles and
rare joint situations; zero examples means no per-role generalization claim.

The first collection targets were 512 immediate evasions, 256 anticipatory
escapes, 128 expiry evasions, 512 escape-to-food cases, 1,152 productive choices,
700 cleanup cases and 400 exposure cases. These were collection targets, not
published counts. Completed native simulations measured a much lower yield for
paired anticipation and expiry than for immediate avoidance and food routing.
We therefore freeze this explicit, measured-yield selection plan:

| Primary root role | Train | Development |
| --- | ---: | ---: |
| Immediate evasion | 640 | 96 |
| Anticipatory escape | 64 | 16 |
| Actual native power-expiry evasion | 16 | 4 |
| Retreat followed by food | 512 | 48 |
| Productive routing | 1,408 | 128 |
| Sparse-pellet cleanup | 888 | 78 |
| Residual broad exposure | 132 | 14 |
| **Targeted roots** | **3,660** | **384** |

The 16/4 expiry allocation is a genuine evidence limit, not broad coverage of
expiry behavior. Four development examples cannot support a strong independent
generalization claim. All quotas still require exact-input uniqueness, the
family cap and measured post-escape windows. A shortage stops export; easy
states, duplicate records and weakened evidence cannot fill it.

## Completed local collection

The frozen core has 376 completed proofs: 145 unique admissions and 231 rejected
recoveries. Admissions comprise 81 training / 38 development anticipatory
escapes, 16 / 6 expiry evasions, and 4 / 0 productive retreats. These source
states span levels 1, 2, 3 and 5, with 42 training and 15 development source
families. The separate recorded-v3 recovery catalog admitted 11 of 12 proofs:
four immediate-evasion corrections and seven productive-junction corrections.
Those roots retain the original checkpoint-bound predictions; surrounding
teacher states do not claim new learner disagreement.

Every admission passed complete zero-loss native recovery, the original loop
and stall gates, independent cold replay and source/hash checks. This is teacher
qualification evidence. It is not evidence that a new learned adapter improves.
The exact export has now passed selection, portable provenance validation
and a token-length audit. See [the completed export report](v4-balanced-export-2026-10-08.md)
for the selected counts, limits, hashes and Colab handoff.

## Local evidence available before the new pilot

Read-only scouting of the 114 independently verified zero-loss source games,
with exact-input deduplication and cross-partition exclusion, found:

| Existing native choice or sequence | Training | Development |
| --- | ---: | ---: |
| Immediate safe/fatal choice | 861 | 377 |
| Immediate-critical root followed by food within 24 teacher decisions | 695 | 297 |
| Teacher action collects food; another surviving action collects none | 14,893 | 5,583 |
| Same food advantage at a junction with at least three options | 2,768 | 1,029 |
| Same advantage, non-opening context, teacher turns from its current heading | 2,654 | 989 |
| Same food advantage with at most 30 pellets remaining | 2,445 | 922 |

These are overlapping candidate counts before the new role assignment and
family cap, not selected training quantities. Existing source evidence can
supply grounded routing choices. Missing early-trap and dangerous-expiry
branches need new native simulation and qualified recovery evidence. Merely
renaming existing proximity categories would not address the imbalance.

## Evaluation

Evaluate old and rebalanced variants on the same admitted, disjoint development
decisions, with per-role counts, CE, teacher probability and agreement. Full
native games remain the behavioral criterion: clears, deaths, avoidable
collisions, all-actions-fatal endpoints, first-life food, pellets per native
frame, dry intervals and sustained cycles. Keep the existing 20 reference
starts fixed for comparisons. Those games have informed diagnosis, so describe
them as a repeated reference benchmark rather than a blind final test.

No data-generation success or teacher qualification establishes improvement
in a learned player. The controlled experiment changes data selection and
retains the model/loss/training budget. Actual learner-disagreement filtering
remains a separately checkpoint-bound measurement when predictions are
available.
