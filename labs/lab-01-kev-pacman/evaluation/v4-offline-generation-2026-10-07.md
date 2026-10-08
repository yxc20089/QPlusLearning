# Completed local v4 teacher-scenario dataset

`pacman-native-v4-offline` was generated locally with the native JavaScript
simulator and qualified CPU teacher. No v4 GPU training has run. Continue from
the completed **native-v3 LoRA and pointer head with a fresh optimizer**, keeping
v2 as a comparator and writing a separate v4 output.

All 144 source jobs completed. The exported snapshot independently replayed
126 full source games, including legal setup prefixes, teacher actions,
observations, immediate counterfactuals and terminal outcomes. **114 qualified
with zero teacher-controlled life losses; 12 were rejected.** Rejection reasons
were 11 games with teacher-controlled life loss, one pellet stall and one
incomplete maze; reasons overlap. The other 18 source games completed after
this snapshot and remain in the raw backup. They supply no labels in this
export. Failed games remain evidence, outside the positive-label pool.

All 83 qualified training games and 31 qualified development games contribute
selected labels. Training and development use disjoint seed families; the
selector also excludes identical model inputs shared across those partitions.
The same maze and fixed opening can otherwise produce duplicates despite
different seeds. It removed 755 cross-partition inputs and found no conflicting
teacher labels. Fresh contributions are capped at 128 requests per level/seed
family; canonical prior replay is capped as well.

| Selected requests | Training | Development |
| --- | ---: | ---: |
| Targeted hard-scenario states | 3,660 | 384 |
| Nearby informative decisions | 1,220 | 128 |
| Canonical prior Pac-Man replay | 912 | 0 |
| Total in exported JSONL | **5,792** | **512** |

Kev adds 304 generic replay requests: 6,096 total training requests and
762 optimizer updates. The proposed recipe retains rank 16, all-module LoRA,
the pointer head, option cross-entropy, one epoch, lr 2e-5, batch 4 ×
accumulation 2, row budget 0, checkpointing and optimized BF16 execution.
Fresh targeted states comprise 75% of the new selected data.

Every selected example now has one primary scenario. Related windows inherit
their targeted root's category, with their own state tags retained separately.
The following counts are additive; each request appears in exactly one row.
The source-game count shows how many independently replayed training
trajectories contributed. A game can contribute to several categories.

| Primary scenario | Targeted train states | Related train windows | All train examples | Training source games | Dev examples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Immediate collision avoidance | 164 | 53 | 217 | 62 | 30 |
| Power expires during the chosen move near a ghost | 14 | 2 | 16 | 13 | 6 |
| Power may expire during a move near a ghost, without measured chosen-action expiry | 12 | 5 | 17 | 12 | 0 |
| Low remaining power clock, other positions | 1,047 | 353 | 1,400 | 83 | 137 |
| Nearby dangerous ghost | 1,586 | 534 | 2,120 | 82 | 233 |
| Routing after a pellet-free interval | 167 | 49 | 216 | 54 | 17 |
| Remaining-pellet cleanup | 670 | 224 | 894 | 82 | 89 |
| Prior Pac-Man replay | 0 | 0 | 912 | Not reclassified | 0 |
| **Total** | **3,660** | **1,220** | **5,792** | **83 distinct fresh games** | **512** |

Priority is immediate collision → actual near-ghost expiry → potential
near-ghost expiry → low power clock → nearby dangerous ghost → pellet stall →
sparse-pellet cleanup. This makes totals unambiguous while the overlapping
tags below preserve mixed situations. Respawn is a context tag, not a primary
hard scenario. The potential-only expiry category has no development example
in this export, so a separate generalization claim for that category is not
supported.

[scenario_catalog.py](../scenario_catalog.py) writes a checksum-bound summary
and per-example CSV containing the primary category, tags, parent/root links,
source episode/index, teacher move and input hashes. The private CSV maps each
training request back to its full trace; it does not change the training input,
labels, selected data or admitted manifest. The [public scenario catalog](v4-offline-scenario-catalog-2026-10-07.json)
contains counts and predicates without private examples.

| Targeted-state exposure; categories overlap | Training | Development |
| --- | ---: | ---: |
| Dangerous outside ghost within six graph steps | 1,760 | 201 |
| Frightened power clock has 1–120 frames remaining | 1,074 | 106 |
| At least 24 decisions since the last pellet | 549 | 54 |
| At most 30 pellets remain | 1,770 | 180 |
| After a native respawn | 1,705 | 203 |
| Both an immediately fatal and a surviving legal alternative | 164 | 22 |
| Potential power expiry during an action near a ghost | 26 | 4 |
| Actual power expiry during the chosen teacher action near a ghost | 14 | 4 |

Respawn alone never qualifies an easy state as targeted. Eating a ghost can
pause the power clock, so potential timing exposure and measured expiry are
separate. The input preserves every legal direction, including unsafe
alternatives; teacher plans, gold labels and future outcomes stay outside the
model input. Each selected label is backed by its independently replayed
winning, zero-loss source continuation. This proves the source teacher's
behavior, rather than global optimality or student performance.

The informative training windows contain 1,201 lead-in decisions and 19
recovery follow-through decisions; development contains 128 lead-ins. These
are bounded, same-life frames linked to selected targeted roots, rather than
complete easy trajectories. They are not balanced recovery sequences. Most
targeted states have no immediately fatal alternative, and the export has no
targeted joint immediate-critical/potential-expiry case. It therefore cannot
be described as resolving every v3 trap or power-expiry failure.

No v3 predictions are fabricated or included. These labels establish teacher
actions on targeted scenarios, not measured learner/teacher disagreements or
long-horizon harm from alternative safe routes. The separate
[checkpoint-bound disagreement experiment](v4-hard-disagreements-plan.md)
can make those measurements. A later sampling ablation could raise immediate-
critical floors from the current selected 164/22 to at least 300/48; the
verified pool has 861/377 available. Preserve this exported artifact when
testing a different allocation.

The builder's admission and portable evidence validation passed. All 14 CPU
exporter tests passed. The pinned Kev renderer and Qwen3.5-4B tokenizer audited
all **6,304** training/development requests: maximum state 2,403 tokens, maximum
estimated packed request 2,651 tokens, zero states above the 4,096-token
allowance. Actual CUDA training must still confirm zero truncation or dropped
requests. The unchanged 20 complete native benchmark games remain the
promotion test; snapshot agreement alone does not establish useful play.

The private export contains training/development JSONL, its manifest, the
full evidence ZIP and token audit. The [machine-readable report](v4-offline-generation-2026-10-07.json)
publishes counts, source bindings and artifact SHA-256 values without raw
private traces. The [local preparation instructions](v4-offline-hard-scenarios-plan.md)
describe generation, resumption and validation. Use its teacher-only validator;
the current disagreement notebook intentionally expects a different dataset.
