# V4 collection sizing — running wave 1

This is a read-only sizing review of the local collection at
`local-data/pacman-native-v4-20261007/collection`. It does not change the running
generator, native game, teacher, source pins or admission rules. The snapshot is
**2026-10-08 03:03:57 UTC** (October 7 locally). Counts will grow as workers finish.

## What exists so far

Ten completed native teacher episodes, across five level-1 training families,
all pass the source-episode gate: ten clears, zero teacher-controlled life
losses, maximum pellet-free interval 59 decisions and zero sustained cycle
streak. The intentional death preceding a post-respawn episode is counted in
the full trace but excluded from its teacher-controlled suffix; those prefix
actions are not teacher labels.

There are 1,369 candidate files and 983 distinct observations: 640 teacher
roots, 380 delayed-retreat candidates and 349 wrong-turn candidates. About 28.2%
of candidate files duplicate an observation already represented elsewhere.
All ten episode-mining receipts are complete. **There are no v3 probes or
case-verification receipts yet. These are candidate states, not admitted hard
negatives or a ready training dataset.**

Native Node workers sampled at roughly 92–96% of one CPU core each. The teacher
uses JavaScript native simulation and rollout search, so this phase is CPU work;
allocating GPU memory would not accelerate that implementation.

## Fixed quotas and the first-wave bound

| Selected requests | Off-policy | Fresh v3 source | Total |
| --- | ---: | ---: | ---: |
| Training hard disagreements | 2,196 | 1,464 | 3,660 |
| Training informative windows | 732 | 488 | 1,220 |
| Development hard disagreements | 230 | 154 | 384 |
| Development informative windows | 77 | 51 | 128 |

Training additionally retains 912 earlier Pac-Man expert requests and 304
generic decision requests: **6,096 total requests / 762 updates**.

Wave 1 has eight training and four development seeds across four levels. Its
normal and post-respawn games share the same level/seed family. Therefore:

* 64 training teacher episodes and 32 development teacher episodes;
* at most 160 raw candidate files per teacher episode: 64 original roots plus
  12 roots × two perturbation types × up to four legal actions;
* at most 10,240 training and 5,120 development off-policy candidate files,
  before failed teacher gates, early stops or exact-state deduplication;
* 32 training families × the 128-request cap = **4,096 fresh selected requests**,
  below the required 4,880. This alone makes full-size wave-1 assembly impossible;
* a corresponding fresh-v3 collection has at most 2,048 initial training roots.
  Obtaining 1,464 hard disagreements solely from these roots would need a 71.5%
  unique admitted yield. Verified earlier windows can themselves become hard
  disagreements, but their yield is unknown and they do not remove the family cap.

At least ten total training seeds are mathematically necessary. That minimum
provides little room for harmless agreements, failed recovery, duplicates,
source quotas or rare behavior floors.

## Recommended next wave

Preserve wave 1. Once actual checkpoint-bound v3 probes establish yield, add an
immutable wave with **16 new training seeds and four new development seeds**.
A disjoint candidate set is `420001 + 1009*i` for `i=0..15`, and
`480003 + 1013*i` for `i=0..3`; run the existing split validation first.

This would add 160 teacher episodes and, if all seeds are also collected with
v3, 80 learner games. Across both waves there would be 96 training families,
12,288 fresh-request capacity and up to 6,144 initial learner training roots.
The 1,464 learner hard target would require a 23.8% initial-root yield, before
any additional verified hard windows. This is a planning envelope, not a
guarantee that the source/cohort floors will be met. If measured yield is
substantially lower, increase fresh families rather than filling with easy
teacher suffixes. An eight-training-seed intermediate wave would reduce work,
but still require a 35.7% initial learner hard yield across both waves.

## Targeting and performance cautions

The root sampler explicitly visits ghost danger, imminent power expiry,
pellet stalls, sparse food, junctions and respawn states. Perturbations are real
legal native actions, with complete coherent prefixes. Nevertheless, four
perturbation turns cover only a short detour, and both perturbation types often
reach the same states. Exact request/checkpoint caching in the probe callback
can avoid repeated GPU scoring while retaining every source receipt. Native
teacher verification must still check each source's coherent continuation and
reject ambiguous teacher labels.

Pending candidate metadata deliberately lacks immediate-counterfactual
information. Its exact next-action expiry and critical-risk flags must not be
interpreted as measured zeros; verification recomputes them from the engine.
After v3 probing, distinguish broad `power_expiry` exposure from an actual
v3-selected action whose duration crosses expiry near a ghost. Also report
anticipatory cases where the first action survives but later continuation fails.

The fixed one-action counterfactual uses the same teacher after either first
move. This rejects harmless route preferences, but can also rescue a bad first
v3 move immediately afterward. As a result, some multi-decision policy traps
may not become hard cases under this conservative gate. Low anticipatory yield
would justify a separately designed sequential-policy experiment, not silently
relaxing the present gate or inventing v3 negatives.

## Timing and what can be established locally

`_configuration` first verifies the existing qualification archive by replaying
its stored actions; it does not run a new teacher qualification search. The
pool's elapsed clock starts afterward, following the initial backup. A worker's
completion line appears only after both its full teacher episode and candidate
mining finish, while an `attempt.json` may already exist during mining. The
observed parent/worker lifetime difference of about ten seconds fits this
initial replay/spawn phase. Early level-1 results are compatible with real
roughly 350-decision searches on ten CPU cores; they are not evidence of skipping
the simulation.

Locally we can establish native episode outcomes, fixed-source provenance,
candidate prefix consistency, behavior exposure and teacher recovery evidence.
We cannot establish current-v3 disagreement, its confidence margin, a selected
v3 counterfactual action or final dataset readiness until the actual completed
v3 adapter is restored and queried. Use those real ordered responses; no
substitute controller or fabricated distribution should authorize a hard label.

## Parent checkpoint

V3 remains the preferable parent for a controlled correction round: the prior
paired benchmark shows its immediately avoidable fatal choices fall from 52 to
22, and safe reversals rise from 6/106 to 131/176 opportunities. It also worsens
food efficiency and first-life food while both v2 and v3 clear 0/20 games. Keep
v2 as an explicit comparator. These prior results favor retaining v3's learned
reactive safety and repairing anticipation/routing, but do not establish a v4
improvement or prove that v2 could not learn the same corrections.
