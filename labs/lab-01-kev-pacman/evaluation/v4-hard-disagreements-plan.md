# V4 experiment: learn the decisions v3 gets wrong

Status: implementation/prework plan. No v4 dataset or trained v4 result is
claimed. Preserve completed v1/v2/v3 checkpoints. Warm-start v4 from the completed
native-v3 LoRA and pointer head, with a new optimizer/schedule and separate output.

For the requested **local teacher-only generation**, use the separate
[offline hard-scenario plan](v4-offline-hard-scenarios-plan.md). CPU teacher
labels do not require Colab or a student query. They must not be reported as
measured v3 disagreements; the checkpoint-bound protocol below remains a
separate filtering experiment.

## Why continue from v3

Use v3 as the main parent: it learned substantially better immediate retreat and
ghost evasion than v2. Its worse food routing is a behavior to correct with the
new data, not evidence that those useful safety updates should be discarded.
This is a provisional experiment choice; neither model clears a benchmark maze.
Keep the completed v2 adapter as a comparator. If we test parent choice directly,
train separate children from v2 and v3 on the exact same admitted v4 dataset,
shuffle seed and update budget, with fresh optimizers, then compare untouched
full games and fixed hard development states. Switching both the parent and
the dataset would not isolate which change helped.

Actual local collection began on October 7. Its candidates are not admitted
training rows until the real frozen v3 probes, native continuation evidence,
deduplication and quotas pass. See [the collection sizing review](v4-collection-sizing-2026-10-07.md):
the first eight-training-seed wave cannot supply all 4,880 fresh requests under
the per-family cap. Preserve that wave and add fresh families according to the
observed admitted yield. Never make up the shortfall with easy suffixes.

## What the existing games establish

On the same 20 native benchmark starts, v3 reduced avoidable immediate deaths
from 52 to 22, but cleared no boards, just like v2. It collected 34.60 pellets
per 1,000 native frames versus v2's 46.73; first-life food fell from 91.45 to
69.70, and its maximum dry interval rose from 131 to 273 decisions. All-actions-
fatal endpoints increased from 8 to 38. These endpoints can indicate an earlier
bad route; a last-moment single-action safety check cannot fix a state that has
already become unrecoverable. See [the trace audit](v3-analysis-2026-10-07/README.md).

The experiment targets anticipatory ghost escape and interception, power expiry
and safe power use, retreat followed by food progress, stalls/cycles, sparse-food
cleanup, and recovery after respawn. Current observations already include the
native ghost modes/headings/offsets and clocks, remaining power frames, the maze
with pellets/power pellets, and recent movement history. Keep this format fixed
for the first data-selection experiment; adding features and raising LoRA rank
would change multiple variables at once.

## Collection and admission

1. **Freeze the learner and teacher.** Record the completed v3 checkpoint hash,
   frozen teacher qualification/source hashes, native observation/action version,
   and seed partition. Keep the 500xx benchmark starts evaluation-only. Every
   episode and related fork stays wholly in training or development.
2. **Explore targeted native states offline.** Begin at new native games; use
   coherent teacher prefixes and legal deviations/delayed retreats to expose
   impending traps, expiry crossings, food detours and endgame cleanup. Preserve
   native clocks and history. Do not edit a JSON board into an impossible state.
   Also collect fresh v3 games, so corrections include states the learner itself
   visits. An initial hard-pool target is approximately 60% offline / 40%
   learner-derived, subject to admission evidence.
3. **Probe those states with the actual v3 adapter.** Concurrent independent HTTP
   requests use Kev's existing server batching queue. Save each state, identity,
   legal options, raw probabilities and selected action. A hazardous state v3
   already handles is not a verified disagreement. Avoid expensive full teacher
   continuations for obvious duplicate/easy candidates.
4. **Prove why the negative matters.** Exact native one-action counterfactuals
   establish an immediate avoidable death. For an earlier trap, stall or detour,
   force the teacher's first move and the learner's first move from the same
   state, then use the same frozen teacher in both continuations. Keep complete
   outcomes, lives lost, food progress, native frames, loops and dry intervals.
   A different successful safe route alone is not a harmful negative. Longer
   route labels need a meaningful survival/progress advantage, not merely the
   teacher's preferred direction.
5. **Qualify each correction.** The positive continuation must clear the native
   maze and pass the existing death/sustained-loop/stall gate. Retain rejected
   and all-fatal branches for diagnosis, but do not use their unsuccessful
   teacher actions as supervision. Independently replay admitted continuations.
   Admit short informative lead-in/recovery-to-progress windows only with their
   own state provenance and evidence; do not copy long easy teacher suffixes.
6. **Build a diverse, auditable mixture.** Deduplicate observations, cap source
   episode/fork contributions and enforce cohort/source quotas. Full trajectories
   remain evidence; each selected decision is an independent Kev training request
   with its native history. Missing hard quotas stop publication/training. Collect
   additional fresh episodes instead of filling quotas with duplicates or easy
   corridor frames.

| Source | Requests | Share |
| --- | ---: | ---: |
| Verified critical/progress-relevant v3–teacher disagreements | 3,660 | 60% |
| Informative hard lead-in/recovery-to-progress decisions | 1,220 | 20% |
| Representative previous Pac-Man replay | 912 | 15% |
| Generic decision replay, added by Kev's trainer | 304 | 5% |

The Pac-Man file contains 5,792 requests; generic replay brings the complete
stage to 6,096. The 4,880-case hard pool must contain at least 75% verified
disagreement roots. The count target is a requirement, not a promise that the
first collection wave will produce enough qualifying examples. Smaller partial
exports are diagnostic and do not authorize the standard training cell.

## Loss and controlled training

Keep all legal actions, including the verified wrong move, in the option set.
Use Kev's published option-softmax cross-entropy:

\[
\mathcal L(s)=-\log p(a^\star\mid s),\qquad
p(a\mid s)=\frac{e^{z_a}}{\sum_{b\in A(s)}e^{z_b}}.
\]

Here \(s\) is the observed game state, \(A(s)\) its legal actions,
\(a^\star\) the qualified teacher action and \(z_a\) the pointer-head score.
For a competing action \(a\), the derivative is
\(\partial\mathcal L/\partial z_a=p(a\mid s)-\mathbf1[a=a^\star]\).
A confident wrong move therefore receives a strong corrective gradient.
Disagreement mining selects useful comparisons; it does not by itself reproduce
[CLM's training implementation](https://github.com/Contrastive-LM/CLM).
An explicit pairwise/ranking loss is a later controlled ablation if this data
change does not fix the measured behaviors. Learner-state relabeling is related
to [DAgger](https://arxiv.org/abs/1011.0686), without claiming its original schedule
or guarantees for this one-round mixture.

Keep rank 16, all-module LoRA, the same pointer head, one epoch, lr 2e-5,
batch 4 × accumulation 2, BF16 autocast, FP32 stored weights, checkpointing and
required optimized kernels. The complete mixture yields 762 optimizer updates.
Do not resume v3's completed optimizer: resume only interrupted v4 recovery.
Measure retention on existing generic held-out tasks because reduced generic
replay can trade off earlier capabilities.

## Evaluation and stop conditions

Before/after probes use the **same frozen, disjoint hard development decisions**.
Report cohort counts, teacher agreement, teacher probability, CE and margin;
these diagnose whether the learner actually learned the intended corrections.
Audit known prior critical training cases separately to distinguish failure to
fit from failure to generalize. Probe accuracy is not the promotion criterion.

The primary benchmark stays the same **20 complete native games**, same starts,
physics, legal directions, action boundaries, observations and watchdogs. V3 and
v4 each play without a teacher/safety override. Save every state/action/probability
trace. Report clears, censoring, lives lost, avoidable immediate deaths with their
opportunity denominator, all-actions-fatal endpoints, first-life food, pellets
per native frame, dry intervals and sustained loops. Break out ghost proximity,
power expiry, stall, respawn and sparse-food cohorts, while acknowledging that
each controller visits different states. Compare to the already recorded frozen
teacher reference as context; do not relabel benchmark traces for training.

Tune data allocation on fresh development games. Require improved safety and
food efficiency without increased censoring before promoting v4. If hard
development fit is poor, inspect conflicting labels/observation equivalence and
optimization before changing rank. If development fit is good but full play
still fails, collect newly visited v4 hard states in a subsequent iteration.

Generation has per-case atomic receipts and resumable stages. With Drive enabled,
back up committed cases and changed indexes at start, every five minutes and
orderly interruption/completion. Restore an absent local collection automatically;
retry incomplete cases. Keep checkpoint recovery separate from collection
recovery, and back up benchmark/interactive traces in the final session archive.
