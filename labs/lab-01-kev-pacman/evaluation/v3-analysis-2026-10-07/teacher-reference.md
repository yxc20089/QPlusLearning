# Paired frozen-teacher reference — 2026-10-07

The frozen qualified teacher cleared **20/20** exact reserved starting games, with **1 life lost**, **0 avoidable immediate deaths**, and **0 watchdog exits**. These are the same level/seed starts used in the completed native-v2 versus native-v3 comparison, closing the different-seed qualification limitation.

| Metric | Native-v2 | Native-v3 | Frozen teacher |
|---|---:|---:|---:|
| Decisions, total | 7959 | 11129 | 9121 |
| Native active frames, total | 56471 | 76493 | 65865 |
| Pellets/native active frame | 0.04673 | 0.03460 | 0.07409 |
| Maximum pellet-free interval | 131 | 273 | 112 |
| Clears | 0 | 0 | 20 |
| Life losses | 60 | 60 | 1 |
| Avoidable immediate deaths | 52 | 22 | 0 |
| Mean pellets collected | 131.95 | 132.35 | 244 |
| Mean first-life pellets | 91.45 | 69.7 | 243.15 |
| Mean decisions | 397.95 | 556.45 | 456.05 |
| Mean longest pellet-free interval | 84.65 | 123.35 | 53.15 |
| Raw no-pellet cycle decisions | 0 | 16 | 0 |

| Level | Teacher clears | Life losses | Avoidable deaths | Maximum pellet-free decisions |
|---|---:|---:|---:|---:|
| 1 | 5/5 | 0 | 0 | 112 |
| 2 | 5/5 | 0 | 0 | 83 |
| 3 | 5/5 | 0 | 0 | 88 |
| 5 | 5/5 | 1 | 0 | 60 |

The teacher's maximum no-pellet interval was 112 decisions and maximum consecutive repeated-cycle streak was 0. All raw repeated cycles remain counted separately; short repeats are not silently suppressed.

The benchmark used the stored `pacman-gameplay-v2` protocol: classic pinned arcade engine, native adjacent-tile actions, levels 1/2/3/5, seeds 50021/50023/50033/50047/50051, all initial/bonus lives, maximum 10,000 decisions or 36,000 active frames and a 512-decision pellet watchdog. Both saved model traces and the teacher trace had exactly matching initial native observations in every game. Each teacher state/transition/metric was independently replayed from its fixed actions after generation.

This newly run paired reference uses the 500xx student seeds. The older qualification suite used 91009/92021/93031/94033/95047; its results are not substituted into the table. The algorithm and options were frozen from that backed-up qualification receipt. It uses native rollout MPC with 240 normal frames, 480 danger/endgame frames, buffers [1,2,4,6] and independently resampled planning scenarios [11117,77717]. It does not inspect the real episode's future RNG. There was no GPU computation or new model inference. Four CPU workers completed generation plus verification in 9.27 minutes on macOS-15.7.9-arm64-arm-64bit-Mach-O.

All source hashes were checked against the qualified receipt before work and were unchanged afterward. [The compact evidence](teacher-reference.json) records the full protocol, configuration, source/model-trace hashes, per-game metrics, runtime and private replay-bundle digest. Full teacher requests, decisions, candidates, trajectories and the replay archive remain under `/private/tmp/kev-v3-analysis-20261007/teacher-reference-private`; they are not published.

**Evaluation only:** these seeds and states remain reserved. This run generated no training inputs and authorizes no retraining or teacher changes. Complete-game success on these 20 starts establishes a reference for this fixed maze and tested levels, not universal optimality or safety. Controllers' later states differ, so conditional risk rates still compare different encountered distributions.

Pellets per native active frame is the ratio of total pellets collected to total active simulation frames, not a mean of per-game ratios. Native active frames exclude the ready/death animations consistently for all three controllers.
