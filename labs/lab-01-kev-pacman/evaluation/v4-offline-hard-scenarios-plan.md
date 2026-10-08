# V4: generate hard Pac-Man scenarios locally

Generate the new data on CPU. The native game, ghost rules, qualified teacher
and observation/action protocol remain pinned. Colab is needed for later Kev
training, not for teacher-labelled scenario generation. Querying the completed
v3 adapter is a separate experiment when we want measured learner/teacher
disagreements.

The local collection started on October 7 with 96 fresh native teacher games:
four levels, eight training seeds, four development seeds, and normal or legal
post-respawn starts. All 96 cleared; five teacher-controlled lives were lost.
The offline exporter uses the stronger **zero teacher-controlled life loss**
gate, so those five source games supply no training labels. Intentional
post-respawn setup actions remain evidence and are never teacher labels.
These initial figures describe generation outcomes. A second wave added four
training seeds and two development seeds, or 48 fresh games. All 144 source jobs
completed. The exported snapshot independently replayed 126 completed source
games: 114 qualified with zero teacher-controlled life losses and 12 were
rejected. The 18 subsequently completed source games remain in the raw backup;
they are not needed to fill this export. See the [completed local report](v4-offline-generation-2026-10-07.md).

## Which states supply labels

Only multi-option states on independently replayed, winning, zero-loss teacher
trajectories qualify. The targeted pool visits approaching dangerous ghosts,
mixed safe/fatal immediate alternatives, imminent power expiry, pellet-free
intervals of at least 24 decisions, and endgames with at most 30 pellets.
Respawn by itself does not make an ordinary state hard. Keep every legal
direction in the input, including unsafe alternatives; label the safe teacher
move. Exact model-input duplicates and conflicting teacher labels are excluded.
Source seed families stay disjoint between training and development.

The teacher's complete native source suffix provides positive survival and
pellet-clearance evidence. Its actions and the counterfactual immediate risks
are checked in an independent native process. A different safe route is not
declared harmful or globally suboptimal. This dataset does not claim that v3
chose any of its alternatives: no student predictions are fabricated.

Power clocks require actual transition measurements. Eating a ghost introduces
pauses during which the power timer can freeze. `remaining_frames <= action_frames`
therefore describes potential timing exposure, not confirmed expiry. The new
export records before/after power clocks and reports actual expiry during the
chosen teacher action separately. Teacher plans, future outcomes and risk
labels remain outside Kev's input.

## Fixed data and training budget

| Partition | Targeted scenarios | Nearby informative decisions | Prior Pac-Man replay |
| --- | ---: | ---: | ---: |
| Training | 3,660 | 1,220 | 912 |
| Development | 384 | 128 | 0 |

Informative frames must be bounded lead-ins or recovery follow-through in the
same teacher-controlled life as a selected targeted state. They are not an
entire easy trajectory. Targeted states provide 75% of fresh selected data.
Deduplicate inputs, retain at most 128 fresh requests per level/seed family,
and require behavior coverage floors from targeted states themselves. Missing
quotas stop admission; they cannot be filled with easy frames or duplicated
examples.

The Pac-Man training file has **5,792 requests**. Kev adds **304 generic replay
requests**, giving **6,096 total / 762 optimizer updates**. Keep rank 16,
all-module LoRA plus the pointer head, option cross-entropy, one epoch,
learning rate 2e-5, and batch 4 × accumulation 2. This isolates the new data
selection. GPU training has not started.

Use **native-v3 as the parent**, with a fresh optimizer and separate output.
V3 learned better immediate retreat than v2, but regressed in food efficiency;
the new data targets the remaining behavior. Keep v2 as a comparator. This is
an experiment choice, not evidence that a v4 player has improved.

## Keep the two evidence modes distinct

The export is named `pacman-native-v4-offline` and contains source trajectories,
independent replay receipts, teacher labels and native immediate risks. It
does not satisfy the [actual-v3 disagreement protocol](v4-hard-disagreements-plan.md),
which additionally needs checkpoint-bound predictions and complete
teacher-first versus learner-first counterfactual continuations. The latter
remains available as an optional filtering/diagnostic experiment. Both use the
same unchanged full-game benchmark; snapshot teacher agreement alone cannot
establish successful Pac-Man play.

Portable training/development files, a manifest and a full evidence archive
passed local admission and validation. All 6,304 selected training/development
requests fit the 4,096-state-token allowance with the pinned renderer/tokenizer;
the maximum state is 2,403 tokens. Local data stay outside the source repository;
a private Drive backup preserves completed work. This does not establish GPU
training behavior or a successful v4 player.

## Reproduce the local preparation

From the lab directory, generate the first wave with Node and Python on CPU:

```sh
python3 v4_data.py harvest /absolute/path/to/collection --workers 10
```

Append the same second wave without changing the frozen generator:

```python
from pathlib import Path
from v4_data import check_seed_splits, harvest_scenarios
collection = Path('/absolute/path/to/collection')
train = [420001 + 1009 * i for i in range(4)]
development = [480003 + 1013 * i for i in range(2)]
check_seed_splits(train, development, collection)
harvest_scenarios(collection, Path('evaluation/teacher-qualification.json'),
                  train_seeds=train, development_seeds=development,
                  workers=10, wave_id='wave-002', roots_per_episode=64,
                  perturb_roots_per_episode=12, perturb_turns=4)
```

Verify completed games and export the full fixed-budget dataset. The verifier
reuses matching complete receipts and retries incomplete work. The builder
stops on missing coverage or family capacity:

```sh
python3 teacher_offline_data.py verify --collection /absolute/path/to/collection --evidence /absolute/path/to/evidence --qualification evaluation/teacher-qualification.json --workers 4
python3 teacher_offline_data.py build --evidence /absolute/path/to/evidence --out /absolute/path/to/dataset --qualification evaluation/teacher-qualification.json
python3 scenario_catalog.py --dataset /absolute/path/to/dataset
```

The dataset uses its own `teacher_offline_data.py validate` command. Do not feed
these files to the disagreement notebook's `v4_data.py` validator or represent
them as checkpoint-bound corrections. Later GPU training can use the existing
`runtime.finetune` API with this export's recipe after validation, native-v3 as
`init_from`, and a new `kev-4b-pacman-native-v4-offline` output. That training has
not been run.
