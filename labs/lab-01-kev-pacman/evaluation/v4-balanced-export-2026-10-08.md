# Balanced v4 export — 8 October 2026

The local CPU collection and exact balanced export are complete. The private
Drive handoff contains both task files, the manifest, token audit and all
portable native evidence. This report records data readiness; no v4 GPU
training or learned-player improvement has been measured.

## Selected behavior categories

Each targeted root has one primary role. Overlap tags remain available and are
not added to the primary counts. Full tables, window types, source-family
counts and checks are in [the machine-readable report](v4-balanced-export-2026-10-08.json)
and [the category CSV](v4-balanced-categories-2026-10-08.csv).

| Primary targeted-root role | Train | Development |
| --- | ---: | ---: |
| Immediate collision evasion | 640 | 96 |
| Anticipatory escape | 64 | 16 |
| Actual native power-expiry evasion | 16 | 4 |
| Retreat followed by food | 512 | 48 |
| Productive routing | 1,408 | 128 |
| Sparse-pellet cleanup | 888 | 78 |
| Residual broad exposure | 132 | 14 |
| **Targeted roots** | **3,660** | **384** |
| Nearby informative decisions | 1,220 | 128 |
| Canonical native-v2 expert replay | 912 | 0 |
| **Exported task records** | **5,792** | **512** |

Kev adds 304 generic decision replay requests: 6,096 training requests and 762
optimizer updates. Training includes 762 measured food-progress windows after
a grounded threat escape; development includes 86. These exceed the 640/64
floors. Ordinary routing progress and lead-ins are counted separately.
Residual broad exposure is 132/3,660 = 3.61% of training roots, below the 15% cap.
The 16/4 expiry allocation remains a coverage limitation; four development
examples support descriptive diagnostics, not a strong generalization claim.

## Why the labels are admitted

The core contains 376 completed proofs: 145 unique admissions and 231
rejections. Its admitted pool includes 119 anticipatory escapes, 22 native
expiry hazards and four productive retreats. The separate archived-v3 catalog
admitted 11 of 12 cases: four immediate evasions and seven productive-junction
corrections. All 11 exact recorded disagreements are selected. Nearby teacher
frames do not invent new v3 predictions.

Every admitted positive continuation completed the native maze without a life
loss and passed the original loop/stall gates, independent cold replay and
source/hash binding. Anticipatory negatives preserve a forced alternative
followed by the same frozen teacher. A failed negative establishes harm under
that continuation policy, not impossibility under every future policy.
Productive routing proves measured local food advantage over a surviving
alternative, not global optimality. The frozen teacher also remains qualified
on its finite 20-game qualification suite; this is not a universal guarantee.

The export preserves rejected attempts and scout/source receipts in 4,046
portable evidence members. Validators bind the original native engine,
observation/action contract, teacher, source receipts, original legal options,
labels and exact recipe. Teacher futures and outcome labels are metadata,
excluded from model input.

## Verified data properties

- Exact model inputs are unique within each split and across training/development.
- Seed partitions are disjoint; the reserved repeated-reference benchmark is excluded.
- Non-replay data uses 46 training and 16 development level/seed families; each family has at most 128 selected records.
- All 6,304 records contain the full 36×28 maze and four ghost states. Actors and terrain remain separate.
- The exact pinned Kev renderer/Qwen tokenizer audit found a maximum of 2,407 state tokens and 2,655 packed tokens. Zero states exceed the 4,096-token training limit. CUDA training must still report its own truncation count.
- Portable semantic validation passed. The 44 focused CPU tests and generated-notebook parsing/source checks passed; they are not GPU training checks.

## Training and evaluation handoff

Use [the v4 notebook](../notebooks/pacman_kev_v4.ipynb). Its helper source is
pinned at `27bfeb72cbc7c9e677a3af6be8477ec9f874e365`. The Drive import folder is:

```text
/content/drive/MyDrive/QPlusLearning/lab-01-kev-pacman/prework-backups/v4-balanced-generation-20261008
```

The evidence archive is stored as two ordered 64-MiB-or-smaller parts and a
parts receipt. The import cell reconstructs the original ZIP and verifies
part sizes/hashes and the final manifest-declared hash before validation.
Upload success and destination names, sizes, parent and owner-only permissions
were checked. The notebook verifies downloaded bytes; no remote digest-readback
claim is made by the upload receipt.

Warm-start completed `kev-4b-pacman-native-v3`, checkpoint SHA-256
`c7c5c8326f74e86e161e05730db5004efdfd3b2351c6f2f9478581cfc13550b0`,
with a fresh optimizer and a separate balanced-v4 output. Retain rank-16
all-module LoRA, the 256-dimensional pointer head, cross-entropy, BF16 compute,
FP32 stored weights, one epoch, learning rate 2e-5, batch 4 × accumulation 2,
gradient checkpointing and the current native protocol.

Evaluate per-role held-out teacher agreement/probability alongside the same
20 complete native reference games: clears, deaths, avoidability, dry spells,
cycles and food per native frame. These reference starts informed diagnosis;
they are not a blind final test. Interactive play is ungraded and retains the
active adapter fingerprint. No rank, loss, teacher or observation change is
mixed into this data-selection experiment.

## Frozen artifact identity

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `pacman-native-v4-balanced-development.jsonl` | 4,542,310 | `5d811b2d63091e1a15307af0f9ef05dd3bc135e09c928bc65861c840a51a19b4` |
| `pacman-native-v4-balanced-train.jsonl` | 51,453,549 | `fb0ded81c1de7393364344de0bd468f9e353ba36b083ed893a7b387e1a74e834` |
| `pacman-native-v4-balanced-evidence.zip` | 133,513,786 | `f57433f484a984ef1029a2931e0074fe61cce92bb9f8da24675838e18be70423` |
| `pacman-native-v4-balanced-manifest.json` | 606,108 | `a1a151682f55e67de745165f8efdfc386e21a90e3eddfc0c002cbd9c40a38096` |
