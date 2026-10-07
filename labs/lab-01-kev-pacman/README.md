# Lab 1 — Train a Decision Model to Play Pac-Man

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yxc20089/QPlusLearning/blob/main/labs/lab-01-kev-pacman/notebooks/pacman_kev_lab.ipynb)

Train **Kev-4B** with its published LoRA plus pointer-head architecture, then teach it to control Pac-Man. Chase/scatter ghost movement follows deterministic personality rules; frightened turns use the browser engine's RNG. The class lasts **90 minutes with 30 minutes for Pac-Man fine-tuning**; installation, downloads, demonstration generation and four general training stages are prework. Time the full Pac-Man stage before class; a slower run also belongs in prework.

The lab has **one assessed checkpoint: CP1 — Fine-tune and evaluate Kev on Pac-Man game states**. Review the planning labels, train the task adapter, compare it with the Skills baseline, and submit the evidence. **Interactive play** is a separate, ungraded activity available before and after fine-tuning; its badge identifies the running LoRA and pointer head.

Use [the notebook](notebooks/pacman_kev_lab.ipynb), [Colab setup](COLAB_SETUP.md) and the **[15-slide lab deck](https://docs.google.com/presentation/d/1_PfmbUEH38jMO_24S_AqtUUh-IIg2LyZYkEYKjZ5uXQ/edit)**. Local exports are [PDF](slides/pacman-lab.pdf) and [PPTX](slides/pacman-lab.pptx).

| Slides | What learners do |
| --- | --- |
| 1–5 | Compare Kev/CLM architectures, watch the embedded CLM walkthrough, and discuss cost/performance/serving |
| 6 | Watch scoring and softmax; change logits with the interactive controls |
| 7–9 | Follow the 90-minute route, prepare Colab and identify the five separate training stages |
| 10–11 | Try ungraded native Pac-Man play and review the ghost-aware planning labels |
| 12–15 | Complete CP1: train, monitor/recover, evaluate and submit measured evidence |

The deck embeds looping GIFs on **slides 2, 6 and 13**, with typeset pointer/softmax, planning-objective, cross-entropy, LoRA and evaluation equations. Each equation has a variable key, and speaker notes provide a spoken teaching script. Open the public [CLM pause/scrub controls](https://yxc20089.github.io/QPlusLearning/animations/clm.html) or [Kev scoring and LoRA controls](https://yxc20089.github.io/QPlusLearning/animations/kev-math.html) from the slides. No download, account or GPU is needed for those controls. Animation values are illustrative, not trained-model results. PDF exports are static; Google Slides and PPTX retain the GIFs.

The [companion CLM lecture](https://docs.google.com/presentation/d/1sdnPkV6VyTW9tr6Xmtyyxns4UHlUGoTtTLj-vmxNhfo/edit) supplies the original 65-second CLM walkthrough and further softmax, contrastive-head, LoRA and evaluation explanations. Its CLM animation shows frozen independent encoders and two projection heads; Kev uses joint encoding, LoRA and a pointer head. The lecture's joint language/decision loss is a proposal, separate from this lab's Kev decision loss.

The target is **one NVIDIA RTX PRO 6000 Blackwell GPU in Colab**. The full Server Edition has 96 GB VRAM. Arrange access before class, inspect the actual allocation, and record elapsed time and compute units. Colab does not guarantee this GPU or free access.

## The five training stages

Start from `Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b`. Only Stage 1 initializes fresh adapters/head; each later stage uses the preceding learner checkpoint. Original base matrices remain frozen while LoRA changes effective encoder features. This is decision-model training over an already pretrained LLM, rather than language-model pretraining from random weights. Old 0.8B adapters cannot initialize 4B.

| Stage | New source records | Replay from decision-v7 train | Published reference settings | Optimizer steps |
| --- | --- | --- | --- | --- |
| 1. Initial decisions | 12,576 decision-v7 | None | 2 epochs, lr 5e-5, batch 4 × accumulation 2, seed 2 | 3,144 |
| 2. Dates/missing evidence | 1,425 generated cases | 2,000 | 1 epoch, lr 2e-5, batch 4 × accumulation 2, seed 1 | 429 |
| 3. Documents | 5,219 finance complaints | 2,000 | 1 epoch, lr 2e-5, batch 2 × accumulation 4, seed 2 | 903 |
| 4. Skills/devtools | 6,000 hard-v1 + 5,320 devtools-v1 | 4,000 | 1 epoch, lr 2e-5, batch 2 × accumulation 4, seed 1 | 1,915 |
| 5. Pac-Man | 4,096 planning-labelled boards | 2,000 | 1 epoch, lr 2e-5, batch 4 × accumulation 2, seed 7 | 762 |

All stages use BF16 autocast over FP32 stored parameters, gradient checkpointing, rank-16 LoRA (alpha 32) on all projections and a 256-dimensional pointer head. Upstream provides AdamW, OneCycleLR and cross-entropy over supplied options. General stages retain option shuffling, none/distractor insertion and 25% none minimal pairs. Documents and skills use `max_state=7552`; dates uses 384. Pac-Man uses 4,096 and disables none/distractor and none-pair augmentation because only legal directions are valid. Option shuffling remains active; replay retains its recorded generic labels. Pac-Man is a new task recipe following the intermediate-stage architecture, learning rate, effective batch, one-epoch pattern and replay, rather than a published upstream Pac-Man experiment.

The initial recipe selects arm 0 of `experiments/q35-4b-s23.json`. [training-stages.json](training-stages.json) pins intermediate settings and data hashes. Documents remain separate from skills, as in the 4B release. Skills data concatenates verified hard-v1 train then devtools-v1 train; its combined checksum is course-specific. We do not assert byte identity to the unavailable historical round10 joint file. No development/test rows enter training. Calibration remains separate and is not fitted or inherited from the release; neither are its benchmark scores.

The notebook's **Stage 1 training cell explicitly uses batch 4 × accumulation 2, non-reentrant gradient checkpointing and `row_budget=0`**. It assigns `runtime.initial_execution` before constructing and printing the actual command. The default `memory_safe` profile supplies **batch 1 × accumulation 8 and `row_budget=2048`** to Dates, Documents and Skills. **Pac-Man explicitly selects 4 × 2 / row budget 0** in `finetuning_command`, after generic profile application; its printed command is authoritative. These executions retain effective batch eight. Microbatch weighting, grouping, dropout execution and optimized arithmetic differ, so changing execution during a run is not an exact numerical reproduction. `TRAINING_PROFILE='published_reference'` retains original intermediate-stage execution flags and still requires optimized kernels. A nonzero row budget splits questions into passes; a longer single question runs intact without silently truncating or dropping records.

Pac-Man now performs 762 updates instead of the former 16, comparable in scale to Dates (429) and Documents (903). Compute time is roughly `762 × steady seconds per update`, plus loading, compilation, saves and Drive copying: 1.5 seconds/update gives 19 minutes of compute; 3 seconds gives 38 minutes. The 20-minute classroom training target requires an instructor measurement on the actual allocation. Full runs have a 180-minute attempt cap, preserve recovery, and are never silently truncated to fit a timetable. More data does not guarantee a stronger learned player.

The published initial execution is **batch 4 × accumulation 2, checkpointing enabled, BF16 autocast, FP32 stored weights and no row budget**. The [model card](https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/docs/model-cards/kev-4b.md#compute) reports about 56 minutes / 24.6 GB peak on one H100. These are not RTX PRO 6000 measurements.

For a running initial stage, wait for a recovery save and interrupt the cell. Keep the original checkpoint path. The Stage 1 cell has one resume setting: `RESUME_CHECKPOINT` defaults to the latest local or Drive-restored snapshot when one exists, otherwise a fresh run. Explicitly set `'latest'` to require recovery, a full snapshot directory path to choose a save, or `None` for a fresh output. Completed checkpoints are reused. It prints the selected path and saved optimizer step, then resumes at 4×2 / row budget 0. A missing or incomplete requested checkpoint stops training without starting over. Explicitly set `RESUME_CHECKPOINT=None` only for a student's first run in a new output directory. The guarded continuation allows only batch/accumulation/row-budget changes at a plain single-GPU optimizer boundary, preserves effective batch and total steps, and maps progress to the same next source records. Weights, optimizer, scheduler and RNG are restored. Grouping changes the numerical trajectory; recovery receipts and final evidence retain the execution history. Other stages' ordinary resume still rejects changed arguments.

On a fresh runtime without a restored checkpoint, complete **Stage 1** before running dates. Setup downloads the pretrained backbone; Stage 1 creates the initial trained LoRA/head checkpoint that dates must load. An older saved notebook may still explicitly request `'latest'`: change `RESUME_CHECKPOINT` to `None` **inside the Stage 1 cell** for a fresh run. Checkpoint handoff errors now name the missing stage or show the native reader's stderr instead of hiding it behind `CalledProcessError`.

To continue from a **completed dates checkpoint**, copy its native files directly into `checkpoints/kev-4b-dates` (or set `DATES` inside the documents cell to its restored folder). Run runtime setup, optimized preparation and storage, then start **Stage 3 documents**; the Stage 1/2 training cells and dates recovery files are unnecessary. Later cells import their own recovery helper and read saved configurations/metrics from disk. Missing older stage archives are identified in the comparison and submission rather than treated as available.

Intermediate training automatically prepares missing curriculum data and verifies its pinned checksum and record count **before loading the GPU model**. You can also run `runtime.prepare_intermediate_data()` ahead of time. An empty output folder left by a failure before data loading is preserved under a unique `-empty-attempt-*` name so the same stage can retry. Nonempty outputs and recovery directories remain protected.

Stages 1–4 each have a separate run cell, checkpoint, backup ZIP, restore control and TensorBoard directory. Each has a configurable 180-minute attempt cap, a scheduling limit rather than a prediction. Keep all four checkpoints before class. Stage 4 is the class baseline. The notebook checks full optimizer-step counts, recipe/data pins, parent fingerprints and learner/instructor ownership. Kev's records-seen count includes augmented siblings and may exceed requested source records.

| Minutes | Activity | Evidence |
| --- | --- | --- |
| Prework | Install; verify kernels; initial → dates → documents → skills | Four checkpoints and separate training curves |
| 0–20 | Compare architectures, inspect the Skills baseline and try interactive play | Configurations, parameter audit, active adapter and legal actions |
| 20–30 | Review player labels | Edited train partition; evaluation closed |
| 30–60 | Inspect loss (5 min), fine-tune (20 min), inspect/save (5 min) | Pac-Man checkpoint and positive optimizer steps |
| 60–80 | Compare decisions, gameplay and cost | Paired predictions, captures, dots, latency and memory |
| 80–90 | Explain and submit | Notebook, labels, five adapters/heads and measurements |

## Required optimized training

[optimized_training.py](optimized_training.py) adds hash-verified wheels from [training-kernels.json](training-kernels.json) to Kev's frozen environment using `--no-deps`: flash-linear-attention/fla-core 0.5.2, causal-conv1d 1.7.0, einops 0.8.1 and Ninja 1.13.0. The CUDA convolution wheel requires Python 3.13, Linux x86_64, Torch 2.8 and CXX11 ABI. Kev's Torch 2.8.0/CUDA 12.8, Triton 3.4.0, Transformers 5.17.0 and PEFT stay locked.

FLA's Triton kernels handle Gated DeltaNet; the CUDA convolution handles its short convolution; PyTorch SDPA handles full attention. Training uses fused AdamW, BF16 autocast, checkpointing and accumulation; the later-stage memory profile also bounds row passes. Kev's inference fusion/CUDA graphs are disabled for this backward path. The trainer source remains pinned and checksum-verified; telemetry, fused optimizer and recovery hooks are inserted in memory.

`prepare_training()` downloads/audits the actual 4B model on CPU, then checks pinned optimized function bindings and CUDA/BF16 availability. The separate two-record training preflight is **skipped by default** (`RUN_TRAINING_PREFLIGHT=False`). Its JSON report records `bindings_verified`, zero optimizer steps and `cuda_loss_backward='not_run'`. Each real training subprocess rechecks bindings and refuses reference-kernel fallback. The first real optimizer step exercises training and saves a recovery checkpoint.

Use `prepare_training(run_training_preflight=True)` for the optional two-record loss/backward/fused AdamW check. It verifies finite adapter/head gradients and actual FLA/convolution/fused AdamW dispatch counters without a CUDA profiler. Phases identify encoding, forward, backward, gradient checks and completed updates. This optional check and real training print Python stacks after 60 seconds without progress. Heartbeats show process liveness. The binding-only attempt has a two-minute timeout and the optional training check has a 20-minute timeout; neither is a duration estimate. Skipping the extra check does not bypass first-use kernel compilation during training. **The optional CUDA check and full 4B curriculum have not been run by the course author.**

For an old preflight that is still running, interrupt its cell and keep the runtime connected. Copy and run the updated bootstrap cell to refresh helpers while keeping downloaded weights/kernels, checkpoints and logs. If the optimized environment is already installed, run the new separate preparation cell with `RUN_TRAINING_PREFLIGHT=False`, then proceed to Stage 1.

## Progress and TensorBoard

Colab supports the embedded TensorBoard cell; open it before training. Each optimizer step records CE, next-step learning rate, gradient norm, epoch, duration, records seen and peak/live allocated/reserved/free GPU memory. These are training curves, with no invented validation curve. Logs remain separate under `logs/<stage>/<attempt>`.

The monitor streams the first step, every ten steps and the final step, plus 15-second heartbeats during loading/compilation or long steps. Raw stdout/stderr, JSONL, CSV, TensorBoard events and failure/timeout status are preserved. `batches.jsonl` records physical rows, longest row, padded tokens and record IDs before each forward pass, including the failing pass. [training_monitor.py](training_monitor.py) verifies all insertion sites against the pinned upstream trainer.

The former `BUNDLED_FILES` embedded helpers and licensed game source. The notebook now downloads committed, hash-verified helpers, the native asset archive and a compact planning dataset, then reloads cached helper modules. [notebook-source.json](notebook-source.json) records that pin. CP1 verifies the teacher qualification receipt and native replays before extracting the two declared training/development JSONL files; reviewed labels and checkpoints are preserved. Model weights and kernels download separately. The notebook kernel hosts teaching helpers/TensorBoard; Kev runs in its own locked Python environment.

## Recovery and the earlier OOM

The learner's old 0.8B batch-8 run failed at step 2,073/3,144 on a 94.97 GiB GPU: 89.90 GiB was actively allocated, with only 73.88 MiB free. Logs confirmed both optimized DeltaNet/convolution packages were missing. The cumulative peak reached 91.91 GiB by step 280 and stayed there; that alone does not establish a leak. Optimized kernels and a smaller execution profile address the observed setup, while new telemetry measures batch shapes and live memory. Allocator `expandable_segments` cannot free active tensors.

**The old notebook saved LoRA weights only after completion. That failed run has logs/configuration but no automatic learned checkpoint.** We cannot reconstruct exited-process weights from those logs. The new `kev-4b-*` output paths preserve old files and start fresh, incompatible 4B adapters.

[lora_recovery.py](lora_recovery.py) saves after optimizer step 1, every 100 steps or five minutes (checked at optimizer boundaries), and the final step. It keeps the latest two complete snapshots under `<output>-recovery`. Each contains loadable adapter/head exports plus optimizer, scheduler, CPU/CUDA/Python RNG and progress counters. Atomic completion markers prevent selecting incomplete saves. Native Kev resume is full-weight-only; the lab implements LoRA recovery separately. CPU tests verify identical continuation including dropout, Adam moments, scheduler and actual pinned epoch-loop behavior. CUDA continuation remains unvalidated.

After interruption, ordinary resume requires identical profile, arguments, inputs, parent checkpoint and output paths. Stage 1 selects recovery with `RESUME_CHECKPOINT` and enables its explicit 4×2 execution-change path as described above. Later-stage resume flags automatically detect restored snapshots, and completed outputs are reused. Saved Pac-Man labels are preserved unless explicitly edited. Completed stages stay completed. A partial snapshot is not a completed curriculum stage. Set `SAVE_TO_DRIVE=True` in the storage cell before training. The notebook mounts Drive, keeps training on the local disk, and automatically archives every completed recovery save plus the final checkpoint under `MyDrive/QPlusLearning/lab-01-kev-pacman/backups/<stage>`. [checkpoint_backup.py](checkpoint_backup.py) verifies the copied archive before publishing its pointer and keeps the latest two backups per stage. Archives contain adapter/head weights, optimizer, scheduler, RNG, counters and any custom training JSONL. A failed copy leaves the previous backup and local snapshot intact and stops training. Wait for `Drive backup complete at optimizer step …` before replacing the runtime.

On a new runtime, enable the same flag and run the storage cell: absent local outputs and saved input labels restore automatically to the original paths. Existing local checkpoints and changed labels are preserved. Stage 1 resumes the restored save automatically; with no backup or local checkpoint it starts fresh. No manual file copying is needed. Drive copying/mounting remains untested on a live Colab runtime; CPU tests exercise archive verification and restored optimizer/dropout continuation after deleting the local workspace. Logs and TensorBoard events use the separate export control. Recovery archives must restore at their original absolute paths; completed checkpoint ZIPs have per-stage restore controls.

To start from **completed Skills**, upload its entire automatic-backup folder unchanged to `MyDrive/QPlusLearning/lab-01-kev-pacman/backups/kev-4b-skills` if it is not already there. Run bootstrap, runtime setup, optimized preparation and storage with `SAVE_TO_DRIVE=True`. Skip the Stage 1–4 training/export cells and run **Load completed Skills checkpoint — start the lab here**. You can then try Interactive play or begin CP1. This startup cell needs only the restored Skills checkpoint and initialized runtime; it does not use recovery helpers or require a Documents checkpoint. It loads available stage metrics, reports absent earlier archives and retains checkpoint ownership. CP1 fine-tunes this baseline on Pac-Man.

## Game, labels and evaluation

The selected faithful browser recreation is [masonicGIT/pacman](https://github.com/masonicGIT/pacman), pinned `7407174c1d6a38be8cd230577489e39e0873145b`. Classic mode keeps its original maze, renderer, font, sounds, four ghost personalities, chase/scatter timers, frightened mode, power pellets, tunnels, fruit, scoring, lives and levels. The author documents small [accuracy differences](https://github.com/masonicGIT/pacman#accuracy). Original source/assets and notices are unchanged in [source.zip](vendor/arcade-pacman/source.zip); [source.json](vendor/arcade-pacman/source.json) verifies each file.

[pacman_lab.py](pacman_lab.py) supplies structured decisions, labels and evaluation. [arcade-engine.js](games/arcade-engine.js) exposes the native state and player controls; it adds no ghost AI or replacement physics. [arcade-browser.js](games/arcade-browser.js) relays decisions through a Colab callback to the notebook-local server. Python evaluations execute the same engine with Node.js. Colab installs a checksum-pinned official Node 22.17.0 Linux binary only if Node 18+ is absent, separate from Torch. Kaggle can use structured decisions and Python rollouts.

The board displays **Active LoRA + pointer head**: verified checkpoint name, stage, optimizer steps, rank, path and adapter/checkpoint SHA-256. **Interactive play — general Skills baseline** explicitly selects `kev-4b-skills`, which has no Pac-Man training. The separate **Interactive play — Pac-Man fine-tuned Kev** cell immediately after Stage 5 explicitly loads the selected task checkpoint (new runs use `kev-4b-pacman-native-v2`; select your existing `kev-4b-pacman-planner-v1` explicitly to inspect it) on the same classic game. It works before benchmarking and after restoring a completed task checkpoint from Drive, without requiring earlier training/evaluation variables. Select **Kev**, press **Start**, and verify the badge says **Pac-Man fine-tuned**. **New game** resets both boards with seed 7. Pause the other board before switching models or benchmarking; both connect to the same inference server. Fine-tuned play records `results/fine-tuned-player-trace.jsonl`. Each trace records the adapter identity. Unexpected serving checkpoints and adapter changes during a request stop play; no rules controller silently replaces model failures.

The v2 benchmark runs **20 complete native games per adapter**, from five reserved seeds at starting levels 1, 2, 3 and 5. Every episode starts in a fresh native closure and continues through all lives until the first maze clears or the native game-over. Large watchdogs detect a hung controller; those exits are incomplete failures. There is no isolated-state accuracy score in the mandatory comparison. Report clears, remaining pellets, immediate avoidable deaths, dry cycles/stalls, power/ghost/fruit events, post-respawn progress and action geometry. Read per-seed paired changes together; higher score alone is insufficient. See [benchmark-spec.json](benchmark-spec.json) and [gameplay_benchmark.py](gameplay_benchmark.py).

New native v2 decisions end on **entry to the requested adjacent tile** while executing complete native frames. This fixes centre checks that could skip tiles or end a reversal in the same tile. The browser and headless benchmark share the boundary; native ghost AI, speeds, collisions and scoring remain upstream. Legacy v1 action/state semantics remain available only for old-data/trace replay. Old scores are not directly comparable to v2 scores.

**Power and movement information reaches Kev.** The v2 maze marks remaining power pellets as `o`; after consumption the tile becomes empty. Global/per-ghost frightened flags, ghost modes and remaining power frames identify whether ghosts can be eaten. State also includes actor pixel offsets and speed phases, queued ghost headings/reversals, release counters, the phase clock, fruit countdown, 64 recent positions, destination visits and decisions since the last pellet. Teachers and students use the same current native facts; future randomness is resampled independently rather than read from the episode RNG.

[Teacher methods](evaluation/teacher-methods.md) compares CS188 food search, minimax/expectimax, native rollout MPC, published real-time Ms. Pac-Man MCTS, symbolic safety advice and approximate Q-learning. The implemented candidate is **native rollout MPC with cached shortest-food routes**, not UCT or a competition-agent port. It evaluates legal first moves under four routing/safety policies (buffers 1, 2, 4, 6) and two future-RNG scenarios for 240 native frames, extended to 480 near dangerous ghosts or in the last 30 pellets. Exact immediate safety and actual/imminent dry-cycle avoidance precede approximate longer-term survival and predicted time to the next pellet. A finite horizon cannot establish global optimality.

[teacher_validation.py](teacher_validation.py) tests complete games with [predeclared gates](teacher-validation-spec.json): every board must clear, no avoidable immediate death is allowed, a sustained cycle (eight consecutive cycle detections) or a pellet stall beyond 128 decisions fails, and movement endpoints must be adjacent. Every brief repeat remains visible in the raw cycle count. Development, current/retired qualification and student benchmark seeds are separate. Source/configuration hashes freeze the measured candidate, and compact replay evidence recomputes native observations, transitions and gate metrics before a receipt can unlock training data. Failed or changed teachers fail closed.

The [archived strict qualification](evaluation/teacher-strict-qualification-v2.json) cleared 20/20 boards with zero deaths but failed its zero-repeat rule on one brief repeat. The user selected sustained-loop/stall rejection instead. The algorithm stays unchanged, and v3 uses 20 fresh qualification cases rather than reclassifying that observed failure. The original code/reports/replays are reproducible at commit `c29f2ad`; the current qualification specification records the retired seeds and the changed criterion.

The [v3 qualification receipt](evaluation/teacher-qualification.json) passes: **20/20 native maze clears, zero deaths, zero raw cycles**, and a longest pellet stall of **67 decisions**. Every pellet is collected. The [qualification replays](evaluation/teacher-qualification-replays.zip) are independently verified after loading the saved JSON report. The frozen teacher and its search options match the development candidate; qualification uses separate, previously unseen seeds. The finite suite does not prove universal optimality or safety.

On the 20 development games, the [fixed MPC candidate](evaluation/teacher-development.json) clears **20/20**, loses **zero lives**, and has **zero dry-cycle decisions**; its longest pellet stall is 92 decisions. It collects every pellet and eats 50 ghosts. The [simple route heuristic](evaluation/heuristic-development.json) clears **16/20**, loses 32 lives and records 77 dry-cycle decisions. These are CPU algorithm results, not learned-Kev results. Development success alone cannot authorize labels; the separate qualification receipt controls that gate.

[teacher_data.py](teacher_data.py) can generate `pacman-native-v2` only after qualification. Collection runs complete native games from new initial-board seeds, interleaving ordinary starts and post-respawn recovery. In recovery cases, a declared legal-action prefix deliberately loses one life; the frozen teacher then takes over and must finish the maze under the same per-game gates. The scripted prefix is never labelled. Full-game metrics retain its death, while separate teacher-control metrics measure the continuation. Native replay verifies both phases before publishing any label partitions. It never discards failed teacher games to manufacture a successful dataset.

Sampling takes every fourth teacher move plus danger, power and late-maze decisions, shuffles within completed episodes and interleaves levels/modes. The target is 4,096 training and 256 development labels, with episode/input-disjoint splits; generic replay remains 2,000 requests, so the full task recipe remains 762 updates. These recovery examples cover actual native post-death release/timer states. They are scripted perturbations, not learner-visited DAgger data. The mandatory benchmark uses actual games rather than evaluation snapshots. Search plans and labels stay in `_meta`/training targets, outside inference input. Manual learner annotations retain their separate provenance.

The released [v2 dataset manifest](data/pacman-native-v2-manifest.json) records **20/20 collection clears**, including ten recovery games. Teacher-controlled continuations lose three lives, with zero avoidable immediate deaths, four raw cycle detections (longest streak two), and a maximum pellet stall of 100 decisions. The ten intentional prefix deaths are retained separately in full-game metrics. All 20 native collection replays and all 4,352 selected input/label pairs verify. The exact [generating source](data/pacman-native-v2-generator.py) is archived by its SHA; the current reader independently recomputes coverage, correcting a collection-mode/post-respawn counter-name collision without changing labels or trajectories.

| Training coverage | Examples out of 4,096 |
| --- | ---: |
| Post-respawn state | 2,200 |
| Dangerous ghost within three maze tiles | 1,044 |
| Immediate safe/fatal action alternatives | 208 |
| Last 30 pellets | 1,558 |
| Frightened mode | 1,858 |
| Power expires within 60 native frames | 444 |
| Actual power-pellet consumption action | 48 |
| Actual ghost-eating action | 23 |
| All four ghosts outside | 3,290 |

These rows overlap. Two training states have no immediately surviving legal action; their targets reflect the approximate teacher's remaining choice, not a claim that every labelled action can avoid death. Exact ranking ties are recorded for review. [Token audit](data/pacman-native-v2-token-budget.json): all 4,352 requests fit the 4,096-state-token allowance; the largest flattened state is 2,414 tokens and the largest estimated packed request is 2,658. [audit_token_budget.py](audit_token_budget.py) reproduces this CPU check with the pinned renderer/tokenizer. Actual CUDA training must separately confirm no truncation or dropped records.

The old `pacman-planner-v1` dataset and checkpoint are archived. Their short-horizon teacher cleared no board in the original first-life report and has inadequate late-maze/respawn coverage. Do not use those results to qualify a new teacher. Your existing adapter remains playable and can be explicitly selected for the new full-game comparison; it was trained with the older schema. New data requires a fresh task output, `kev-4b-pacman-native-v2`; Skills and Stages 1–4 require no retraining.

Human mode runs at 60 simulation frames per second. Kev pauses the simulation while choosing each move. The display shows simulation seconds and every ghost's mode. Ghosts leave home gradually: Blinky starts outside; Pinky leaves first; Inky and Clyde use pellet counters or a no-pellet timeout (about four simulation seconds at level 1). Repeated reversals can advance very few frames per decision, making release appear slow in wall-clock time. The browser retains three lives and level progression. Truncation or dropped records fail the training audit. Versioned v1 files preserve previous experiments; new v2 training never overwrites them. Reuse your completed Skills checkpoint and run CP1; general stages need no retraining.

**Refresh an existing runtime:** copy and run the new bootstrap cell, then rerun `runtime.setup()`, optimized preparation with `RUN_TRAINING_PREFLIGHT=False`, and the storage cell with `SAVE_TO_DRIVE=True` if using Drive. Bootstrap creates a new runtime helper, so these cells restore its hardware, training and backup settings while reusing cached files. Run **Load completed Skills checkpoint — start the lab here**, then CP1. Interactive play is a separate activity. Checkpoints/logs remain in place; general stages do not need retraining.

## Trajectory diagnosis and next data plan

The [October 5 audit](evaluation/trajectory-audit-2026-10-05.json) reproduces every one of **251 supplied observations** with the pinned native engine and recorded choices. The adapter identity is consistent with the completed, 762-step `kev-4b-pacman-planner-v1` checkpoint: adapter SHA-256 `c10d45dce2e14d8cc2de7c1344ae20b9a0d6d74f309f2c0007f24b882b77e92e`. The game ends at **760 points, 72/244 pellets, three deaths, no level clear**. This confirms the observed fine-tuned run, not a before/after comparison; no `comparison.json` was supplied for this audit. The raw user trace stays outside this repository.

| Observed issue | Evidence and explanation | What must change before new labels |
| --- | --- | --- |
| Ghost avoidance | At turns 16, 135 and 250, Kev chooses the fatal action with probabilities 0.8311, 0.8704 and 0.8776. A native counterfactual reversal avoids each immediate collision. At turn 134 the planner already recommends leaving the corridor. | More survival-critical, learner-visited states; explicit current movement/timer information; an immediate-safety gate in the teacher. Avoiding one collision is not proof of eventual survival. |
| Loops | A **24-decision circuit** repeats without pellet progress. The longest within-life dry spell is **73 decisions / 593 active frames**. Kev keeps its heading at 210/215 opportunities, including 24/29 junction opportunities; training labels keep it at only 190/516 comparable junction opportunities. | Longer cycle/coverage information and a teacher objective for progress and maze cleanup. Heading bias is established; its learning cause still needs controlled model probes. |
| Skipped pellet branches | At turn 162, from row 8/column 6, the left branch contains a dot; Kev chooses up. There are 22 junction decisions with zero pellet progress despite an immediately nonfatal pellet-collecting alternative. **All 69 observed arrivals on pellet-containing tiles consume the pellet.** | Teach routing into remaining branches. The native consumption mechanism works; a locally safe dot is not always the best long-term action. |
| Action contract | Turns 70, 81 and 92 travel **three, two and two tiles** before another decision. Some reversal alternatives return to the same centre after two frames. The native actor steers after a movement substep, and the integration checks centres only after a whole frame. | A separately versioned adjacent-tile-entry boundary over native frames, correct reversal and explicit spawn alignment; matching browser/CPU tests before regenerating data. |

The audited **v1** observation contains the maze, remaining normal/power pellets, named ghost positions/headings/modes, current power mode, phase, legal moves, lives, a 12-position history and visits to the current tile. It omits exact native movement/timing information and cannot contain the preceding 24-step circuit. The v2 observation above supplies that missing current information. Present pellet/ghost facts were already visible, so missing fields alone do not explain every wrong choice.

There is also a substantial collection gap: **all 4,096 training states have three lives**, while **234/251 supplied decisions happen after a death**. Death resets release counters and phase timing while global elapsed frames continue. The old evaluation has only **23/256 survival-critical search decisions, 8/256 with danger within one maze tile, three heavily revisited positions, and no post-respawn or ≤30-pellet endgame examples**. Averaging agreement across that set can improve while live play remains poor. This trace consumes one power pellet and eats zero ghosts; it cannot establish power-expiry or ghost-hunting skill.

The teacher also needs repair. Its five-seed CPU report raises mean score to 3,590 but produces **zero clears and four capped runs**. At turn 135 its 20-move search reports both actions eventually dying and selects the immediately fatal one; at turn 195 three actions have exactly equal search values. Lifetime visit penalties can discourage necessary transit through empty corridors. We should not multiply these labels and assume more examples will fix them.

While auditing every counterfactual, we found a further native **rewind omission**: `ghostReleaser.save/load` does not preserve the global/personal release-counter mode. A hypothetical move can switch it after a respawn and contaminate restored gameplay. The new CPU benchmark augments that rewind hook to preserve both counter sets and mode; forward physics, v1 action semantics and learner observations are unchanged. It then matches all 251 trace states exactly. The v2 teacher must inherit this repair before collecting multi-life examples.

The [ordered diagnosis plan](evaluation/pacman-v2-plan.json) records the original hypotheses. Movement/state fixes and complete-game evaluation are now implemented. Teacher qualification precedes any new dataset. Scripted native post-respawn recovery is included in v2 collection. Learner-visited mistake states and DAgger rounds remain a further collection step, requiring teacher validation on those actual trajectories and separate, reserved benchmark seeds. On-policy and scripted-recovery success do not prove that a student will recover from all off-policy mistakes. Seed 7 must stay out of evaluation if its observed failures inform training.

Controlled option-order probes and a new GPU before/after run are still required to establish the learned model's improvement. The native CPU teacher results establish only the algorithm's behavior on the stated suite. Raw user traces remain private.

## Gameplay benchmark

`comparison.json` uses schema 3: `gameplay.general`, `gameplay.fine_tuned` and paired per-seed differences. Complete requests, responses, verified adapter identities and native diagnostics are saved under `results/gameplay-general/` and `results/gameplay-fine-tuned/`, and included in the submission. The mandatory comparison has no isolated snapshot score. Measure serving latency and actual episode lengths before promising classroom timing.

Counterfactual actions diagnose immediate avoidable deaths from identical native state/RNG; they never select Kev's action or insert a safety filter. Empty danger denominators are unavailable, not perfect safety. Report native game-overs and watchdog failures separately. Current benchmark scope is the same classic maze at levels 1,2,3,5, not unseen mazes or levels.

To reproduce a private recorded game locally, run:

```bash
python3 labs/lab-01-kev-pacman/gameplay_benchmark.py /path/to/player-trace.jsonl \
  --seed 7 --level 1 --out /path/to/private-audit.json
```

Replay fails visibly on a differing state or adapter identity. The diagnostic output can contain trajectory details; keep it private unless sharing is intended. The published audit contains derived findings only. CPU tests cover all-life continuation/game-over, counterfactual isolation across release-mode transitions, exact trace replay, identity/state corruption, censoring, loop progress resets, protocol pairing and empty snapshot strata. They do not substitute for the requested baseline/fine-tuned GPU comparison.

## Validation and provenance

Local CPU tests cover rules/data splits, all stage commands, TensorBoard events, process failures, LoRA recovery, kernel installation guards and checkpoint provenance. Completion of the optimized 4B profile, CUDA recovery, timing, reloads and live Colab callback remains **unvalidated**. Run the complete notebook on the intended allocation before teaching and prepare compatible instructor checkpoints. Label supplied results as instructor results.

Kev code is pinned at `84847f0a883d900f7de5b7a57eaa341ca7f9a6b4`. [upstream.json](upstream.json) records base, suite-mirror and optional released-model pins. Downloaded source partitions are verified against their manifests/checksums. Vendored recipe files remain unchanged and licensed.

From the repository root:

```bash
python3 -m pip install -r labs/lab-01-kev-pacman/test-requirements.txt
python3 -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python3 -m unittest discover -s labs/lab-01-kev-pacman -p 'test_*.py'
python3 labs/lab-01-kev-pacman/build_artifacts.py
python3 scripts/check_notebook.py
```

After helper edits, commit them before rebuilding with `--pin-source`, then commit the notebook/lock. Ordinary rebuilds preserve the source pin. The builder does not write Google Slides.

Sources: [Kev-4B model card](https://huggingface.co/jaredpalmer/kev-4b), [initial recipe](https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/experiments/q35-4b-s23.json), [Qwen optimized kernels](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5), [FLA release](https://github.com/fla-org/flash-linear-attention/releases/tag/v0.5.2), [CUDA convolution release](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0). Kev/CLM/llama.cpp/SGLang comparison sources remain in the slide notes.
