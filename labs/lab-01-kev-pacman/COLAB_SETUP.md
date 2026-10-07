# Colab setup for Kev-4B Pac-Man

Target **one NVIDIA RTX PRO 6000 Blackwell GPU**, nominally 96 GB on the full Server Edition. Inspect the actual allocation. An RTX 6000 Ada or RTX A6000 is a different card. Arrange access before class; Colab does not guarantee this GPU, including on paid plans. Measure cost using actual compute units and elapsed time.

## Prework

1. Open [Lab 1 in Colab](https://colab.research.google.com/github/yxc20089/QPlusLearning/blob/main/labs/lab-01-kev-pacman/notebooks/pacman_kev_lab.ipynb) and save a fresh copy. Select the target GPU if offered.
2. Run setup. It downloads hash-verified helpers, classic-game source/assets and starter data, installs pinned Kev in a separate Python 3.13 environment and adds the hash-pinned optimized training wheels without replacing locked dependencies.
3. Keep `runtime-preflight.json` and run the separate preparation cell with `RUN_TRAINING_PREFLIGHT=False` (default). This fetches the pinned **Qwen3.5-4B-Base**, audits LoRA/head parameters on CPU, and checks pinned optimized bindings/CUDA/BF16 availability. Keep `optimized-training-preflight.json`; it records `bindings_verified`, zero optimizer steps and `cuda_loss_backward='not_run'`. The first real training step exercises loss/backward/fused AdamW and saves a recovery checkpoint. Reference-kernel fallback stops training.
4. Optional: set `RUN_TRAINING_PREFLIGHT=True` for two extra suite-record training checks. This runs without a CUDA profiler, verifies gradients, and counts actual FLA/convolution/fused AdamW calls. Both this check and real training dump Python stacks after 60 seconds without progress. First-use compilation/autotuning may still occur in training when the check is skipped. The binding-only attempt has a two-minute cap; the optional check has a 20-minute cap. These are timeout limits, not duration estimates.

5. Open TensorBoard before training. Select `SAVE_TO_DRIVE=True` before any run if weights should survive runtime loss. Local `/content` is temporary.
6. Complete **initial → dates/evidence → documents → skills/devtools** in the four separate prework sections. Each saves its own checkpoint and backup. Use fresh `kev-4b-*` directories: 0.8B adapters cannot initialize 4B. Keep all four ZIPs and stage curves.
7. The notebook starts the skills checkpoint for inference. Confirm `/v1/models` and a legal player decision. Record the starting compute-unit balance for the class.

If an older notebook is stuck in the two-record preflight, interrupt its cell and keep the runtime connected. Copy/run the updated bootstrap cell to refresh verified helpers without deleting downloads or checkpoints. Rerun setup, preparation with the default `False`, and storage before continuing. Cached downloads are reused.

The [recipe table](README.md#the-five-training-stages) lists exact source counts, replay and selected settings. General training and native-engine planning-label generation belong outside the 90-minute class. Pac-Man uses **4,096 task examples plus 2,000 decision-v7 replay examples, one epoch, lr 2e-5, batch 4 × accumulation 2 / row budget 0: 762 updates**. This replaces the former 16-update demonstration. The class keeps a mandatory 30-minute fine-tuning block with a 20-minute compute target. Measure the full stage before class: at 1.5 seconds/update compute alone is 19 minutes; at 3 seconds it is 38 minutes. Complete slower full runs as prework. All full-stage attempt caps are 180 minutes, scheduling limits rather than measured durations; no automatic short-step truncation is used.

## Optimized execution and precision

Kev's frozen stack uses Torch 2.8.0/CUDA 12.8, Triton 3.4.0, Transformers 5.17.0 and PEFT. The additive overlay pins FLA/fla-core 0.5.2, causal-conv1d 1.7.0, einops 0.8.1 and Ninja 1.13.0. The CUDA wheel matches Python 3.13, Linux x86_64, Torch 2.8 and CXX11 ABI. FLA handles DeltaNet, CUDA handles short convolution, and SDPA handles full attention. BF16 autocast, FP32 stored parameters and fused AdamW are required. Kev inference fusion/CUDA graphs are disabled for training.

Default `TRAINING_PROFILE='memory_safe'`: batch 1 × accumulation 8, non-reentrant checkpointing, `row_budget=2048` for Dates, Documents and Skills. Initial and Pac-Man explicitly select batch 4 × accumulation 2 and row budget 0. Source data, effective batch and schedules remain unchanged; microbatch/dropout execution and kernel arithmetic differ. The published-reference profile retains original intermediate-stage execution flags but still requires optimized kernels. A row budget does not truncate a single longer question. Optional allocator `expandable_segments` cannot release actively used tensors.

Stage 1 has one resume selector: `RESUME_CHECKPOINT` defaults to the latest available snapshot, or a fresh run if none exists. Use `'latest'` to require recovery, a complete snapshot directory path to select a save, or `None` for a fresh output. Completed outputs are reused. The cell selects 4 × 2 / row budget 0 before printing its command; it runs no separate timing trials. Initial continuation permits only batch/accumulation/row-budget changes, preserves effective batch and total steps, and maps progress to the same next source records. Optimizer/scheduler/RNG and execution history are restored. Grouping/dropout may change numerics. Intermediate/task resume requires matching arguments.

This contract requires BF16 and rejects T4/P100 as drop-in training fallbacks. An explicitly allowed alternative GPU must satisfy the same software/kernel checks and receive its own full validation. Kaggle supports structured evaluation/rollouts but does not provide the Colab browser callback.

## Progress and interruption

TensorBoard runs in the notebook kernel while Kev trains in its locked environment. Separate `logs/<stage>/<attempt>` directories contain per-step events, CSV/JSONL, stdout/stderr and status. Curves include CE, learning rate, gradient norm, timing, records and peak/live allocated/reserved/free VRAM. `batches.jsonl` preserves token shapes and IDs before each forward pass. These are training curves, not fabricated validation results.

Snapshots save adapter/head, optimizer/scheduler/RNG and progress after step 1, every 100 steps or five minutes at optimizer boundaries, and the final step. The latest two complete saves remain under `<output>-recovery`. After interruption, keep data, parent and paths and rerun the affected stage; Stage 1 uses `RESUME_CHECKPOINT`, while later stages detect matching recovery automatically. Do not use Kev's native full-weight-only resume flag. Export recovery and logs after exceptions, or use Drive storage. Recovery ZIPs restore to the original absolute checkpoint root. Completed checkpoint ZIPs restore through `RESTORE_ARCHIVE`, `RESTORE_DATES_ARCHIVE`, `RESTORE_DOCUMENTS_ARCHIVE` and `RESTORE_SKILLS_ARCHIVE`, in order. Declare learner/instructor ownership.

The old 0.8B run failed at step 2,073 with missing FLA/convolution kernels and nearly all 95 GiB actively occupied. Its notebook saved weights only at completion, so that failed run has no automatic learned checkpoint. Inspect its old output directory before restarting; logs/configuration cannot reconstruct exited-process weights. Fresh 4B output names preserve those files.

## Before teaching

The local CPU tests do not establish CUDA compatibility or full-stage fit. **The optimized 4B CUDA preflight, full curriculum, CUDA continuation, reloads and live browser callback have not been run by the course author.** Complete them on the intended allocation, measure duration/compute-unit use and prepare compatible instructor checkpoints. Keep the 90-minute class and 30-minute fine-tuning block; report supplied results as instructor results.

Sources: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [RTX PRO 6000 Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/), [Kev-4B card](https://huggingface.co/jaredpalmer/kev-4b), [Qwen kernels](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5).

## Classic Pac-Man update

Use the new bootstrap and the single **CP1 — Fine-tune and evaluate Kev on Pac-Man game states** section. After bootstrap on an existing or fresh runtime, run setup, optimized preparation with `RUN_TRAINING_PREFLIGHT=False`, and storage with `SAVE_TO_DRIVE=True` if using Drive. Bootstrap creates a new runtime helper, so those cells re-establish its hardware, training and backup settings; existing files are reused. Run **Load completed Skills checkpoint — start the lab here**, then CP1. Keep the completed Skills baseline. CP1 writes `kev-4b-pacman-native-v2`, with qualified `pacman-native-v2-*` demonstrations and a 4,096-token state budget. Existing `kev-4b-pacman-planner-v1` weights are kept separate. It prints the exact 4 × 2 command and automatically resumes matching recovery. With `SAVE_TO_DRIVE=True`, the new checkpoint uses the same automatic step-1 / periodic / final backup flow.

**Interactive play** remains a separate, ungraded activity. It starts with the completed Skills model; CP1's evaluation leaves the fine-tuned Pac-Man adapter serving. Rerun the play cell after evaluation and before export to try that adapter. The badge identifies the actual live adapter and hashes. CP1 includes label review, the full task training stage, paired held-out evaluation and submission; Stages 1–4 remain separate prework.

Each interactive-play and comparison invocation prints its own dated recording directory under `results/`; repeated runs and v1/v2 adapters do not overwrite earlier recordings. State/action/probability traces are flushed after every model decision. Partial benchmark recordings remain available if evaluation is interrupted.

Before replacing the runtime, pause play and run the final **Back up to Google Drive — checkpoints, all trajectories and traces** cell. It mounts Drive even if automatic backup was previously disabled, saves available checkpoints/latest recovery snapshots under `MyDrive/QPlusLearning/lab-01-kev-pacman/backups/<checkpoint-name>/`, and preserves all results, logs/TensorBoard events, labels and teacher replays in a dated `session-backups/session-….zip`. It does not require a completed benchmark or submission. Wait for **Session backup complete** and verify `checkpoint_errors` is empty. Checkpoints retain the existing automatic-restore flow; session archives remain separate on Drive.

Node.js executes the same pinned four-ghost engine for CPU evaluation. Setup uses an installed Node 18+ or a checksum-pinned official Node 22.17.0 binary. The native renderer, font, sounds and classic mechanisms are included. Human play uses 60 simulation frames per second; Kev pauses simulation time while deciding.
