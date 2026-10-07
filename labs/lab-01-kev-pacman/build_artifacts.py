"""Rebuild the stage-by-stage Kev notebook and browser game."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HELPERS = ["api_client.py", "pacman_lab.py", "cloud_runtime.py", "training_monitor.py",
           "lora_recovery.py", "checkpoint_backup.py", "optimized_training.py", "training-kernels.json", "training_stages.py", "training-stages.json", "games/arcade-engine.js", "games/arcade-worker.js", "games/arcade-browser.js", "games/arcade-shell.html",
           "vendor/arcade-pacman/source.json", "vendor/arcade-pacman/source.zip",
           "planner_data.py", "games/arcade-planner.js", "data/pacman-planner-v1.zip",
           "data/pacman-planner-v1-manifest.json", "data/pacman-planner-v1-quality.json",
           "gameplay_benchmark.py", "benchmark-spec.json", "games/benchmark-hooks.js", "games/benchmark-worker.cjs",
           "games/arcade-teacher.js", "teacher_validation.py", "teacher-validation-spec.json", "teacher_data.py",
           "evaluation/teacher-qualification.json", "evaluation/teacher-qualification-replays.zip",
           "data/pacman-native-v2.zip", "data/pacman-native-v2-manifest.json", "data/pacman-native-v2-replays.zip", "data/pacman-native-v2-generator.py"]


def source_lock(pin=False):
    path = ROOT / "notebook-source.json"
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in HELPERS}
    if pin:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        # A notebook must only fetch reviewed, committed sources.
        for name in HELPERS:
            content = subprocess.check_output(["git", "show", f"{revision}:labs/lab-01-kev-pacman/{name}"], cwd=ROOT)
            if hashlib.sha256(content).hexdigest() != hashes[name]:
                raise ValueError(f"Commit helper changes before pinning the notebook: {name}")
        path.write_text(json.dumps({"repository": "yxc20089/QPlusLearning", "revision": revision,
                                    "files": hashes}, indent=2) + "\n")
    lock = json.loads(path.read_text())
    if lock["files"] != hashes:
        raise ValueError("Helper sources changed; commit them and rebuild with --pin-source")
    return lock


def notebook():
    cells = []
    def add(kind, content):
        cell = {"id": f"cell-{len(cells):02d}", "cell_type": kind, "metadata": {}, "source": content.strip().splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    md = lambda text: add("markdown", text)
    code = lambda text: add("code", text)
    md("""# Lab 1 — Train a Decision Model to Play Pac-Man

Follow Kev's published **LoRA plus pointer-head** architecture. First train a fresh decision model from `Qwen/Qwen3.5-4B-Base` on the frozen `decision-v7` training suite. Continue through separate dates/missing-evidence, documents, and skills/devtools stages, then fine-tune your own adapter and head on Pac-Man labels. Original base matrices stay frozen, and LoRA changes the encoder's effective features. This is initial decision-model training over an already pretrained LLM.

**You train Pac-Man. Four ghosts use the classic arcade engine.** The faithful browser recreation supplies the classic maze, original renderer/font/sounds, power pellets, fruit, tunnels, lives and chase/scatter phases. Kev chooses player directions; upstream code controls Blinky, Pinky, Inky and Clyde.

The lab has **one assessed checkpoint: CP1 — fine-tune and evaluate Kev on Pac-Man game states**. Interactive play with the trained adapter is a separate, ungraded activity. The 90-minute route is 0–20 overview and interactive play; 20–30 inspect planning labels; **30–60 mandatory fine-tuning**; 60–80 evaluate; 80–90 explain and submit the checkpoint evidence. Installation, downloads, label generation and the four general-decision training stages are prework.

The target runtime is **Colab with one NVIDIA RTX PRO 6000 Blackwell GPU**. The full Server Edition has 96 GB VRAM. Check the actual allocation in the prework cell. The notebook uses BF16 inference and training autocast, with FP32 stored backbone, adapter and head parameters in the initial recipe. Colab does not guarantee this GPU, including on paid plans. Arrange access before class and record actual runtime cost. All stages still need an instructor GPU preflight.

The notebook separates **Stage 1: initial decision training → Stage 2: dates/missing evidence → Stage 3: documents → Stage 4: skills/devtools → Stage 5: Pac-Man fine-tuning**. Each saves its own checkpoint and training logs. Calibration is a separate probability-fitting procedure and is not applied here; these fresh runs do not inherit the released model's benchmark scores or fitted temperature. [Recipe and compute](https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/docs/model-cards/kev-4b.md#training-procedure).""")

    md("""## Game provenance

Game source: [masonicGIT/pacman](https://github.com/masonicGIT/pacman), GPL-3.0, pinned `7407174c1d6a38be8cd230577489e39e0873145b`. Its source/assets are unchanged; the author documents small [accuracy differences](https://github.com/masonicGIT/pacman#accuracy). Browser and CPU evaluation execute this same arcade engine. Model source: [jaredpalmer/kev](https://github.com/jaredpalmer/kev). The model receives structured state, not screenshots. Inference stays inside the GPU runtime. The board identifies the active adapter, stage, steps, rank and checkpoint hashes.""")
    md("""## Prework: prepare the runtime

Save your own copy of this notebook in Colab. Open **Runtime > Change runtime type**, select the RTX PRO 6000 Blackwell option if your account offers it, then connect. Run the prework cells before class. Confirm the GPU name rather than relying on a menu label. A different allocation requires an instructor-approved, timed fallback.

Setup creates a separate Python 3.13 environment using Kev's locked dependencies, including PyTorch 2.8.0 with CUDA 12.8. It checks GPU identity, BF16 support and a CUDA forward/backward pass in that environment, then writes `runtime-preflight.json`. The notebook kernel only runs the teaching helpers and Colab callback. Complete installation, data/model downloads and all four general stages before class.

The target GPU is an assumption for this session, not a free-compute promise. [Colab availability](https://research.google.com/colaboratory/faq.html), [NVIDIA GPU specifications](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/), [PyTorch Blackwell support](https://pytorch.org/blog/pytorch-2-7/).""")
    lock = source_lock()
    code("""from pathlib import Path
from urllib.request import urlopen
import hashlib, json, os, sys
TRAINING_PROFILE = 'memory_safe'  # Later stages; Stage 1 explicitly selects 4 x 2 below
LAB_DIR = Path.cwd() / 'pacman-kev-lab'
LAB_DIR.mkdir(exist_ok=True)
COURSE_REVISION = """ + repr(lock['revision']) + """
FILES = """ + repr(lock['files']) + """
for name, expected_sha256 in FILES.items():
    target = LAB_DIR / name
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected_sha256:
        continue
    url = f'https://raw.githubusercontent.com/yxc20089/QPlusLearning/{COURSE_REVISION}/labs/lab-01-kev-pacman/{name}'
    with urlopen(url, timeout=60) as response:
        content = response.read()
    assert hashlib.sha256(content).hexdigest() == expected_sha256, f'Unexpected source: {name}'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    print('Downloaded verified helper:', name, flush=True)
sys.path.insert(0, str(LAB_DIR))
# Refresh helpers when upgrading an existing notebook; keep checkpoints/logs.
if 'runtime' in globals():
    runtime.stop()
import importlib
importlib.invalidate_caches()
for name in ['cloud_runtime', 'training_monitor', 'lora_recovery', 'checkpoint_backup', 'optimized_training', 'training_stages', 'gameplay_benchmark', 'teacher_validation', 'teacher_data', 'planner_data', 'pacman_lab', 'api_client']:
    sys.modules.pop(name, None)
from cloud_runtime import CloudRuntime
from lora_recovery import latest_snapshot
from training_stages import specifications, inspect_checkpoint, restore_checkpoint, backup_checkpoint
from pacman_lab import *
from teacher_data import prepare_dataset, PREFIX, RECIPE
from gameplay_benchmark import benchmark_gameplay, paired_gameplay
from api_client import call, distribution
ensure_node(LAB_DIR)
runtime = CloudRuntime(LAB_DIR, training_profile=TRAINING_PROFILE)
CHECKPOINT_ROOT = LAB_DIR / 'checkpoints'
INITIAL = CHECKPOINT_ROOT / 'kev-4b-initial'
DATES = CHECKPOINT_ROOT / 'kev-4b-dates'
DOCUMENTS = CHECKPOINT_ROOT / 'kev-4b-documents'
SKILLS = CHECKPOINT_ROOT / 'kev-4b-skills'
STAGES = specifications()
manifest = None  # Loaded with its teacher qualification receipt in CP1
GAME = notebook_game(action_version=2)
STAGE_OWNERS = {}
print('Prepared the native arcade engine, four ghosts and full-game benchmark. CP1 verifies qualified demonstrations.')
""")
    md("""Setup fetches readable helpers, a compressed planning dataset and a 9.7 MB archive of original game source/assets, verifying every SHA-256. CP1 extracts the two declared training/development files after checking their individual hashes and the teacher qualification receipt. The instructor qualifies the teacher on complete native games before generating labels. Native replays verify the receipt; learners do not generate demonstrations during the training block. Node.js executes that same engine for Python evaluation; Colab installs a checksum-pinned official Node binary only if needed, separate from Torch. Cached downloads are reused. Network access is needed on first use.

## Training monitor: open TensorBoard before running any stage

Colab supports TensorBoard inside the notebook. Every stage writes a separate run below `logs/`: per-optimizer-step cross-entropy, next-step learning rate, gradient norm before clipping, epoch, step time, records seen, peak/live allocated memory, reserved memory and free VRAM. These are training curves; no validation loss or accuracy is invented. Raw subprocess output, JSONL and CSV are also saved, including failed attempts. `batches.jsonl` records question-row counts, longest padded rows and record IDs before each forward pass, including the batch that fails.

The monitor checks the exact Kev trainer checksum and instruments telemetry plus LoRA save/restore hooks in memory. It leaves the upstream checkout intact. Console progress appears at the first step, every ten steps and the last step; a 15-second heartbeat also covers loading and data preparation. [TensorBoard in Colab](https://www.tensorflow.org/tensorboard/tensorboard_in_notebooks).""")
    code("""import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'tensorboard==2.20.0'])
LOG_DIR = LAB_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)
SHOW_TENSORBOARD = True
if SHOW_TENSORBOARD:
    from IPython import get_ipython
    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic('load_ext', 'tensorboard')
        ip.run_line_magic('tensorboard', f'--logdir "{LOG_DIR}" --reload_interval 5')
    else:
        print('Outside a notebook, run: tensorboard --logdir', LOG_DIR)
""")
    code("runtime.setup()")
    md("""## Required optimized training stack

Setup extends Kev's frozen environment with hash-pinned wheels: **Flash Linear Attention / fla-core 0.5.2**, **causal-conv1d 1.7.0** (Python 3.13, Torch 2.8, CUDA 12, Linux x86_64, CXX11 ABI), einops 0.8.1 and Ninja 1.13.0. Torch 2.8.0 / CUDA 12.8, Triton 3.4.0, Transformers 5.17 and PEFT remain at Kev's locked versions. Dependencies are installed without replacing that stack. SDPA handles full attention; FLA's Triton kernels handle DeltaNet; the CUDA convolution handles its short convolution. Training uses fused AdamW, BF16 autocast, gradient checkpointing and gradient accumulation; the later-stage memory profile also bounds row passes. The LoRA/head recipe and curriculum remain in Kev's trainer.

Preparation audits the actual pinned base/LoRA/head on CPU and verifies the selected optimized function bindings and CUDA/BF16 availability. It does **not** run a separate model forward/backward by default. `optimized-training-preflight.json` records `bindings_verified`, zero optimizer steps and `cuda_loss_backward='not_run'`. Every training subprocess verifies bindings again and refuses reference-kernel fallback. The first real training step exercises loss/backward/fused AdamW and saves a recovery snapshot after successful completion.

Set `RUN_TRAINING_PREFLIGHT=True` below only when you want the additional two-record compatibility check. This loads the actual model on CUDA, checks finite adapter/head gradients and counts FLA, convolution and fused AdamW calls without running a CUDA profiler. It reports encoding, forward, backward, gradient checks and completed optimizer updates separately. Both this optional check and real training print Python stacks after 60 seconds without progress; logs retain the active call. Fifteen-second heartbeats show process liveness. The binding-only attempt has a two-minute timeout; the optional training check has a 20-minute timeout. These are limits, not expected durations. Initial kernel compilation/autotuning can still happen during the first real training step. Full-curriculum CUDA compatibility, VRAM and timing remain unvalidated by the course author.

To retry after a stalled old preflight, interrupt its cell and keep the Colab runtime connected. Copy the updated bootstrap cell into your notebook and run it to refresh the verified helpers; existing downloads, checkpoints and logs stay in place. With the existing environment already installed, run the preparation cell below with the default `False`. Installation and model verification now have separate cells.

Fused optimizer/kernel arithmetic can differ numerically from the reference implementation. The published-reference profile retains upstream training flags; both profiles now require optimized kernels. Optional inference fusion and serving CUDA graphs remain disabled for training because they do not implement this backward path. [FLA release](https://github.com/fla-org/flash-linear-attention/releases/tag/v0.5.2), [causal-conv1d release](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0).""")
    code("""RUN_TRAINING_PREFLIGHT = False  # Optional extra model checks; optimized bindings remain required
audit = runtime.prepare_training(run_training_preflight=RUN_TRAINING_PREFLIGHT)
print('Training suite records:', audit['suite_records'])
print('Trainable parameters:', sum(audit['trainable_parameters'].values()))
print('Optimized training verification:', json.dumps(runtime.training_preflight, indent=2))
""")
    md("""## Memory profile and recovery storage

A learner's original **0.8B** batch-8 run failed at step 2,073/3,144 with 89.90 GiB actively allocated on a 94.97 GiB GPU. Startup warnings confirmed `causal_conv1d` and `flash-linear-attention` were missing, so Qwen used reference PyTorch kernels with substantial training intermediates. The peak reached 91.91 GiB by step 280 and then stayed flat through step 2,070; that cumulative maximum alone cannot diagnose a leak. [Qwen kernel documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5#usage-tips-and-notes). Record counts are not physical batch sizes: records can contain multiple question rows and none-pair siblings.

The **Stage 1 cell explicitly selects batch 4 × accumulation 2 and `row_budget=0`**, matching Kev's published initial execution. This assignment happens in the training cell before the command is printed, so the generic profile cannot silently replace it. The default **`memory_safe`** profile still supplies **batch 1 × accumulation 8**, gradient checkpointing and `row_budget=2048` to later stages. Both executions preserve the data, epochs, optimizer-step count, architecture, learning-rate schedule and augmentation settings. Microbatching, row grouping and dropout execution differ; a continued run that changes these settings is not an exact numerical reproduction. The published later-stage execution flags remain available as `TRAINING_PROFILE='published_reference'`. Full-run CUDA fit and timing remain unvalidated by the course author. A single question longer than a nonzero row budget still runs intact; records are not silently truncated or dropped.

Save at optimizer step 1, every 100 steps or five minutes (checked at optimizer boundaries), and the final step. Keep the latest two complete snapshots. Each contains the LoRA adapter/head, optimizer, scheduler, RNG and progress counters. Kev's native `--resume` supports full-weight runs only; the lab adds a separate LoRA recovery implementation. After a failure, Stage 1 uses its `RESUME_CHECKPOINT` selector; later stages use their `RESUME_*` flags. Keep the same inputs and output path, and rerun only the affected stage. Stage 1 permits its explicit batching change; other training arguments must match. Do not rerun completed earlier stages. A snapshot can resume training or supply an intermediate model; it is not a completed curriculum stage.

**The older notebook saved LoRA weights only at the end. Its failed step-2,073 run has logs/configuration but no automatic learned checkpoint.** New output names below preserve that failed directory. Local `/content` files disappear when the runtime is replaced.

Set **`SAVE_TO_DRIVE=True`** below before training. Colab mounts Drive once. Training and recovery saves stay on the local disk; after each complete recovery save, the notebook copies a verified archive to `MyDrive/QPlusLearning/lab-01-kev-pacman/backups`. It keeps the latest two backups per stage and also archives the completed final checkpoint. Adapter/head weights, optimizer, scheduler, RNG, progress and any custom training JSONL are included; foundation weights and packages are downloaded again by setup. Wait for **`Drive backup complete at optimizer step …`** before deleting a runtime. If backup fails, the local snapshot remains and training stops with the error.

On a new runtime, enable the same flag and run this storage cell. Available Drive backups automatically restore to their original local paths without overwriting existing local checkpoints or changed labels. Stage 1 then detects the restored snapshot and resumes by default. With no checkpoint it starts fresh. You can still select `latest` or a specific snapshot explicitly. Logs and TensorBoard events have their separate export control below.""")
    code("""SAVE_TO_DRIVE = False  # True enables automatic backup and restore; no manual file copying
RESTORE_RECOVERY_ARCHIVE = None  # ZIP from the inspection/export cell; restore into an absent root
CHECKPOINT_ROOT = LAB_DIR / 'checkpoints'
DRIVE_BACKUP_ROOT = None
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_BACKUP_ROOT = Path('/content/drive/MyDrive/QPlusLearning/lab-01-kev-pacman/backups')
if RESTORE_RECOVERY_ARCHIVE is not None:
    restore_checkpoint(RESTORE_RECOVERY_ARCHIVE, CHECKPOINT_ROOT)
else:
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
print('Automatic Drive backup:', json.dumps(runtime.configure_backup(DRIVE_BACKUP_ROOT, CHECKPOINT_ROOT), indent=2))
print('Later-stage training profile:', runtime.training_profile, 'overrides:', runtime.selected_recipe({}))
print('Stage 1 selects batch 4 x accumulation 2 / row_budget 0 in its own training cell.')
print('Checkpoints and recovery:', CHECKPOINT_ROOT)
""")
    md("""### Inspect a failed run / export recovery

This cell is safe to rerun after a training exception. It checks files rather than assuming a checkpoint exists. Select the failed stage's output in `INSPECT_OUTPUT`; choose the original `decision-v7-initial` path to inspect the earlier OOM. Set `EXPORT_RECOVERY=True` to download the output plus its recovery snapshots. Download logs separately before disconnecting. A ZIP containing only configuration files cannot recover model weights.""")
    code("""INSPECT_OUTPUT = CHECKPOINT_ROOT / 'kev-4b-initial'
EXPORT_RECOVERY = False
EXPORT_LOGS = False
def show_training_state(output):
    output = Path(output)
    weights = [output / 'head.pt', *output.glob('adapter_model.*')]
    print('Output:', output, 'exists:', output.exists())
    print('Weight files:', {p.name: p.stat().st_size for p in weights if p.is_file()})
    print('Completed metrics:', (output / 'training_metrics.json').is_file())
    pointer = Path(str(output) + '-recovery') / 'latest.json'
    print('Latest recovery:', json.loads(pointer.read_text()) if pointer.is_file() else 'none')
show_training_state(LAB_DIR / 'checkpoints/decision-v7-initial')
show_training_state(INSPECT_OUTPUT)
archives = []
if EXPORT_RECOVERY:
    import zipfile
    archive = LAB_DIR.parent / 'pacman-stage-recovery.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as saved:
        for folder in [INSPECT_OUTPUT, Path(str(INSPECT_OUTPUT) + '-recovery')]:
            for file in folder.rglob('*'):
                if file.is_file():
                    saved.write(file, file.relative_to(INSPECT_OUTPUT.parent))
    archives.append(archive)
if EXPORT_LOGS:
    archives.append(backup_checkpoint(LOG_DIR, LAB_DIR.parent / 'pacman-training-logs.zip'))
for archive in archives:
    try:
        from google.colab import files
    except ImportError:
        print('Save:', archive)
    else:
        files.download(str(archive))
""")
    md("""## Stage 1 — Initial decision training (prework)

Only this stage starts fresh LoRA adapters and a pointer head over the pretrained Qwen base. It uses the selected published `experiments/q35-4b-s23.json` seed-2 recipe: all 12,576 `decision-v7` training records, two epochs, batch 4, accumulation 2, gradient checkpointing, learning rate 5e-5, rank 16 / alpha 32, a 256-dimensional head and BF16 autocast over FP32 weights. Option shuffling, none/distractor insertion and 25% none minimal pairs remain active. There is no `--init_from`.

The initial cell performs this stage only. It explicitly sets `INITIAL_EXECUTION={'batch': 4, 'accum': 2, 'row_budget': 0}` before constructing the command; gradient checkpointing remains enabled. The generic memory profile applies to later stages. **By default it resumes a complete recovery snapshot when one exists locally or was restored from Drive; with no checkpoint it starts fresh.** You can select `latest` or a specific saved snapshot with `RESUME_CHECKPOINT`; an explicitly requested missing checkpoint stops the run. A completed initial checkpoint is reused. It streams progress, saves recovery snapshots and writes TensorBoard curves. The full run is prework with a configurable 180-minute attempt cap, not a promised duration. To restore a completed initial checkpoint manually, upload its ZIP, set `RESTORE_ARCHIVE` and declare learner/instructor ownership.""")
    md("""### Resume initial training

Keep the same `INITIAL` output. The cell below uses one setting: **`RESUME_CHECKPOINT`**. Its default detects whether a recovery snapshot exists. Set it to **`'latest'`** to require the latest checkpoint, or to a full snapshot directory path to choose a specific save. Set it to `None` for a fresh run in a new output. The launcher prints the saved optimizer step and selected path before restoring weights, optimizer, scheduler, RNG and progress. A missing or incomplete explicitly requested checkpoint stops the run; it does not start fresh.

The cell explicitly selects **batch 4 × accumulation 2 and row budget 0**, permitting only the batching changes needed to continue an earlier 1×8 run. Base, data, epochs, learning rates, precision, effective batch and total optimizer steps must match. Grouping and dropout draws can change, and the execution change is recorded in recovery receipts. Work after the chosen snapshot is repeated. For a student's first run with no checkpoint, explicitly set `RESUME_CHECKPOINT=None` to initialize a new output. Existing outputs are protected from accidental replacement.""")
    code("""INITIAL = CHECKPOINT_ROOT / 'kev-4b-initial'
INITIAL_EXECUTION = {'batch': 4, 'accum': 2, 'row_budget': 0}  # Explicit published initial execution
runtime.initial_execution = dict(INITIAL_EXECUTION)
print('Selected initial execution:', runtime.initial_execution, flush=True)
print('Selected command:', runtime.pretraining_command(INITIAL), flush=True)
from lora_recovery import latest_snapshot
RESUME_CHECKPOINT = 'latest' if latest_snapshot(Path(str(INITIAL) + '-recovery')) else None
# To choose a saved step, replace the line above with its full snapshot directory path.
RESTORE_ARCHIVE = None  # Example: '/content/pacman-initial-checkpoint.zip'
RESTORED_CHECKPOINT_OWNER = 'learner'  # 'instructor' for a supplied fallback
if RESTORE_ARCHIVE is None:
    if (INITIAL / 'run-evidence.json').is_file():
        print('Using completed initial checkpoint:', INITIAL)
    else:
        runtime.pretrain(INITIAL, steps=0, resume_from=RESUME_CHECKPOINT,
                         allow_execution_change=RESUME_CHECKPOINT is not None)
    STAGE_OWNERS['initial'] = 'learner'
else:
    restore_checkpoint(RESTORE_ARCHIVE, INITIAL)
    STAGE_OWNERS['initial'] = RESTORED_CHECKPOINT_OWNER
published = json.loads((runtime.repo / 'experiments/q35-4b-s23.json').read_text())[0]
initial_config, initial_metrics = inspect_checkpoint(INITIAL, stage='initial', owner=STAGE_OWNERS['initial'], recipe=runtime.selected_initial_recipe(published))
""")
    code("""PREWORK_ARCHIVE = backup_checkpoint(INITIAL, LAB_DIR.parent / 'pacman-initial-checkpoint.zip')
try:
    from google.colab import files
except ImportError:
    print('Save initial checkpoint:', PREWORK_ARCHIVE)
else:
    files.download(str(PREWORK_ARCHIVE))
""")
    md("""## Prepare the intermediate-stage data (prework)

Verify dates/missing-evidence JSONL and the separate `documents-v1` train partition. Reconstruct skills/devtools by concatenating verified `hard-v1` train then `devtools-v1` train. Each input matches its published checksum and count. The course pins the combined checksum; the unavailable historical round-10 concatenation is not claimed byte-identical. Replay always samples `decision-v7` **train**. Development/test rows do not enter any training stage.""")
    code("""runtime.prepare_intermediate_data()
STAGES = specifications()
print({name: {'new_records': stage['records'], 'replay_records': stage['replay']} for name, stage in STAGES.items()})
""")
    md("""## Stage 2 — Dates and missing evidence (prework)

Warm-start from your **completed Stage 1** checkpoint. On a fresh runtime without a restored checkpoint, run Stage 1 to completion first; setup/model downloads do not create this trained checkpoint. Use 1,425 generated records (900 date-policy cases, 255 missing-fact cases and 270 intact controls) plus 2,000 replayed `decision-v7` training records. Published settings: one epoch, learning rate 2e-5, batch 4, accumulation 2, gradient checkpointing, BF16, seed 1 and 25% none minimal pairs. This is a separate run and a separate checkpoint. Its 180-minute attempt cap is a scheduling limit pending GPU measurements.""")
    code("""DATES = CHECKPOINT_ROOT / 'kev-4b-dates'
from lora_recovery import latest_snapshot
STAGES = specifications()
RESUME_DATES = latest_snapshot(Path(str(DATES) + '-recovery')) is not None
RESTORE_DATES_ARCHIVE = None
RESTORED_DATES_OWNER = 'learner'
if RESTORE_DATES_ARCHIVE is None:
    if (DATES / 'run-evidence.json').is_file():
        print('Using completed dates checkpoint:', DATES)
    else:
        print('Dates command:', runtime.intermediate_command('dates', DATES, INITIAL), flush=True)
        runtime.intermediate('dates', DATES, init_from=INITIAL, resume=RESUME_DATES)
    STAGE_OWNERS['dates'] = 'learner'
else:
    restore_checkpoint(RESTORE_DATES_ARCHIVE, DATES)
    STAGE_OWNERS['dates'] = RESTORED_DATES_OWNER
dates_config, dates_metrics = inspect_checkpoint(DATES, stage='dates', owner=STAGE_OWNERS['dates'], parent=INITIAL, recipe=runtime.selected_recipe(STAGES['dates']['args']), data_spec=STAGES['dates'])
""")
    code("""DATES_ARCHIVE = backup_checkpoint(DATES, LAB_DIR.parent / 'pacman-dates-checkpoint.zip')
try:
    from google.colab import files
except ImportError:
    print('Save dates checkpoint:', DATES_ARCHIVE)
else:
    files.download(str(DATES_ARCHIVE))
""")
    md('## Stage 3 — Documents (prework)\n\nWarm-start from **Stage 2**. If you restored a completed dates checkpoint, run runtime setup, optimized preparation, storage and intermediate-data preparation, then start here. You can skip the Stage 1/2 training cells; recovery files are not needed to initialize documents. Set `DATES` to the restored folder below. Earlier stage archives that are unavailable are reported as missing in the comparison/export.\n\nUse 5,219 consumer-finance complaint records plus 2,000 replayed decision-v7 training records. Published settings: one epoch, learning rate 2e-5, batch 2 × accumulation 4, BF16, FP32 stored weights, state budget 7,552, gradient checkpointing, seed 2 and 25% none minimal pairs. Expect 903 optimizer steps. The memory profile uses batch 1 × accumulation 8 with bounded row passes. Save this checkpoint before continuing; documents and skills are separate in Kev-4B.')
    code("""DOCUMENTS = CHECKPOINT_ROOT / 'kev-4b-documents'
from lora_recovery import latest_snapshot
STAGES = specifications()
DATES = CHECKPOINT_ROOT / 'kev-4b-dates'  # Set this to your completed dates folder
STAGE_OWNERS.setdefault('dates', 'learner')  # 'instructor' for a supplied checkpoint
RESUME_DOCUMENTS = latest_snapshot(Path(str(DOCUMENTS) + '-recovery')) is not None
RESTORE_DOCUMENTS_ARCHIVE = None
RESTORED_DOCUMENTS_OWNER = 'learner'
if RESTORE_DOCUMENTS_ARCHIVE is None:
    if (DOCUMENTS / 'run-evidence.json').is_file():
        print('Using completed documents checkpoint:', DOCUMENTS)
    else:
        print('Documents command:', runtime.intermediate_command('documents', DOCUMENTS, DATES), flush=True)
        runtime.intermediate('documents', DOCUMENTS, init_from=DATES, resume=RESUME_DOCUMENTS)
    STAGE_OWNERS['documents'] = 'learner'
else:
    restore_checkpoint(RESTORE_DOCUMENTS_ARCHIVE, DOCUMENTS)
    STAGE_OWNERS['documents'] = RESTORED_DOCUMENTS_OWNER
documents_config, documents_metrics = inspect_checkpoint(DOCUMENTS, stage='documents', owner=STAGE_OWNERS['documents'], parent=DATES, recipe=runtime.selected_recipe(STAGES['documents']['args']), data_spec=STAGES['documents'])
""")
    code("""DOCUMENTS_ARCHIVE = backup_checkpoint(DOCUMENTS, LAB_DIR.parent / 'pacman-documents-checkpoint.zip')
try:
    from google.colab import files
except ImportError:
    print('Save documents checkpoint:', DOCUMENTS_ARCHIVE)
else:
    files.download(str(DOCUMENTS_ARCHIVE))
""")
    md("""## Stage 4 — Skills and developer tools (prework)

Warm-start from **Stage 3 documents**. Use 6,000 generated skill records plus 5,320 developer-tooling records, with 4,000 replayed decision-v7 records. Published settings: one epoch, learning rate 2e-5, batch 2 × accumulation 4, BF16, FP32 stored weights, state budget 7,552, gradient checkpointing, seed 1 and 25% none minimal pairs. Expect 1,915 optimizer steps. The memory profile uses batch 1 × accumulation 8. This checkpoint is the class baseline. Complete all four general stages before class; their configurable 180-minute attempt caps are scheduling limits, not timing estimates.

A 4B model still uses hybrid DeltaNet/full attention and needs optimized kernels. Start fresh from the pinned 4B base; 0.8B adapters cannot initialize this curriculum. Published H100/H200 timings do not predict this GPU's duration.""")
    code("""SKILLS = CHECKPOINT_ROOT / 'kev-4b-skills'
from lora_recovery import latest_snapshot
STAGES = specifications()
RESUME_SKILLS = latest_snapshot(Path(str(SKILLS) + '-recovery')) is not None
RESTORE_SKILLS_ARCHIVE = None
RESTORED_SKILLS_OWNER = 'learner'
if RESTORE_SKILLS_ARCHIVE is None:
    if (SKILLS / 'run-evidence.json').is_file():
        print('Using completed skills checkpoint:', SKILLS)
    else:
        print('Skills and developer tools command:', runtime.intermediate_command('skills', SKILLS, DOCUMENTS), flush=True)
        runtime.intermediate('skills', SKILLS, init_from=DOCUMENTS, resume=RESUME_SKILLS)
    STAGE_OWNERS['skills'] = 'learner'
else:
    restore_checkpoint(RESTORE_SKILLS_ARCHIVE, SKILLS)
    STAGE_OWNERS['skills'] = RESTORED_SKILLS_OWNER
skills_config, skills_metrics = inspect_checkpoint(SKILLS, stage='skills', owner=STAGE_OWNERS['skills'], parent=DOCUMENTS, recipe=runtime.selected_recipe(STAGES['skills']['args']), data_spec=STAGES['skills'])
""")
    code("""SKILLS_ARCHIVE = backup_checkpoint(SKILLS, LAB_DIR.parent / 'pacman-skills-checkpoint.zip')
try:
    from google.colab import files
except ImportError:
    print('Save skills checkpoint:', SKILLS_ARCHIVE)
else:
    files.download(str(SKILLS_ARCHIVE))
""")
    md("""## Load completed Skills checkpoint — start the lab here

On a new runtime or after refreshing the bootstrap, run `runtime.setup()`, optimized preparation with `RUN_TRAINING_PREFLIGHT=False`, and the storage cell first. Bootstrap creates a new runtime helper; these cells establish its hardware, training and backup settings while reusing cached files. If using the automatic Drive backup, keep the complete `kev-4b-skills` backup folder (including `latest.json`, ZIPs and their `.zip.json` receipts) at `MyDrive/QPlusLearning/lab-01-kev-pacman/backups/kev-4b-skills`. With `SAVE_TO_DRIVE=True`, storage restores the completed native checkpoint to `/content/pacman-kev-lab/checkpoints/kev-4b-skills`.

**When Skills is already complete, skip the Stage 1–4 training and backup/export cells. Run the cell below, then the single CP1 exercise.** Try the Skills baseline in the first interactive play cell. A separate fine-tuned play cell follows Stage 5. This startup cell loads the completed Skills model, reads available saved stage metrics and starts inference. It does not resume Skills training or require the Documents checkpoint. Earlier stage weights and training curves are available only if you also restored them; their absent archives are reported in the submission.

Set Skills ownership to `instructor` below if the instructor supplied this checkpoint. Interactive play initially uses this general model; CP1 fine-tunes it on your reviewed Pac-Man labels.""")
    code("""# Load completed Skills checkpoint for the lab
from pathlib import Path
import json
CHECKPOINT_ROOT = LAB_DIR / 'checkpoints'
INITIAL = CHECKPOINT_ROOT / 'kev-4b-initial'
DATES = CHECKPOINT_ROOT / 'kev-4b-dates'
DOCUMENTS = CHECKPOINT_ROOT / 'kev-4b-documents'
SKILLS = CHECKPOINT_ROOT / 'kev-4b-skills'
GENERAL = SKILLS
required = ['head.pt', 'adapter_config.json', 'training_config.json', 'training_metrics.json', 'run-evidence.json']
missing = [name for name in required if not (GENERAL / name).is_file()]
if missing or not any(file.is_file() for file in GENERAL.glob('adapter_model.*')):
    raise RuntimeError(f'Restore the completed Skills checkpoint to {GENERAL}; missing files: {missing}, or adapter weights.')
STAGE_OWNERS = globals().get('STAGE_OWNERS', {})
STAGE_OWNERS.setdefault('skills', 'learner')  # 'instructor' for a supplied checkpoint
stage_checkpoints = {'initial': INITIAL, 'dates': DATES, 'documents': DOCUMENTS, 'skills': SKILLS}
stage_metrics, stage_configs = {}, {}
for stage, folder in stage_checkpoints.items():
    config_path, metrics_path = folder / 'training_config.json', folder / 'training_metrics.json'
    stage_configs[stage] = json.loads(config_path.read_text()) if config_path.is_file() else None
    stage_metrics[stage] = json.loads(metrics_path.read_text()) if metrics_path.is_file() else None
initial_config, dates_config, documents_config, skills_config = [stage_configs[name] for name in stage_checkpoints]
initial_metrics, dates_metrics, documents_metrics, skills_metrics = [stage_metrics[name] for name in stage_checkpoints]
skills_evidence = json.loads((GENERAL / 'run-evidence.json').read_text())
if (skills_evidence['stage'] != 'skills' or skills_metrics['optimizer_steps'] != 1915
        or skills_metrics['requested_records'] != 15320 or skills_config['args'].get('max_steps', 0)
        or skills_metrics.get('truncated_records', 0) or skills_metrics.get('rejected_records', 0)):
    raise ValueError('Complete the full Skills stage before starting the lab.')
missing_stage_archives = [stage for stage, folder in stage_checkpoints.items()
                         if not (folder / 'head.pt').is_file() or not list(folder.glob('adapter_model.*'))]
print('Earlier stage archives unavailable:', missing_stage_archives)
models = runtime.start(GENERAL)
print(json.dumps(models, indent=2))
""")
    md("""## Calibration is separate from training

Kev's release fitted one probability temperature after its four training stages. We do not copy that fitted value into freshly trained checkpoints. This notebook keeps their own raw probabilities. A workload calibration experiment needs suitable held-out labels and is outside the mandatory 30-minute Pac-Man fine-tuning block. It does not update LoRA/head weights or change the top-ranked action.""")
    md("""## Interactive play — general Skills baseline

Try **Human** mode with arrows/WASD; click the board for keyboard focus. Restart, select **Kev**, and watch its choices. Human mode runs at 60 simulation frames per second. Kev pauses the simulation while choosing a direction at each adjacent-tile entry boundary, then player and all four ghosts advance using upstream speeds/timers. Wall-clock survival is not a fair skill metric. Pause before running training cells.

This cell explicitly selects `kev-4b-skills`: general Skills training, no Pac-Man fine-tuning. After Stage 5, use **Interactive play — Pac-Man fine-tuned Kev** to try the task adapter on the same game. Pause the other board before switching models. Expand the **Active LoRA + pointer head** badge for paths and SHA-256 fingerprints. The badge and traces verify the serving model card; `kev-latest` alone is an API alias.

Colab supplies the notebook callback below. On Kaggle, use the Python evaluation and rollouts instead. API errors pause visibly; the game does not replace failed player decisions with a hidden rules controller. Pause play before training, and finish interactive play before exporting results.""")
    code("""# Interactive play with the general Skills checkpoint
from IPython.display import display, HTML, JSON
GENERAL_PLAY_CHECKPOINT = LAB_DIR / 'checkpoints/kev-4b-skills'
if runtime.active_checkpoint is None or runtime.active_model_info()['checkpoint'] != str(GENERAL_PLAY_CHECKPOINT):
    runtime.start(GENERAL_PLAY_CHECKPOINT)
bridge = NotebookBridge(LAB_DIR / 'results/player-trace.jsonl', runtime.active_model_info)
try:
    from google.colab import output
except ImportError:
    print('Kaggle/local: continue with the decision and rollout cells below.')
else:
    output.register_callback('pacman.decide', lambda state, selected_bridge=bridge: JSON(selected_bridge.decide(state)))
    output.register_callback('pacman.model', lambda selected_bridge=bridge: JSON(selected_bridge.model()))
    display(HTML(GAME))
""")
    md("""## CP1 — Fine-tune and evaluate Kev on Pac-Man game states

This is the only assessed checkpoint. Review demonstrations from a teacher that passed complete native-game qualification, adapt your Skills LoRA/head, and compare full games before and after training. Submit the adapter/head, reviewed labels, curves and `comparison.json`. Improvement must be measured; higher agreement with teacher labels alone is insufficient.

### Inspect qualified demonstrations (20–30 minutes)

The instructor first tests a fixed teacher on **20 complete native games: five separate qualification seeds at levels 1, 2, 3 and 5**. Every maze must clear. Avoidable immediate deaths, sustained loops (eight consecutive cycle detections), pellet stalls beyond 128 decisions and incorrect movement boundaries fail the gate. All brief repeats remain visible. The original zero-repeat suite is archived as failed; the user-selected sustained-loop criterion uses fresh seeds and an unchanged teacher. A source/configuration receipt and compact native replays are verified before labels can be loaded. Development and current/retired qualification seeds never enter training. This is finite-suite evidence, not a guarantee for every future state.

The teacher combines CS188-style cached shortest food routes with **native rollout MPC**: enumerate each legal first direction, simulate routing policies with safety buffers 1, 2, 4 and 6 for 240 native frames (480 near dangerous ghosts or with 30 or fewer pellets), and use two independently resampled future-RNG scenarios. Native targeting predicts Blinky, Pinky, Inky and Clyde; frightened turns retain uncertainty. Exact immediate safety comes first, then actual/imminent dry-cycle avoidance, approximate rollout survival and predicted time to the next pellet. The teacher is an independently implemented finite policy portfolio, not UCT/MCTS or a globally optimal solver. See the [algorithm comparison](https://github.com/yxc20089/QPlusLearning/blob/main/labs/lab-01-kev-pacman/evaluation/teacher-methods.md).

There are **4,096 training and 256 development labels** sampled from completed teacher games, including danger, power, late-maze and post-respawn decisions. Collection interleaves ordinary starts with recovery games: a declared legal-action prefix intentionally loses one native life, then the frozen teacher takes over and must finish. The prefix is excluded from labels. Every teacher continuation must pass the same per-game gates; full-game metrics still show the scripted death. Native collection replays verify both phases, and failed games are never discarded. Whole episode seeds and exact inputs are disjoint. The learner sees the current maze/power pellets, pixel offsets, movement phases, power countdown, ghost/release/phase clocks, destination visits and recent history. Labels and search information stay outside the model inputs. One action ends when entering the requested adjacent tile using whole native frames; browser play and headless evaluation use this same v2 contract. Old v1 checkpoints/data remain available for diagnosis, but new training uses `pacman-native-v2`.

Inspect three training boards and enter reviewed legal labels in `EDITS`. Keep edits inside training and record them as learner annotations. Fix the candidate before running the reserved gameplay benchmark.""")
    code("""manifest = prepare_dataset(LAB_DIR / 'data')  # Fails closed if the teacher/receipt/data changed
qualification_receipt = json.loads((LAB_DIR / 'evaluation/teacher-qualification.json').read_text())
print('Teacher qualification:', qualification_receipt['qualification'])
training_file = LAB_DIR / 'data/pacman-native-v2-train-reviewed.jsonl'
label_source = training_file if training_file.is_file() else LAB_DIR / 'data/pacman-native-v2-train.jsonl'
training = [json.loads(line) for line in label_source.read_text().splitlines()]
for row in training[:3]:
    print(row['_meta']['id'], json.dumps(row['state'], indent=2))
    print('Legal:', list(row['questions']['move']['criteria']))
EDITS = {}  # Example after inspecting a board: {'train-board-0000': 'left'}
assert set(EDITS).issubset({r['_meta']['id'] for r in training}), 'Unknown training board ID'
for row in training:
    if row['_meta']['id'] in EDITS:
        label = EDITS[row['_meta']['id']]
        assert label in row['questions']['move']['criteria'], 'Choose a legal move'
        row['questions']['move'].update(label=label, src='learner_annotation')
        row['_meta']['label_source'] = 'learner annotation'
if EDITS or not training_file.is_file():
    training_file.write_text(''.join(json.dumps(row) + '\\n' for row in training))
print('Labels:', [(r['_meta']['id'], r['questions']['move']['label']) for r in training[:3]])
print('Search alternatives:', training[0]['_meta']['teacher']['candidates'])
print('Equally ranked directions:', training[0]['_meta']['equally_ranked_actions'])
print('Counts:', manifest['counts'], 'coverage:', manifest['coverage'])
""")
    md("""### Stage 5 — Fine-tune Pac-Man decisions (30–60 minutes)

30-minute block: 5 minutes inspect a labelled request and the loss; target up to 20 minutes training; 5 minutes inspect and save the checkpoint. This is supervised imitation. Kev updates the same all-module rank-16 LoRA adapters and 256-dimensional pointer head as the preceding stages; original base matrices stay frozen. Planning happens only when producing labels, not inside the training loss or the model controller.

Warm-start from **your Stage 4 Skills checkpoint**. Following the intermediate-stage pattern, train **one complete epoch**, learning rate **2e-5**, with **2,000 decision-v7 training examples mixed as replay**. There are **6,096 requests / 762 optimizer updates**, versus the old 64-board exercise's 16 updates. Dates used 3,425 requests / 429 updates; Documents 7,219 / 903; Skills 15,320 / 1,915. More labels provide a substantive adaptation experiment; improved play must still be measured.

For the target 96 GB GPU this stage explicitly uses **batch 4 × accumulation 2, row budget 0**, BF16 autocast, FP32 stored weights, optimized FLA/conv kernels, fused AdamW and gradient checkpointing. The effective batch stays eight. The printed command is authoritative. None/distractor and none-pair augmentation are disabled for this mixed task stage, because the Pac-Man output must remain a legal move. Generic replay retains its recorded labels; option shuffling stays active. These task/execution choices depart from the generic augmentation settings.

The helper reads architecture from the checkpoint and stops inference to free GPU memory. The state budget is 4,096 tokens; truncation/rejected records fail the audit. Existing recovery and Drive backup apply. The full stage has no short-step cap and may exceed the classroom block: at 1.5 seconds/update, compute alone is 19 minutes; at 3 seconds/update, 38 minutes. Measure steady step time before class and complete a slower full run as prework. CPU checks do not establish GPU timing, memory fit or trained-model improvement.""")
    code("""print(json.dumps(training[0], indent=2))
CHECKPOINT = CHECKPOINT_ROOT / 'kev-4b-pacman-native-v2'
from lora_recovery import latest_snapshot
RESUME_PACMAN = latest_snapshot(Path(str(CHECKPOINT) + '-recovery')) is not None
print('Pac-Man recipe:', RECIPE)
print('Selected Pac-Man command:', runtime.finetuning_command(training_file, CHECKPOINT, GENERAL))
if (CHECKPOINT / 'run-evidence.json').is_file():
    evidence = json.loads((CHECKPOINT / 'run-evidence.json').read_text())
    assert evidence['training_data_sha256'] == hashlib.sha256(training_file.read_bytes()).hexdigest(), 'Saved checkpoint used different labels; select a new checkpoint output to retrain.'
    checkpoint = CHECKPOINT
    print('Using completed Pac-Man checkpoint:', checkpoint)
else:
    checkpoint = runtime.finetune(training_file, CHECKPOINT, init_from=GENERAL, steps=0, resume=RESUME_PACMAN)
metrics = json.loads((checkpoint / 'training_metrics.json').read_text())
print(metrics)
assert len(training) == manifest['counts']['train'], 'Preserve the complete training partition'
saved_recipe = json.loads((checkpoint / 'training_config.json').read_text())['args']
for key, value in RECIPE.items():
    assert saved_recipe[key] == (float(value) if key == 'lr' else value), f'Checkpoint used another recipe: {key}'
assert metrics['optimizer_steps'] == manifest['expected_optimizer_steps'], 'Complete the full planner/replay stage'
assert not metrics.get('truncated_records', 0) and not metrics.get('rejected_records', 0), 'No state truncation or dropped records'
assert metrics['records_seen'] == metrics['requested_records'] == manifest['expected_training_requests'], 'Complete one epoch including replay'
""")
    md("""## Interactive play — Pac-Man fine-tuned Kev

Run the cell below after Stage 5 completes, or after the storage cell restores your completed `kev-4b-pacman-native-v2` checkpoint from Drive. It loads that adapter and pointer head explicitly; you can play before running the benchmark. This cell needs only the initialized runtime, game helpers and completed checkpoint. It does not require earlier training or evaluation variables.

Pause the baseline board first. On the new board, select **Kev** and press **Start**. Check that **Active LoRA + pointer head** says **`kev-4b-pacman-native-v2` / Pac-Man fine-tuned**. The classic maze, assets, four native ghosts, power pellets, lives and levels are the same as in baseline play. **New game** resets the board using the same seed (7). You can pause, restart or switch to Human mode. The badge shows the verified adapter and each model decision records it in `results/fine-tuned-player-trace.jsonl`.

This is an ungraded activity. Pause the game before benchmarking, switching models or exporting results. Run only one board at a time: both cells connect to the same notebook-local inference server. Colab supplies the callbacks; Kaggle/local users can use the Python rollouts.""")
    code("""# Interactive play with the completed Pac-Man adapter
from IPython.display import display, HTML, JSON
PACMAN_PLAY_CHECKPOINT = CHECKPOINT_ROOT / 'kev-4b-pacman-native-v2'
# To inspect your existing v1 adapter, explicitly select CHECKPOINT_ROOT / 'kev-4b-pacman-planner-v1' instead.
if not (PACMAN_PLAY_CHECKPOINT / 'run-evidence.json').is_file():
    raise RuntimeError(f'Complete Stage 5 or restore its completed checkpoint to {PACMAN_PLAY_CHECKPOINT} before playing.')
if runtime.active_checkpoint is None or runtime.active_model_info()['checkpoint'] != str(PACMAN_PLAY_CHECKPOINT):
    runtime.start(PACMAN_PLAY_CHECKPOINT)
if not runtime.active_model_info()['pacman_fine_tuned']:
    raise RuntimeError('The active checkpoint is not recorded as Pac-Man fine-tuned. Load the completed Stage 5 checkpoint.')
fine_tuned_bridge = NotebookBridge(LAB_DIR / 'results/fine-tuned-player-trace.jsonl', runtime.active_model_info)
try:
    from google.colab import output
except ImportError:
    print('Kaggle/local: use the Python rollouts below; interactive callbacks require Colab.')
else:
    output.register_callback('pacman.decide', lambda state, selected_bridge=fine_tuned_bridge: JSON(selected_bridge.decide(state)))
    output.register_callback('pacman.model', lambda selected_bridge=fine_tuned_bridge: JSON(selected_bridge.model()))
    display(HTML(GAME))
""")
    md("""### Evaluate actual games before and after training (60–80 minutes)

Pause interactive play. Both adapters run **20 full native games** from the same initial boards: five reserved seeds at starting levels **1, 2, 3 and 5**. Each game continues after a lost life, until the first maze clears or native game-over ends all lives. Initial ghosts, release rules, bonus lives, power, fruit and collisions are native. Every episode starts in a fresh game closure. No graphics, random isolated states, teacher-agreement score or hidden replacement controller is involved.

Report clears/game-overs, remaining pellets, avoidable immediate deaths, every raw cycle, longest consecutive cycle streak, pellet stalls, power pellets, ghosts eaten, post-respawn progress and action geometry. Compare paired per-seed results; higher score alone is insufficient. Large watchdogs (10,000 decisions, 600 simulated seconds or 512 consecutive dry decisions) stop hung agents. These exits are incomplete failures, never wins or native game-overs. Full games may exceed the classroom block; time serving and complete a slower benchmark as prework. The published CPU teacher results do not establish learned-Kev performance.

This v2 benchmark changes action boundaries and exposes more current state. Old v1 benchmark scores are not directly comparable. You may explicitly evaluate an existing v1 adapter using this protocol, but it was trained with the older schema. A new v2 adapter requires the newly qualified demonstrations and a fresh Pac-Man output directory.""")
    code("""# This comparison can also evaluate your existing completed v1 checkpoint explicitly.
GENERAL = CHECKPOINT_ROOT / 'kev-4b-skills'
checkpoint = CHECKPOINT_ROOT / 'kev-4b-pacman-native-v2'
# checkpoint = CHECKPOINT_ROOT / 'kev-4b-pacman-planner-v1'  # Existing adapter, older training schema
runtime.start(GENERAL)
before_identity = runtime.active_model_info()
before_gameplay = benchmark_gameplay(model_info=runtime.active_model_info,
                                    trace_dir=LAB_DIR / 'results/gameplay-general')
runtime.start(checkpoint)
after_identity = runtime.active_model_info()
after_gameplay = benchmark_gameplay(model_info=runtime.active_model_info,
                                   trace_dir=LAB_DIR / 'results/gameplay-fine-tuned')
gameplay_comparison = paired_gameplay(before_gameplay, after_gameplay)
comparison = {'schema_version': 3, 'baseline_checkpoint': str(GENERAL),
    'fine_tuned_checkpoint': str(checkpoint), 'active_adapters': {'before': before_identity, 'after': after_identity},
    'gameplay': {'general': before_gameplay, 'fine_tuned': after_gameplay, 'paired': gameplay_comparison},
    'runtime': runtime.gpu, 'pacman_dataset': manifest,
    'teacher_qualification_sha256': hashlib.sha256((LAB_DIR / 'evaluation/teacher-qualification.json').read_bytes()).hexdigest(),
    'checkpoint_owners': STAGE_OWNERS}
(LAB_DIR / 'comparison.json').write_text(json.dumps(comparison, indent=2))
for name, games in [('general', before_gameplay), ('fine_tuned', after_gameplay)]:
    print(name, 'means:', games['means'], 'clears:', games['level_clears'], 'incomplete:', games['capped_episodes'])
print('Paired native gameplay:', json.dumps(gameplay_comparison, indent=2))
""")
    md("""The comparison leaves the chosen task adapter serving. Rerun **Interactive play — Pac-Man fine-tuned Kev**, select the same checkpoint, choose **Kev** and start a new game. Verify its path/hash in the badge. Both gameplay reports use the same v2 observation/action schema and native engine. Results apply to the listed levels/seeds in the same maze. Counterfactual actions only diagnose immediate avoidable deaths; they never replace Kev's choices.

### Explain and export the checkpoint evidence (80–90 minutes)

Explain one changed move or remaining mistake. Relate a planning label to the board and ghost personalities. Identify the LoRA/head parameters that trained and the limits of finite-horizon native rollout search.

Submit the executed notebook, reviewed training JSONL, `comparison.json`, all available small adapter/head checkpoints and stage configurations/metrics, training logs and TensorBoard events and `runtime-preflight.json`. A full fresh run produces five checkpoints. If you continued from a completed intermediate checkpoint without older archives, record those missing stages; the imported checkpoint retains its recorded parent provenance, but absent parent weights cannot be rechecked or exported. Record the Colab compute units consumed and elapsed GPU time from your session. Save outputs before the temporary runtime disconnects, then stop the server. On Colab, run the download cell.""")
    code("""runtime.stop()
archive = str(LAB_DIR.parent / 'pacman-lab-submission.zip')
print('Unavailable earlier stage archives:', missing_stage_archives)
# Export small adapters/heads and results, without foundation weights or packages.
import zipfile
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as out:
    for file in [LAB_DIR/'comparison.json', LAB_DIR/'benchmark-spec.json', training_file, LAB_DIR/'data/pacman-native-v2-manifest.json', LAB_DIR/'data/pacman-native-v2-replays.zip', LAB_DIR/'evaluation/teacher-qualification.json', LAB_DIR/'evaluation/teacher-qualification-replays.zip', LAB_DIR/'runtime-preflight.json', LAB_DIR/'optimized-training-preflight.json', LAB_DIR/'trainable-parameters.json']:
        out.write(file, file.relative_to(LAB_DIR))
    for folder in [INITIAL, DATES, DOCUMENTS, SKILLS, checkpoint]:
        for file in folder.rglob('*'):
            if file.is_file():
                out.write(file, Path('checkpoints') / folder.name / file.relative_to(folder))
    for file in LOG_DIR.rglob('*'):
        if file.is_file():
            out.write(file, file.relative_to(LAB_DIR))
    for file in (LAB_DIR/'results').rglob('*'):
        if file.is_file():
            out.write(file, file.relative_to(LAB_DIR))
try:
    from google.colab import files
except ImportError:
    print('Download:', archive)
else:
    files.download(archive)
""")
    result = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}, "colab": {"name": "pacman_kev_lab.ipynb"}}, "nbformat": 4, "nbformat_minor": 5}
    (ROOT / 'notebooks/pacman_kev_lab.ipynb').write_text(json.dumps(result, indent=2) + '\n')



def game():
    from pacman_lab import notebook_game
    html = notebook_game(action_version=2)
    (ROOT / 'games/pacman.html').write_text(html)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pin-source', action='store_true', help='pin committed helpers before rebuilding')
    args = parser.parse_args()
    if args.pin_source:
        source_lock(pin=True)
    notebook()
    game()
    print('Built Lab 1 notebook and Pac-Man browser game.')
