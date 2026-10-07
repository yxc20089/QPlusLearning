"""Colab RTX PRO 6000 Blackwell helpers for an interactive Kev notebook.

Setup runs before the 90-minute session. GPU training duration needs a preflight.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from urllib.error import URLError
from urllib.request import urlopen

from training_monitor import stream_training
from training_stages import checkpoint_fingerprint, specifications, verify_data
from lora_recovery import latest_snapshot
from optimized_training import install_kernels, file_hash
from checkpoint_backup import save_backup, restore_backup

CODE_REVISION = "84847f0a883d900f7de5b7a57eaa341ca7f9a6b4"
MODEL_RUN = "jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52"
TARGET_GPU = "RTX PRO 6000 Blackwell"
BASE_MODEL = "Qwen/Qwen3.5-4B-Base"
BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
TRAINING_SUITE = "evals/v7/decision-v7"
MEMORY_OVERRIDES = {"batch": 1, "accum": 8, "checkpointing": 1, "row_budget": 2048}


def validate_initial_execution(execution):
    if execution is None:
        return None
    if set(execution) != {"batch", "accum", "row_budget"}:
        raise ValueError("Initial execution needs exactly batch, accum and row_budget")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in execution.values()):
        raise ValueError("Initial execution settings must be integers")
    if execution["batch"] < 1 or execution["accum"] < 1 or execution["row_budget"] < 0:
        raise ValueError("Positive batch/accum and nonnegative row_budget required")
    if execution["batch"] * execution["accum"] != 8:
        raise ValueError("The initial execution must preserve effective batch 8")
    return dict(execution)


GPU_PREFLIGHT = """
import importlib.util, json, torch
if not torch.cuda.is_available():
    raise RuntimeError('Select a GPU in Runtime > Change runtime type, then reconnect.')
props = torch.cuda.get_device_properties(0)
bf16 = torch.cuda.is_bf16_supported()
# Exercise CUDA kernels and a backward pass in the installed Kev environment.
dtype = torch.bfloat16 if bf16 else torch.float32
x = torch.randn(32, 32, device='cuda', dtype=dtype, requires_grad=True)
(x @ x.T).float().square().mean().backward()
torch.cuda.synchronize()
print(json.dumps({'name': props.name, 'memory_gib': props.total_memory / 2**30,
                  'compute_capability': [props.major, props.minor],
                  'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'bf16_supported': bf16, 'kernel_backward': 'passed',
                  'optional_packages_before_overlay': {name: importlib.util.find_spec(name) is not None
                                                for name in ['causal_conv1d', 'fla']}}))
"""


def validate_gpu(info, allow_other_gpu=False):
    """Select precision only after checking the classroom hardware/software contract."""
    target = TARGET_GPU.lower() in info["name"].lower()
    if not target and not allow_other_gpu:
        raise RuntimeError(f"This lab targets {TARGET_GPU}; received {info['name']}. "
                           "Select the target GPU if available, or use the instructor's validated fallback.")
    # The pinned PyTorch 2.8 CUDA 12.8 wheel does not support Pascal/P100.
    if tuple(info["compute_capability"]) < (7, 0):
        raise RuntimeError("This environment does not support P100/Pascal GPUs. Use the target GPU or a validated BF16-capable fallback.")
    if not info["torch"].startswith("2.8.0") or tuple(map(int, info["cuda"].split(".")[:2])) < (12, 8):
        raise RuntimeError("The lab requires its pinned PyTorch 2.8.0 / CUDA 12.8 environment. Re-run setup.")
    if info["kernel_backward"] != "passed":
        raise RuntimeError("CUDA forward/backward preflight did not pass.")
    if not info["bf16_supported"]:
        raise RuntimeError("Optimized training must support BF16. Check the installed CUDA environment.")
    return "bf16"


class CloudRuntime:
    def __init__(self, workspace, allow_other_gpu=False, training_profile="memory_safe", initial_execution=None):
        if training_profile not in {"memory_safe", "published_reference"}:
            raise ValueError("Choose memory_safe or published_reference training_profile")
        self.workspace = Path(workspace).resolve()
        self.repo = self.workspace / "kev"
        self.python = self.repo / ".venv" / "bin" / "python"
        self.process = None
        self.log = None
        self.allow_other_gpu = allow_other_gpu
        self.gpu = None
        self.dtype = "bf16"
        self.training_profile = training_profile
        self.training_preflight = None
        self.initial_execution = validate_initial_execution(initial_execution)
        self.backup_root = None
        self.active_checkpoint = None

    def _backup_checkpoint(self, output, snapshot):
        config_path = Path(output) / 'training_config.json'
        data = json.loads(config_path.read_text()).get('args', {}).get('data') if config_path.is_file() else None
        if data and not Path(data).is_absolute():
            data = self.repo / data
        return save_backup(output, snapshot, self.backup_root, self.workspace, training_data=data)

    def configure_backup(self, backup_root, checkpoint_root):
        """Enable automatic archives, restore absent local outputs, back up existing ones."""
        self.backup_root = Path(backup_root).resolve() if backup_root is not None else None
        result = {'enabled': self.backup_root is not None, 'restored': [], 'backed_up': []}
        if self.backup_root is None:
            return result
        self.backup_root.mkdir(parents=True, exist_ok=True)
        result['backup_root'] = str(self.backup_root)
        for name in ('kev-4b-initial', 'kev-4b-dates', 'kev-4b-documents', 'kev-4b-skills', 'kev-4b-pacman-arcade', 'kev-4b-pacman-planner-v1', 'kev-4b-pacman-native-v2', 'kev-4b-pacman-native-v3', 'kev-4b-pacman-native-v4'):
            output = Path(checkpoint_root).resolve() / name
            restored = restore_backup(output, self.backup_root, self.workspace)
            if restored:
                result['restored'].append({'output': str(output), 'step': restored['step']})
                continue
            snapshot = latest_snapshot(Path(str(output) + '-recovery'))
            complete = (output / 'run-evidence.json').is_file()
            if snapshot is not None or complete:
                pointer = self.backup_root / name / 'latest.json'
                previous = json.loads(pointer.read_text()) if pointer.is_file() else {}
                if (previous.get('snapshot') == (snapshot.name if snapshot else None)
                        and previous.get('output_path') == str(output)
                        and previous.get('completed_stage') == complete):
                    continue
                receipt = self._backup_checkpoint(output, snapshot)
                result['backed_up'].append({'output': str(output), 'step': receipt['step']})
        return result

    def selected_recipe(self, recipe):
        return {**recipe, **(MEMORY_OVERRIDES if self.training_profile == "memory_safe" else {})}

    def selected_initial_recipe(self, recipe):
        return {**self.selected_recipe(recipe), **(validate_initial_execution(self.initial_execution) or {})}

    def _profile(self, command):
        if self.training_profile == "memory_safe":
            for key, value in MEMORY_OVERRIDES.items():
                flag = "--" + key
                if flag in command:
                    command[command.index(flag) + 1] = str(value)
                else:
                    command += [flag, str(value)]
        return command

    def setup(self):
        self.workspace.mkdir(parents=True, exist_ok=True)
        from pacman_lab import ensure_node
        ensure_node(self.workspace)
        subprocess.run(["nvidia-smi"], check=True)
        if not shutil.which("uv"):
            subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
        if not self.repo.exists():
            subprocess.run(["git", "clone", "https://github.com/jaredpalmer/kev.git", str(self.repo)], check=True)
            subprocess.run(["git", "checkout", CODE_REVISION], cwd=self.repo, check=True)
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.repo, text=True).strip()
        if actual != CODE_REVISION:
            raise RuntimeError("Existing Kev checkout differs from the lab pin. Use a fresh workspace.")
        subprocess.run(["uv", "sync", "--frozen", "--no-dev", "--extra", "serve", "--python", "3.13"], cwd=self.repo, check=True)
        self.gpu = json.loads(subprocess.check_output([str(self.python), "-c", GPU_PREFLIGHT], cwd=self.repo, text=True))
        self.dtype = validate_gpu(self.gpu, self.allow_other_gpu)
        install_kernels(self.python, self.repo, self.workspace)
        (self.workspace / "runtime-preflight.json").write_text(json.dumps(self.gpu, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(self.gpu, indent=2))
        print(f"Prepared pinned Kev environment using {self.dtype}. Download weights before class.")

    def environment(self):
        env = dict(os.environ)
        # BF16 on the target or another explicitly validated compatible GPU.
        env.update(KEV_DTYPE=self.dtype, KEV_BACKEND="torch", KEV_FUSED="0", KEV_CUDA_GRAPHS="0", PYTHONUNBUFFERED="1",
                   USE_HUB_KERNELS="0", LAB_REQUIRE_OPTIMIZED_KERNELS="1", LAB_FUSED_ADAMW="1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # Recovery settings belong to one launch, never inherit another run's.
        for name in ("LAB_RECOVERY_ROOT", "LAB_RESUME_FROM", "LAB_SAVE_STEPS", "LAB_SAVE_SECONDS", "LAB_ALLOW_EXECUTION_CHANGE",
                     "LAB_BACKUP_ROOT", "LAB_BACKUP_WORKSPACE"):
            env.pop(name, None)
        env.pop("KEV_API_KEY", None)  # this server binds only to notebook-local loopback
        return env

    def prepare_training(self, run_training_preflight=False):
        """Fetch verified training data/base weights and inspect exactly what can train."""
        self.training_preflight = None
        audit_path = self.workspace / "trainable-parameters.json"
        self.stop()
        # A helper refresh in the same connected runtime need not reinstall the
        # environment. Keep the GPU receipt from its completed setup.
        gpu_receipt = self.workspace / "runtime-preflight.json"
        if self.gpu is None and gpu_receipt.is_file():
            self.gpu = json.loads(gpu_receipt.read_text())
            self.dtype = validate_gpu(self.gpu, self.allow_other_gpu)
        code = """
import json, sys
from kev.suite import load_split
from kev.model import DecisionModel, load_tokenizer
rows = load_split(sys.argv[1], 'train')
tok = load_tokenizer(sys.argv[2], revision=sys.argv[3])
model = DecisionModel(sys.argv[2], tok, 'cpu', lora=16, revision=sys.argv[3], lora_targets='all')
trainable = {n:p.numel() for n,p in model.named_parameters() if p.requires_grad}
assert trainable and any(n.startswith('head.') for n in trainable)
assert all(n.startswith('head.') or 'lora_A' in n or 'lora_B' in n for n in trainable)
assert any('lora_A' in n for n in trainable)
report = {'base':sys.argv[2], 'base_revision':sys.argv[3], 'suite_records':len(rows),
          'trainable_parameters':trainable, 'original_base_matrices_frozen':True,
          'effective_encoder_features_fixed':False, 'lora_rank':16, 'lora_alpha':32}
with open(sys.argv[4], 'w') as out: json.dump(report, out, indent=2)
print('Prepared', len(rows), 'verified training records and base weights. LoRA/head audit saved.')
"""
        subprocess.run([str(self.python), "-c", code, TRAINING_SUITE, BASE_MODEL, BASE_REVISION, str(audit_path)],
                       cwd=self.repo, env=self.environment(), check=True)
        report = self.workspace / "optimized-training-preflight.json"
        log_dir = self.workspace / "logs" / "kernel-preflight" / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
        command = [str(self.python), "-u", str(Path(__file__).with_name("optimized_training.py")),
                   "--base", BASE_MODEL, "--revision", BASE_REVISION,
                   "--suite", TRAINING_SUITE, "--report", str(report)]
        if run_training_preflight:
            print("Running optional two-record Kev loss/backward check (no CUDA profiler)", flush=True)
        else:
            command.append("--bindings_only")
            print("Checking optimized bindings; skipping separate two-record training preflight", flush=True)
        stream_training(command,
                        cwd=self.repo, env=self.environment(), log_dir=log_dir,
                        stage="kernel-preflight", timeout_seconds=1200 if run_training_preflight else 120, require_metrics=False)
        self.training_preflight = json.loads(report.read_text())
        return json.loads(audit_path.read_text())

    def pretraining_command(self, output, steps=0):
        if steps < 0:
            raise ValueError("steps must be nonnegative; 0 runs the complete two-epoch recipe")
        recipe = json.loads((self.repo / "experiments/q35-4b-s23.json").read_text())[0]
        assert recipe["base"] == BASE_MODEL and recipe["base_revision"] == BASE_REVISION
        if self.dtype != "bf16":
            raise RuntimeError("The published initial recipe requires a BF16-capable GPU.")
        command = [str(self.python), "-m", "kev.train", "--suite", TRAINING_SUITE,
                   "--out", str(Path(output).resolve()), "--device", "cuda",
                   "--lora", "16", "--lora_targets", "all", "--head_dim", "256",
                   "--weights_dtype", "fp32"]
        for key, value in recipe.items():
            command += ["--" + key, str(value)]
        if steps:
            command += ["--max_steps", str(steps)]
        command = self._profile(command)
        for key, value in (validate_initial_execution(self.initial_execution) or {}).items():
            flag = "--" + key
            if flag in command:
                command[command.index(flag) + 1] = str(value)
            else:
                command += [flag, str(value)]
        return command

    def pretrain(self, output, steps=0, timeout_minutes=180, resume=False, allow_execution_change=False, resume_from=None):
        """Published decision-v7 base stage, without --init_from. Full run is prework."""
        return self._train(self.pretraining_command(output, steps), output, timeout_minutes, "initial", resume=resume,
                           allow_execution_change=allow_execution_change, resume_from=resume_from)

    def prepare_intermediate_data(self):
        code = "import sys; sys.path.insert(0, sys.argv[1]); from training_stages import prepare_data; prepare_data(sys.argv[2])"
        subprocess.run([str(self.python), "-c", code, str(Path(__file__).parent), str(self.repo)],
                       cwd=self.repo, env=self.environment(), check=True)

    def intermediate_command(self, stage, output, init_from):
        if self.dtype != "bf16":
            raise RuntimeError("Published intermediate stages require a BF16-capable GPU.")
        spec = specifications()[stage]
        command = self.training_command(self.repo / spec["data"], output, init_from=init_from)
        recipe = spec["args"]
        for key, value in recipe.items():
            flag = "--" + key
            if flag in command:
                command[command.index(flag) + 1] = str(value)
            else:
                command += [flag, str(value)]
        # Restore the generic dates-stage defaults, rather than inheriting the
        # task-specific Pac-Man augmentation/length/checkpointing settings.
        if stage == "dates":
            for key, value in {"checkpointing": 1, "max_state": 384,
                               "p_none": 0.1, "p_none_distract": 0.12, "p_distract": 0.15}.items():
                command[command.index("--" + key) + 1] = str(value)
        command += ["--suite", TRAINING_SUITE]
        return self._profile(command)

    def intermediate(self, stage, output, init_from, timeout_minutes=180, resume=False):
        command = self.intermediate_command(stage, output, init_from)
        spec = specifications()[stage]
        data = self.repo / spec['data']
        if not data.is_file():
            print(f'Preparing missing {stage} training data before loading the GPU model: {data}', flush=True)
            self.prepare_intermediate_data()
        verify_data(data, spec)
        return self._train(command, output, timeout_minutes, stage, resume=resume)

    def start(self, run=MODEL_RUN):
        self.stop()
        source = Path(run)
        source = source if source.is_absolute() else self.repo / source
        if (Path(run).is_absolute() or source.exists()) and not (source / 'head.pt').is_file():
            raise RuntimeError(f'Checkpoint has no pointer head: {source}. Load a completed checkpoint first.')
        self.log = open(self.workspace / "server.log", "w", encoding="utf-8")
        self.process = subprocess.Popen([str(self.python), "-m", "kev.serve", "--run", str(run), "--host", "127.0.0.1", "--port", "8009"], cwd=self.repo, env=self.environment(), stdout=self.log, stderr=subprocess.STDOUT)
        os.environ["KEV_BASE_URL"] = "http://127.0.0.1:8009"
        os.environ.pop("KEV_API_KEY", None)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.stop()
                raise RuntimeError("Kev stopped while loading. Inspect server.log in the notebook workspace.")
            try:
                with urlopen("http://127.0.0.1:8009/v1/models", timeout=3) as response:
                    models = json.load(response)
                card = self._serving_card(models)
                if card['run'] != str(run):
                    self.stop()
                    raise RuntimeError(f"Kev loaded {card['run']} instead of requested checkpoint {run}.")
                config = json.loads((source / 'training_config.json').read_text()) if (source / 'training_config.json').is_file() else {}
                metrics = json.loads((source / 'training_metrics.json').read_text()) if (source / 'training_metrics.json').is_file() else {}
                evidence = json.loads((source / 'run-evidence.json').read_text()) if (source / 'run-evidence.json').is_file() else {}
                self.active_checkpoint = {
                    'checkpoint': card['run'], 'checkpoint_name': source.name if source.is_dir() else str(run),
                    'stage': evidence.get('stage', 'released' if not source.is_dir() else 'unrecorded'),
                    'optimizer_steps': metrics.get('optimizer_steps'),
                    'checkpoint_sha256': checkpoint_fingerprint(source) if source.is_dir() else None,
                    'adapter_sha256': file_hash(source / 'adapter_model.safetensors') if (source / 'adapter_model.safetensors').is_file() else None,
                    'base': card['base'], 'base_revision': config.get('args', {}).get('base_revision'),
                    'lora_rank': card['lora'], 'dtype': card['dtype'], 'temperature': card['temperature'],
                    'api_alias': card['name'], 'pacman_fine_tuned': evidence.get('stage') == 'pacman'}
                (self.workspace / 'active-checkpoint.json').write_text(json.dumps(self.active_checkpoint, indent=2) + '\n')
                (self.workspace / "models.json").write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
                print("Kev ready at notebook-local localhost:8009")
                print('Active LoRA/head:', json.dumps(self.active_checkpoint, indent=2))
                return models
            except (URLError, TimeoutError):
                time.sleep(3)
            except (KeyError, ValueError, RuntimeError):
                self.stop()
                raise
        self.stop()
        raise TimeoutError("Model startup exceeded 15 minutes. Inspect server.log and connection status.")

    @staticmethod
    def _serving_card(models):
        cards = models.get('models', [])
        if not cards:
            raise RuntimeError('Kev returned no serving model card; checkpoint identity is unverified.')
        return next((card for card in cards if card.get('name') == 'kev-latest'), cards[0])

    def active_model_info(self):
        """Check the live serving run, rather than labelling an API alias as an adapter."""
        if self.active_checkpoint is None or self.process is None or self.process.poll() is not None:
            raise RuntimeError('No verified active checkpoint. Load the Skills or Pac-Man checkpoint first.')
        with urlopen('http://127.0.0.1:8009/v1/models', timeout=3) as response:
            card = self._serving_card(json.load(response))
        if (card['run'] != self.active_checkpoint['checkpoint'] or card['base'] != self.active_checkpoint['base']
                or card['lora'] != self.active_checkpoint['lora_rank']):
            raise RuntimeError('The serving checkpoint changed outside this runtime. Reload it before playing.')
        return dict(self.active_checkpoint)

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None
        self.active_checkpoint = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def training_command(self, training_data, output, init_from, steps=0, max_state=4096):
        # Read architecture from the checkpoint rather than guessing compatible flags.
        source = Path(init_from)
        source = source if source.is_absolute() else self.repo / source
        missing = [name for name in ('head.pt', 'adapter_config.json') if not (source / name).is_file()]
        if not any(file.is_file() for file in source.glob('adapter_model.*')):
            missing.append('adapter_model.*')
        if missing:
            stage = {'kev-4b-initial': 'Stage 1 (initial decision training)',
                     'kev-4b-dates': 'Stage 2 (dates)',
                     'kev-4b-documents': 'Stage 3 (documents)',
                     'kev-4b-skills': 'Stage 4 (skills)'}.get(source.name, 'the preceding training stage')
            message = (f'Starting checkpoint is missing or incomplete: {source}. Missing: {", ".join(missing)}. '
                       f'Run {stage} to completion or restore its completed checkpoint before continuing. '
                       'Setup and model downloads do not create a trained checkpoint.')
            snapshot = latest_snapshot(Path(str(source) + '-recovery'))
            if snapshot is not None:
                receipt = json.loads((snapshot / 'complete.json').read_text())
                message += (f' Recovery is available at optimizer step {receipt["step"]}: {snapshot}. '
                            'Rerun that stage\'s training cell to resume it.')
            raise RuntimeError(message)
        code = "import json,sys; from kev.checkpoint import Checkpoint; m=Checkpoint(sys.argv[1]).meta; print(json.dumps({k:getattr(m,k) for k in ['base','base_revision','lora','head_dim','option_isolation','special_embeddings','weights_dtype','weights']}))"
        try:
            result = subprocess.check_output([str(self.python), "-c", code, str(init_from)],
                                             cwd=self.repo, env=self.environment(), text=True, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as error:
            diagnostic = (error.stderr or error.output or 'No diagnostic output was returned.').strip()
            raise RuntimeError(f'Could not read starting checkpoint {source} (exit {error.returncode}):\n{diagnostic}') from None
        meta = json.loads(result)
        if meta["base"] != BASE_MODEL or meta["base_revision"] != BASE_REVISION:
            raise RuntimeError("This Kev-4B lab requires a checkpoint from its pinned 4B backbone. Start fresh; 0.8B adapters are incompatible.")
        if meta["weights"] != "lora":
            raise RuntimeError("This classroom exercise expects the pinned small LoRA checkpoint.")
        command = [str(self.python), "-m", "kev.train", "--init_from", str(init_from), "--data", str(Path(training_data).resolve()), "--out", str(Path(output).resolve()), "--base", meta["base"], "--lora", str(meta["lora"]), "--head_dim", str(meta["head_dim"]), "--option_isolation", str(int(meta["option_isolation"])), "--special_embeddings", str(int(meta["special_embeddings"])), "--weights_dtype", meta["weights_dtype"], "--dtype", self.dtype, "--device", "cuda", "--epochs", "2", "--lr", "2e-5", "--batch", "1", "--accum", "8", "--max_steps", str(steps), "--max_state", str(max_state), "--checkpointing", "1", "--p_none", "0", "--p_none_distract", "0", "--p_distract", "0", "--seed", "7"]
        command[command.index("--max_state") + 1] = str(max_state)
        command[command.index("--dtype") + 1] = self.dtype
        # The pinned release targets all modules (including hybrid projections).
        # warm_start checks that its adapter tensors match the training model.
        command += ["--lora_targets", "all"]
        if meta["base_revision"]:
            command += ["--base_revision", meta["base_revision"]]
        return self._profile(command)

    def finetuning_command(self, training_data, output, init_from, steps=0, recipe=None):
        from planner_data import RECIPE
        recipe = dict(RECIPE if recipe is None else recipe)
        if set(recipe) != set(RECIPE):
            raise ValueError('An explicit fine-tuning recipe must provide the existing domain-training fields')
        if (int(recipe['batch']) * int(recipe['accum']) != 8 or int(recipe['replay']) < 0
                or int(recipe['epochs']) != 1 or not 0 < float(recipe['lr']) < 1):
            raise ValueError('Fine-tuning keeps one epoch and an effective request batch of eight')
        command = self.training_command(training_data, output, init_from, steps)
        # Pac-Man's short-context execution selection is explicit, just like
        # Stage 1's 4 x 2 selection; long-document memory overrides stay separate.
        for key, value in recipe.items():
            flag = '--' + key
            if flag in command:
                command[command.index(flag)+1] = str(value)
            else:
                command += [flag, str(value)]
        command += ['--suite', TRAINING_SUITE]
        return command

    def finetune(self, training_data, output, init_from, steps=0, resume=False, recipe=None):
        command = self.finetuning_command(training_data, output, init_from, steps, recipe=recipe)
        return self._train(command, output, 180, "pacman", resume=resume)

    def _train(self, command, output, timeout_minutes, stage, resume=False, allow_execution_change=False, resume_from=None):
        resume = resume or resume_from is not None
        if allow_execution_change and (not resume or stage != "initial"):
            raise ValueError("Execution-change continuation is restricted to resuming the initial stage")
        output = Path(output).resolve()
        recovery_root = Path(str(output) + "-recovery")
        snapshot = latest_snapshot(recovery_root) if resume and resume_from in (None, "latest") else None
        if resume_from is not None and resume_from != "latest":
            snapshot = Path(resume_from).resolve()
            if not snapshot.is_relative_to(recovery_root) or not (snapshot / "complete.json").is_file():
                raise ValueError("Choose a complete recovery snapshot inside this output's recovery directory.")
        if resume and (snapshot is None or not output.is_dir()):
            instruction = ("For a fresh Stage 1 run, set RESUME_CHECKPOINT=None inside the Stage 1 cell and rerun it. "
                           if stage == 'initial' else "Disable this stage's resume flag for a fresh output. ")
            raise RuntimeError(f"No resumable LoRA snapshot for {output}. Resume requested; training will not start fresh. "
                               + instruction + "If you intended recovery, restore the backup first.")
        if snapshot is not None and not (snapshot / "recovery.pt").is_file():
            raise ValueError("Selected recovery snapshot is missing recovery.pt; training will not start fresh.")
        if (not resume and output.is_dir() and not any(output.iterdir()) and not recovery_root.exists()):
            preserved = output.with_name(output.name + '-empty-attempt-' + uuid.uuid4().hex[:8])
            output.rename(preserved)
            print(f'Preserved empty output from an unstarted attempt: {preserved}', flush=True)
        if (output.exists() or recovery_root.exists()) and not resume:
            raise RuntimeError("Choose a new checkpoint output directory, or resume=True for a run with recovery snapshots.")
        if resume and (output / "run-evidence.json").exists():
            raise RuntimeError("This checkpoint is already complete; use it as the next stage's input.")
        if snapshot is not None:
            receipt = json.loads((snapshot / "complete.json").read_text())
            print(f"Resuming from optimizer step {receipt['step']}: {snapshot}", flush=True)
        report = self.workspace / "optimized-training-preflight.json"
        if self.training_preflight is None or self.training_preflight.get("result") not in {"passed", "bindings_verified"}:
            raise RuntimeError("Run setup and prepare_training first; pinned optimized bindings are required.")
        self.stop()  # release the inference model's GPU allocation before training
        started = time.perf_counter()
        log_dir = self.workspace / "logs" / stage / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
        # The pinned trainer is instrumented in memory; native --resume is
        # full-weight-only, so the lab restores LoRA state through guarded hooks.
        observer = Path(__file__).with_name("training_monitor.py")
        launch = [command[0], "-u", str(observer), *command[3:]]
        env = self.environment()
        env.update(LAB_RECOVERY_ROOT=str(recovery_root), LAB_SAVE_STEPS="100", LAB_SAVE_SECONDS="300")
        if self.backup_root is not None:
            env.update(LAB_BACKUP_ROOT=str(self.backup_root), LAB_BACKUP_WORKSPACE=str(self.workspace))
        if snapshot:
            env["LAB_RESUME_FROM"] = str(snapshot)
        if allow_execution_change:
            env["LAB_ALLOW_EXECUTION_CHANGE"] = "1"
        try:
            stream_training(launch, cwd=self.repo, env=env, log_dir=log_dir,
                            stage=stage, timeout_seconds=timeout_minutes * 60)
        except (Exception, KeyboardInterrupt):
            saved = latest_snapshot(recovery_root)
            if saved:
                info = json.loads((saved / "complete.json").read_text())
                print(f"Recovery available at step {info['step']}/{info['steps']}: {saved}. "
                      "Rerun with resume=True and identical arguments. Preserve this directory before disconnecting.", flush=True)
            else:
                print("No learned checkpoint saved yet. Logs: " + str(log_dir), flush=True)
            raise
        evidence = {"stage": stage, "command": command, "elapsed_seconds_including_load_save": time.perf_counter() - started,
                    "gpu": self.gpu, "code_revision": CODE_REVISION, "observer_command": launch,
                    "training_logs": str(log_dir), "telemetry": "one sample per optimizer step",
                    "training_profile": self.training_profile, "recovery_root": str(recovery_root),
                    "initial_execution": self.initial_execution if stage == "initial" else None,
                    "execution_change_allowed": allow_execution_change,
                    "resumed_from": str(snapshot) if snapshot else None,
                    "optimized_training_preflight": self.training_preflight,
                    "optimized_training_preflight_sha256": file_hash(report),
                    "drive_backup_root": str(self.backup_root) if self.backup_root is not None else None}
        final_snapshot = latest_snapshot(recovery_root)
        if final_snapshot:
            evidence["execution_history"] = json.loads((final_snapshot / "complete.json").read_text()).get("execution_history", [])
        status_path = log_dir / "status.json"
        if status_path.is_file():
            resumed_status = json.loads(status_path.read_text()).get("resumed") or {}
            change = resumed_status.get("execution_change")
            if change and change not in evidence.get("execution_history", []):
                evidence.setdefault("execution_history", []).append(change)
        if "--init_from" in command:
            evidence["parent_checkpoint_sha256"] = checkpoint_fingerprint(command[command.index("--init_from") + 1])
        if "--data" in command:
            import hashlib
            evidence["training_data_sha256"] = hashlib.sha256(Path(command[command.index("--data") + 1]).read_bytes()).hexdigest()
        (output / "run-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        metrics = json.loads((output / "training_metrics.json").read_text())
        if metrics["optimizer_steps"] < 1:
            raise RuntimeError("Training completed without a positive optimizer step.")
        if self.backup_root is not None:
            self._backup_checkpoint(output, final_snapshot)
        print(f"{stage} training finished in {evidence['elapsed_seconds_including_load_save'] / 60:.1f} minutes")
        return output
