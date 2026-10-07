# Benchmark Readiness Audit

> **UPDATE 2026-09-21 — SUPERSEDED IN PART. The environment has since been built and verified.**
> Conda env **`dp`** now runs training, rollout, checkpointing and `eval.py` on the GPU. See
> [`REPRODUCE.md`](REPRODUCE.md) for the verified build log. This document is kept for the original
> evidence; three claims below were corrected by actually running things, and are marked inline.
> Remaining blocker: one `sudo apt install libosmesa6-dev` for Kitchen/Robomimic only.

**Question:** can this machine run the Diffusion Policy benchmarks, so a novel image-based robot
motion policy can be evaluated against them?

**Verdict: NOT READY.** Nothing is installed, no data is present, and — the part that matters —
the repo's pinned PyTorch cannot generate code for this machine's GPU. This is not a 20-minute
`conda env create`; budget **1–2 days of dependency work** before the first benchmark number.

Companion document: **[`DIFFUSION_POLICY_GUIDE.md`](DIFFUSION_POLICY_GUIDE.md)** — what the code is
and how it fits together.

Audited 2026-09-20 against HEAD `5ba07ac`.

---

## 1. Summary

| # | Blocker | Severity | Effort |
|---|---|---|---|
| 1 | **RTX 5070 is sm_120; the repo pins PyTorch 1.12.1 / CUDA 11.6 (sm_86 max)** | **Hard** | 1–2 days — bump torch, then repair the cascade |
| 2 | No `robodiff` conda env; none of the stack is installed anywhere | Hard | ~1 h, after #1 is resolved |
| 3 | No datasets at all (`data/` does not exist) | Hard | minutes to hours, by task |
| 4 | The README's `apt` line is invalid on Ubuntu 24.04 and installs nothing | Medium | 1 min, with the corrected command |
| 5 | OSMesa missing; no rendering backend is set for MuJoCo | Medium | Needed only for Kitchen / Robomimic |

What *is* fine: disk (697 GB free), CPU (20 cores), the upstream package artifacts (all still
resolve — this spec has not bit-rotted), and the dataset download host (live).

---

## 2. What was measured

```console
$ nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
name, compute_cap, memory.total [MiB], driver_version
NVIDIA GeForce RTX 5070, 12.0, 12227 MiB, 595.71.05

$ conda env list
base                 * /home/elijahong/anaconda3
autoresearch           /home/elijahong/anaconda3/envs/autoresearch
edmund_lens            /home/elijahong/anaconda3/envs/edmund_lens
env_isaaclab           /home/elijahong/anaconda3/envs/env_isaaclab
xacro_env              /home/elijahong/anaconda3/envs/xacro_env
                                        # <-- no robodiff

$ ldconfig -p | grep -c osmesa
0

$ ls -l /usr/bin/patchelf
ls: cannot access '/usr/bin/patchelf': No such file or directory

$ apt-cache policy libgl1-mesa-glx
libgl1-mesa-glx:
  Installed: (none)
  Candidate: (none)                     # <-- removed in Ubuntu 24.04

$ ls data
ls: cannot access 'data': No such file or directory
```

Additional environment facts:

| | |
|---|---|
| OS | Ubuntu 24.04.4 LTS (noble), kernel 7.0.0-28-generic |
| CPU / RAM | 20 cores / 31 GiB total, 26 GiB available |
| Disk | 913 G total, **697 G free** (repo and `$HOME` on the same filesystem) |
| Active python | `/home/elijahong/anaconda3/bin/python` → **3.13.5** (conda base) |
| conda | 25.5.1, libmamba solver; `mamba` itself not on PATH |
| Toolchain | gcc 13.3.0, glibc 2.39; **no `nvcc`, no `/usr/local/cuda*`** |
| Present GL libs | `libglfw3`, `libglew2.2`, `libEGL_nvidia.so.0`, `libEGL_mesa.so.0`, `libglvnd0` |
| Missing | `libOSMesa`, system `patchelf` (only `~/anaconda3/bin/patchelf` exists) |

Base env has **none** of torch, gym, zarr, hydra, pymunk, pygame, mujoco, mujoco-py, robosuite,
robomimic, pybullet, or wandb. All import probes fail with `ModuleNotFoundError`.

---

## 3. Blocker 1 — the GPU outruns the pinned stack

This is the one that costs real time.

```yaml
# conda_environment.yaml:10-11
  - cudatoolkit=11.6
  - pytorch=1.12.1
```

The RTX 5070 is Blackwell, **compute capability 12.0 (sm_120)**. CUDA 11.6's `nvcc` tops out at
sm_86, and the official PyTorch 1.12.1+cu116 build ships cubins only through sm_86 — Blackwell
postdates it by three years. The bundled cuDNN 8.3.2 and cuBLAS have no Blackwell kernels either,
so no PTX-JIT path rescues it.

`conda env create` will **succeed**, and then the first CUDA op will fail with some variant of:

```
NVIDIA GeForce RTX 5070 with CUDA capability sm_120 is not compatible with the current PyTorch
installation. The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_61
sm_70 sm_75 sm_80 sm_86.
```

> Labelled honestly: the exact message is **inferred** from well-established PyTorch behaviour, not
> observed here — nothing has been installed. What *is* measured is `compute_cap 12.0` and the
> `pytorch=1.12.1` / `cudatoolkit=11.6` pins. The incompatibility follows from those two facts.

The first PyTorch release with official sm_120 support is **2.7 with CUDA 12.8**. Driver 595.71.05
(CUDA 13.2) is new enough. CPU fallback is not viable: the reference config is 3050 epochs of a
`[512, 1024, 2048]` 1-D UNet.

### The cascade that bumping torch triggers

**1. `pytorch3d=0.7.0` becomes unobtainable.** The only matching build is
`pytorch3d-0.7.0-py39_cu116_pyt1121.tar.bz2` — hard-pinned to torch 1.12.1.

How much this hurts depends entirely on which benchmarks you run. Measured reach:

```console
$ grep -rn "pytorch3d" --include=*.py .
diffusion_policy/model/common/rotation_transformer.py:2:import pytorch3d.transforms as pt
```

One import site. `RotationTransformer` is used only by `robomimic_{image,lowdim}_runner.py` and
`robomimic_replay_{image,lowdim}_dataset.py`. (`common/normalize_util.py:47` merely takes
`rotation_transformer` as a *parameter name* — no import.)

> **Push-T, BlockPush and Kitchen are pytorch3d-free. Robomimic is not.** Since Robomimic is one of
> your two target suites, you will need pytorch3d built from source against the new torch — or a
> replacement for `RotationTransformer`, which only needs axis-angle ↔ matrix ↔ rotation-6d
> conversions and is ~100 lines that `scipy.spatial.transform.Rotation` can cover.

**2. ~~`torch.load` breaks on the published checkpoints.~~ — CORRECTED: no patch needed.**
I predicted that torch ≥2.6's `weights_only=True` default would break the `dill`-pickled
checkpoints and that four call sites would need patching. **Testing disproved this.** PyTorch only
defaults `weights_only=True` when `pickle_module` is *not* supplied; every call site here passes
`pickle_module=dill`, which flips the default back to `False`. A dill payload containing an
OmegaConf config round-trips fine on torch 2.8, and `eval.py` loaded a real checkpoint
successfully. `eval.py` and `base_workspace.py` are untouched.

**3. Pins that must NOT move, whatever else you change:**

| Pin | Why |
|---|---|
| `python=3.9` | `free-mujoco-py==2.1.6` declares `requires_python >=3.7.1,<3.11` |
| `gym=0.21.0` | The codebase is written against the 4-tuple `step` API — `gym_util/multistep_wrapper.py:109` is `observation, reward, done, info = super().step(act)`. The repo also vendors forks of `gym.vector` importing module paths that moved in later gym. **Do not "helpfully" upgrade to gymnasium.** |
| `numpy=1.23.3` | `numba==0.56.4` requires `numpy<1.24`, and numba is on the hot path (`common/sampler.py:3`, `common/replay_buffer.py:377`). numpy 2.x is not an option. |
| `robosuite @ cheng-chi/robosuite@277ab95` | A **fork pinned by SHA**, not upstream robosuite. Upstream produces import errors or wrong results. |
| `robomimic==0.2.0` | 0.4.0+ changed `algo_factory` and `ObsUtils` signatures. |

### Why `env_isaaclab` cannot be reused

It has `torch 2.5.1+cu121` (also pre-Blackwell), `gym 0.23.1` + `gymnasium 1.1.1` (wrong API),
`robomimic 0.4.0` (wrong API), and no pymunk / zarr / pybullet / mujoco-py. Likewise the sibling
checkout at `~/robot-learning/robomimic` is **robomimic 0.5.0** — five minor versions past the pin.
Neither is a shortcut.

---

## 4. Blockers 2–5

### No environment

There is no `robodiff` env and no warm conda package cache to start from
(`ls ~/anaconda3/pkgs | grep -iE "pytorch|mujoco|robosuite|gym|zarr"` is empty).

Good news on supply: every pinned artifact was confirmed to still resolve —
`pytorch-1.12.1-py3.9_cuda11.6_cudnn8.3.2_0`, `pytorch3d-0.7.0-py39_cu116_pyt1121`,
`gym-0.21.0-py39hef51801_2` (conda-forge — so you avoid the notorious
`pip install gym==0.21.0` metadata failure), `ray-2.2.0`, `imagecodecs-2022.9.26`, and both GitHub
tarballs (`cheng-chi/robosuite@277ab95`, `facebookresearch/r3m`) return HTTP 200. **The spec has not
bit-rotted; the hardware moved on.**

Source-only installs that are known sharp edges: `robomimic==0.2.0` (sdist only) pulls `egl_probe`,
which also ships sdist-only and CMake-builds against OpenGL/EGL headers; and `free-mujoco-py`
compiles a Cython extension (`cymj`) on first import, needing gcc, Cython <0.30 (pinned 0.29.32 ✓),
`patchelf`, and GL headers. *(gcc 13 against a 2021-era Cython codebase is a plausible failure
point — unverified.)*

Dead weight you can drop to ease the solve: `pytorchvideo==0.1.5`, `accelerate`, and `datasets` are
declared but never imported anywhere in the codebase.

### No data

Nothing at all: no `data/`, no `.zarr`, no `.hdf5`, no `.ckpt` under `~/robot-learning`. `data` is
in `.gitignore`. The host is live (`last-modified: Mon, 27 Feb 2023`). There is **no download
script** in the repo — only `wget` lines in the README.

| Archive | Size | Needed for |
|---|---|---|
| `pusht.zip` | **0.03 GB** | Push-T (image + lowdim + keypoints) |
| `block_pushing.zip` | 0.01 GB | Block Pushing |
| `kitchen.zip` | 0.78 GB | Franka Kitchen |
| `robomimic_lowdim.zip` | 1.93 GB | 5 robomimic lowdim tasks |
| **`robomimic_image.zip`** | **84.75 GB** | 5 robomimic image tasks |
| `pusht_real.zip` | 2.45 GB | real robot only — not needed |

697 GB free, so even the full set fits. Base URL:
`https://diffusion-policy.cs.columbia.edu/data/training/`

Two things the download does **not** give you:

- **`*_abs.hdf5` are generated locally**, not shipped. Run
  `diffusion_policy/scripts/robomimic_dataset_conversion.py` before any `*_abs` task config.
- `RobomimicImageRunner` and `KitchenLowdimRunner` read dataset files **in `__init__`**
  (`robomimic_image_runner.py:79,166`; `kitchen_lowdim_runner.py:84-85`) — they cannot even be
  constructed without data. `PushTImageRunner` is the only runner that can.

### The README's apt line is broken on this OS

README says:

```bash
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf     # DO NOT RUN — aborts
```

`libgl1-mesa-glx` was removed in Ubuntu 24.04 (`Candidate: (none)`, measured above). apt aborts
with `E: Unable to locate package libgl1-mesa-glx` and **installs nothing** — including
`libosmesa6-dev` and `patchelf`, which are genuinely missing and genuinely needed. Use:

```bash
sudo apt install -y libosmesa6-dev libgl1 libglx-mesa0 libglfw3 patchelf
```

Both `libosmesa6-dev` (25.1.7-1ubuntu2~24.04.2) and `patchelf` (0.18.0-1.1build1) are available in
noble.

### Rendering backend is your responsibility

```console
$ grep -rniE "osmesa|MUJOCO_GL|GLEW" --include=*.py . | wc -l
0
```

The repo sets no rendering backend anywhere in Python. EGL is available (`libEGL_nvidia.so.0`
present) and is the faster choice on an NVIDIA box, so export `MUJOCO_GL=egl` for Kitchen and
Robomimic. OSMesa is currently unavailable (0 hits in `ldconfig`) until the apt command above runs.

Per-task rendering risk:

| Task | Backend | Headless risk |
|---|---|---|
| **Push-T** | pygame + pymunk, **no OpenGL at all** | **None** |
| BlockPush | pybullet (software TinyRenderer fallback) | Low |
| Kitchen | MuJoCo offscreen GL | Medium — needs `MUJOCO_GL=egl` |
| Robomimic | robosuite MuJoCo + forked `AsyncVectorEnv` | **Highest** — the README itself warns that envs creating a GL context at init "often causes obscure bugs like segmentation fault"; mitigated but not eliminated by `dummy_env_fn` |

`~/.mujoco` does not exist and there is no `mjkey.txt` — **this is fine by design.**
`free-mujoco-py` bundles the MuJoCo 2.1 binaries in its wheel, and MuJoCo has been licence-free
since 2021. No manual setup needed.

---

## 5. How to plug your policy in

The good news, and the reason this repo is worth the setup cost: **the evaluation harness is
genuinely policy-agnostic.** Nothing about the runners assumes diffusion.

### The contract

`diffusion_policy/env_runner/base_image_runner.py:8` in full:

```python
class BaseImageRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BaseImagePolicy) -> Dict:
        raise NotImplementedError()
```

Your policy needs four members:

```python
import torch
from typing import Dict
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer

class MyMotionPolicy(BaseImagePolicy):
    def __init__(self, shape_meta: dict, n_obs_steps: int, n_action_steps: int, horizon: int):
        super().__init__()
        self.n_action_steps = n_action_steps
        self.normalizer = LinearNormalizer()
        # ... your model ...

    def reset(self):
        """Called once per rollout batch, before the episode loop. Clear temporal state."""

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # obs_dict is FLAT, keyed by obs name, already on self.device:
        #   'image'/'agentview_image'/... -> (B, To, 3, H, W) float32 in [0,1], channel-first
        #   'agent_pos'/'robot0_eef_pos'/... -> (B, To, D)
        B = next(iter(obs_dict.values())).shape[0]
        action = ...                       # (B, Ta, Da)
        return {'action': action}
```

`.device` and `.dtype` come free from `ModuleAttrMixin`. `'action_pred'` `(B, T, Da)` is only needed
if you also train inside this repo's workspaces — the env runners never read it.

### Do not use `eval.py`

`eval.py:31-45` loads a checkpoint, pulls `cfg` out of it, and reconstructs **the workspace class
the checkpoint names**. That works for reproducing published results; it cannot evaluate an
external policy. Write a small driver instead:

```python
# bench_my_policy.py  (repo root)
import pathlib, json, hydra
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)

# task configs interpolate ${horizon} etc. from the parent config, so supply them
cfg = OmegaConf.load('diffusion_policy/config/task/pusht_image.yaml')
cfg = OmegaConf.merge(OmegaConf.create({
    'horizon': 16, 'n_obs_steps': 2, 'n_action_steps': 8,
    'n_latency_steps': 0, 'past_action_visible': False,
}), cfg)
OmegaConf.resolve(cfg)

out = pathlib.Path('data/bench/my_policy_pusht'); out.mkdir(parents=True, exist_ok=True)

policy = MyMotionPolicy(shape_meta=cfg.shape_meta, n_obs_steps=2, n_action_steps=8, horizon=16)
policy.set_normalizer(hydra.utils.instantiate(cfg.dataset).get_normalizer())
policy.to('cuda:0').eval()

runner = hydra.utils.instantiate(cfg.env_runner, output_dir=str(out))
log = runner.run(policy)
print('test/mean_score =', log['test/mean_score'])
json.dump({k: v for k, v in log.items() if isinstance(v, (int, float))},
          open(out / 'eval_log.json', 'w'), indent=2)
```

Start with `task.env_runner.n_envs=4 n_test=4 n_train=1` — the shipped config forks **56 processes**.

### Reading the numbers

| Task family | `test/mean_score` means |
|---|---|
| Push-T | mean over episodes of **max goal coverage**, in `[0,1]` — not a binary success rate |
| Robomimic | mean of per-episode max sparse reward = **success rate** |
| Kitchen | fraction of the 7 subtasks completed; also reports `p_1` … `p_7` |
| BlockPush | also reports `p1`, `p2` and per-event probabilities |

Per-episode detail arrives as `{train,test}/sim_max_reward_{seed}` and
`{train,test}/sim_video_{seed}`. Seeds are deterministic: test seeds are
`test_start_seed + i` = `100000..100049` by default, so runs are comparable across policies.

If you add custom observation keys on the Robomimic path, note that `RobomimicImageWrapper`
whitelists keys by **suffix** (`*image`, `*quat`, `*qpos`, `*pos`) and raises `RuntimeError`
otherwise.

---

## 6. Recommended order of attack

**Do Push-T first, and treat it as the smoke test for your policy plumbing.** It is pure
pymunk/pygame — no MuJoCo, no robosuite, no pytorch3d, no OpenGL — with a 31 MB dataset. That
isolates the torch/CUDA problem from the MuJoCo problem, so when something breaks you know which
one broke.

```bash
# 1. system libs — the CORRECTED command (the README's aborts on noble)
sudo apt install -y libosmesa6-dev libgl1 libglx-mesa0 libglfw3 patchelf

# 2. create the env; expect success, then a CUDA-broken torch
conda env create -f conda_environment.yaml

# 3. THE DECISION POINT — confirm Blocker 1 before investing further
conda run -n robodiff python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.zeros(1).cuda()+1)"

# 4. data (31 MB)
mkdir -p data && cd data \
  && wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip \
  && unzip pusht.zip && rm -f pusht.zip && cd ..

# 5. smoke test: 2 epochs, offline logging, tiny rollout
conda run -n robodiff python train.py --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.debug=True training.device=cuda:0 logging.mode=offline \
  task.env_runner.n_envs=4 task.env_runner.n_test=4 task.env_runner.n_train=1 \
  hydra.run.dir='data/outputs/smoke'
```

When step 3 fails, the repair is a modified `conda_environment.yaml`: keep `python=3.9`,
`numpy=1.23.3`, `gym=0.21.0`, `numba=0.56.4` **unchanged**; drop `cudatoolkit`, `pytorch`,
`torchvision`, `pytorch3d`; install `torch>=2.7` from
`--index-url https://download.pytorch.org/whl/cu128`; then patch the four `torch.load` sites for
`weights_only=False`. Add pytorch3d back (built from source) only when you move to Robomimic.

Then, in increasing order of setup cost:

| Order | Task | New dependencies | Data |
|---|---|---|---|
| 1 | **Push-T image** | none beyond core | 0.03 GB |
| 2 | BlockPush | pybullet (assets vendored) | 0.01 GB |
| 3 | Kitchen | mujoco-py cymj build, `MUJOCO_GL=egl` | 0.78 GB |
| 4 | **Robomimic lowdim** | robosuite fork + robomimic 0.2.0 + pytorch3d | 1.93 GB |
| 5 | **Robomimic image** | as above + EGL offscreen under fork | **84.75 GB** |

### Resource sizing for this box

> **VRAM is not the blocker.** Blocker 1 (§3) is a *code-generation* failure — the pinned PyTorch
> contains no machine code your GPU can execute, so it dies on the first CUDA op regardless of
> tensor size. The notes below are about comfort once that is fixed, not about feasibility.

- **VRAM (12 GB) is NOT a constraint — MEASURED.** I estimated ~5 GB of persistent state and
  called batch 64 "genuinely unclear without measuring". It has now been measured: the shipped
  `batch_size: 64` peaks at **6.47 GB of 12.3 GB**, with model + grads + AdamW + the EMA copy
  dominating (batch 32 peaks at 6.42 GB — nearly identical, because activations are minor for a
  1-D UNet over T=16). **Leave `batch_size: 64` alone.** If you ever do OOM,
  `dataloader.batch_size=32 training.gradient_accumulate_every=2` is the identical effective batch.
  Mixed precision remains unavailable — there is no `autocast`/`GradScaler` in this codebase.
- **System RAM, not VRAM, is what constrains Robomimic image *eval*.** `n_envs: 28` at ~1 GB per
  forked env against **31 GB total system RAM**; the config comments ask for a 16-core/64 GB
  instance. Lower `n_envs`. This is a separate resource from the 12 GB on the card.
- **The Ray multi-seed workflow assumes ≥3 GPUs.** With one GPU, run seeds sequentially.
- 20 cores is comfortable for `dataloader.num_workers: 8`.

---

## 7. Repo state and provenance

```console
$ git status --porcelain
?? prompt.md
$ git diff HEAD --stat
(empty)
$ git remote -v
origin  git@github.com:real-stanford/diffusion_policy.git
```

Working tree clean. **HEAD is not the official upstream tip:**

```
5ba07ac 2024-12-24 Yihuai Gao   Done adapting mujoco image dataset      <-- HEAD, fork-only
548a52b 2023-10-26 Cheng Chi    Merge pull request #27 from pointW/main <-- upstream tip
```

The fork commit is purely additive (293 lines, 3 files) and touches no benchmark path, so HEAD is
functionally equivalent to upstream for all standard tasks. It adds an **orphaned**
`dataset/mujoco_image_dataset.py` (no config, env, or runner references it) that carries two real
defects — its `get_normalizer()` builds a 14-d `agent_pos` while `_sample_to_data()` emits 8-d, and
`get_all_actions()` is unimplemented. Details in
[`DIFFUSION_POLICY_GUIDE.md` §10.13](DIFFUSION_POLICY_GUIDE.md). Do not use it as a template.

If you want canonical paper baselines rather than a fork, `git checkout 548a52b`.

---

## 8. Unverified at audit time — now resolved

The original audit listed six open risks. Five have since been settled by building the environment:

| Open risk | Outcome |
|---|---|
| Does `conda env create` solve end-to-end? | **Yes** — after removing `diffusers`/`accelerate`/`datasets`, which drag conda-forge `pytorch 1.12.1` + `cudatoolkit 11.8` back in. |
| Does `free-mujoco-py`'s cymj compile under gcc 13? | **Not yet** — it never reached the compiler for the right backend. `mujoco_py/builder.py:26` globs `/usr/lib/nvidia-[0-9][0-9][0-9]`, a 2016-era driver layout, so it falls back to the OSMesa builder and dies on a missing `GL/osmesa.h`. Needs `sudo apt install libosmesa6-dev`. |
| Does `egl_probe` build? | **Yes** — `egl_probe-1.0.2-cp39-cp39-linux_x86_64.whl` built cleanly. |
| Does EGL offscreen rendering work for robosuite? | **Still open** — gated behind the mujoco_py build above. |
| Actual VRAM headroom at `batch_size: 64`? | **Measured: 6.47 GB of 12.3 GB.** Fits comfortably. |
| Exact PyTorch sm_120 error text? | **Moot** — torch 2.8.0+cu128 runs on sm_120 at 14.4 TFLOP/s fp32. |

Two problems the audit did *not* anticipate, both found only by building:
`diffusers 0.11.1` fails against `huggingface_hub >= 0.26` (`cached_download` removed), and
`robosuite 1.2.0` demands `numba<=0.53.1` while the repo pins `numba==0.56.4` — a contradiction
inside the upstream spec itself. Both are documented in [`REPRODUCE.md`](REPRODUCE.md) §3.
