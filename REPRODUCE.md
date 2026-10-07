# Reproducing Diffusion Policy — verified build log

Environment **`dp`** on this machine (Ubuntu 24.04, RTX 5070 sm_120, 12 GB VRAM, 20 cores, 31 GB RAM).

**Status: built and verified end-to-end on 2026-09-21.** Training, rollout, checkpointing and
`eval.py` all run on GPU. One optional piece needs your `sudo` (§6).

The upstream README does **not** work on this machine. This file records what was actually done,
what broke, and why. Companion docs: [`DIFFUSION_POLICY_GUIDE.md`](DIFFUSION_POLICY_GUIDE.md),
[`BENCHMARK_READINESS.md`](BENCHMARK_READINESS.md).

---

## 1. Proof it works

```console
$ python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml \
    training.debug=True training.device=cuda:0 logging.mode=offline \
    task.env_runner.n_envs=4 task.env_runner.n_test=2 task.env_runner.n_train=1 \
    task.env_runner.max_steps=60 hydra.run.dir=data/outputs/smoke
```

produced, in `data/outputs/smoke/`:

```
checkpoints/epoch=0000-test_mean_score=0.197.ckpt
checkpoints/epoch=0001-test_mean_score=0.089.ckpt
checkpoints/latest.ckpt
media/*.mp4                       # 6 rendered rollout videos
logs.json.txt                     # train_loss 1.19 -> 1.08, val_loss 1.10 -> 1.03
.hydra/{config,hydra,overrides}.yaml
```

and `eval.py` on that checkpoint reproduced the training rollout exactly:

```console
$ python eval.py --checkpoint data/outputs/smoke/checkpoints/latest.ckpt \
    --output_dir data/eval_smoke --device cuda:0
  test/mean_score  = 0.08892709519418862      # identical to the training run's final rollout
  train/mean_score = 0.1996563627097745
```

(Scores are low because `training.debug=True` caps the run at 2 epochs / 3 steps. The point is that
every stage executes.)

Measured facts:

| | |
|---|---|
| GPU | `torch 2.8.0+cu128`, RTX 5070 **sm_120**, 14.4 TFLOP/s fp32, 12.3 GB |
| Model | 262.7M params (251.5M diffusion + 11.2M vision) |
| **VRAM at the shipped `batch_size: 64`** | **6.47 GB of 12.3 GB** — fits, no tuning needed |
| VRAM at `batch_size: 32` | 6.42 GB (model states dominate; activations are minor for a 1-D UNet) |
| Dataset | PushT: 24,208 samples, image `(16,3,96,96)` in `[0,1]`, action `(16,2)` |

## 2. What is installed

```console
$ conda activate dp
```

| | |
|---|---|
| python | 3.9.18 |
| torch / torchvision | **2.8.0+cu128** / 0.23.0+cu128 |
| numpy / numba / gym / zarr | 1.23.3 / 0.56.4 / **0.21.0** / 2.12.0 |
| diffusers / huggingface_hub | 0.11.1 / **0.25.2** |
| robosuite / robomimic / egl_probe | 1.2.0 (fork `277ab95`) / 0.2.0 / 1.0.2 |
| dm-control / mujoco | 1.0.9 / 3.3.7 |
| pytorch3d | `0.0.0+dp.rotation.shim` — see §5 |
| wandb | **0.26.1**, not the repo's pinned 0.13.3 — see §7 |
| mujoco_py | installed, **cymj not yet compiled** — see §6 |

Rebuild from scratch with [`conda_environment_blackwell.yaml`](conda_environment_blackwell.yaml)
plus the pip stages in §3.

## 3. The build, and the five upstream contradictions

`conda env create -f conda_environment.yaml` (the README's file) produces an environment that
installs cleanly and then fails on the first CUDA call. Five separate problems had to be solved.

### 3.1 pytorch 1.12.1 / cu116 has no sm_120 kernels
Blackwell is `compute_cap 12.0`; that build ships cubins through sm_86. **Not a VRAM problem** — it
is a code-generation problem and fails on `torch.zeros(1).cuda()`.
Fix: `python=3.9` is forced by mujoco-py, and cu128 publishes cp39 wheels for torch 2.7.0, 2.7.1
and 2.8.0 only. Take the newest:
```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

### 3.2 conda-forge `diffusers`/`accelerate` drag pytorch back in
Both declare `pytorch >=1.4`, so leaving them in the conda section silently reinstalls
`pytorch 1.12.1` + `cudatoolkit 11.8`. Fix: install `diffusers` by **pip, after torch**; drop
`accelerate` and `datasets` (neither is imported anywhere in the repo).

### 3.3 `diffusers 0.11.1` is broken by modern huggingface_hub
```
ImportError: cannot import name 'cached_download' from 'huggingface_hub'
  diffusers/dynamic_modules_utils.py:29
```
`cached_download` was removed in hub 0.26; conda resolved 0.34.4. Every policy in this repo does
`from diffusers.schedulers.scheduling_ddpm import DDPMScheduler`, so nothing imported.
```bash
pip install "diffusers==0.11.1" "huggingface_hub<0.26"      # resolves to hub 0.25.2
```

### 3.4 robosuite's numba ceiling contradicts the repo's numba pin
```
robosuite 1.2.0 depends on numba<=0.53.1 and >=0.52.0
The user requested (constraint) numba==0.56.4
```
The repo needs numba 0.56.4 (numpy 1.23.3 compatibility, and the `@numba.jit` `SequenceSampler`
hot path). Without a constraints file pip resolves this by downgrading numba, which then drags
numpy below 1.21 and breaks the sampler. Fix — install robosuite ignoring its stale ceiling:
```bash
pip install --no-deps "robosuite @ https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz"
pip install robomimic==0.2.0        # builds egl_probe; succeeded here
```
`pip check` will then report exactly one complaint, and it is expected:
`robosuite 1.2.0 has requirement numba<=0.53.1,>=0.52.0, but you have numba 0.56.4.`

### 3.5 `dm-control` pulls a `mujoco` with no cp39 wheel
The current `mujoco` release dropped cp39, so pip source-builds it and dies on
`RuntimeError: MUJOCO_PATH environment variable is not set`. Pin the last cp39 release:
```bash
pip install dm-control==1.0.9 mujoco==3.3.7
```
Optional anyway: `SimRobot(use_dm_backend=False)` is the default, so Kitchen uses mujoco_py and
never imports `dm_control`.

> Use a pip constraints file (`numpy==1.23.3`, `numba==0.56.4`, `gym==0.21.0`, `zarr==2.12.0`)
> across all pip stages so ray/robomimic/dm-control cannot silently bump them. Verified unmoved
> after every stage.

## 4. No `torch.load` patch is needed

Earlier drafts of this document told you to patch four `torch.load` call sites with
`weights_only=False` for torch ≥ 2.6. **That was wrong, and testing showed it:** a dill-pickled
payload containing an OmegaConf config round-trips fine on torch 2.8. PyTorch only defaults
`weights_only=True` when `pickle_module` is **not** given; every call site here passes
`pickle_module=dill`, which flips the default to `False`. `eval.py` and `base_workspace.py` are
untouched, and `eval.py` loaded a real checkpoint successfully (§1).

## 5. pytorch3d, and why there is a shim

`pytorch3d` has exactly one importer — `diffusion_policy/model/common/rotation_transformer.py:2` —
and the repo only ever uses the `axis_angle` <-> `rotation_6d` pair (confirmed at every
`RotationTransformer(...)` call site and every `rotation_rep:` in the configs). It is needed only by
the **Robomimic** runners and datasets; PushT, BlockPush and Kitchen never touch it.

Upstream publishes no build for python 3.9 + torch 2.8 + cu128, and a source build is slow and
fragile. Installed instead: a 6-function pure-torch package that identifies itself as a shim.

```console
$ pip show pytorch3d
Name: pytorch3d
Version: 0.0.0+dp.rotation.shim
Summary: MINIMAL SHIM, not real PyTorch3D. ...
```

Validated by **the repo's own test**, which checks the round trip against
`scipy.spatial.transform.Rotation` and asserts `det == 1` on perturbed 6-D inputs:

```console
$ python -c "from diffusion_policy.model.common.rotation_transformer import test; test()"
# passes;  geodesic max error over 20,000 random rotations = 2.87e-15
```

Beware the wrong metric: an *elementwise* `|inverse(forward(aa)) - aa|` shows ~6.28 = 2π, because
axis-angle wraps and θ=π/−π are the same rotation. Geodesic distance is the correct measure.

To use real PyTorch3D instead: `pip uninstall pytorch3d` and build it from source.

## 6. The one thing that still needs you

`mujoco_py` cannot compile its `cymj` extension:

```
mujoco_py/gl/osmesashim.c:1:10: fatal error: GL/osmesa.h: No such file or directory
```

Root cause is a stale probe in `mujoco_py/builder.py:26`:

```python
paths = glob.glob('/usr/lib/nvidia-[0-9][0-9][0-9]')   # driver layout from ~2016
```

Modern Ubuntu has `/usr/lib/x86_64-linux-gnu`, so the glob returns empty, `get_nvidia_lib_dir()`
returns `None`, and mujoco_py falls back to `LinuxCPUExtensionBuilder` — the OSMesa (software)
path — even though this box has a GPU and EGL headers. The OSMesa headers are absent, so the
build fails.

**Run this one command:**

```bash
sudo apt install -y libosmesa6-dev libgl1 libglx-mesa0 libglfw3 patchelf
```

Note the README's version of this line uses `libgl1-mesa-glx`, which **does not exist on Ubuntu
24.04** (`apt-cache policy` → `Candidate: (none)`), so the whole command aborts and installs
nothing. The line above is the working equivalent. `patchelf` is already provided by conda in `dp`.

Then, for Kitchen and Robomimic:
```bash
export MUJOCO_GL=egl
python -c "import mujoco_py"        # compiles cymj once, takes a few minutes
```

**This blocks only Kitchen and Robomimic.** PushT works today, and so does training on your own
data.

---

## Track A — reproduce the published number (~1 h)

The authors' checkpoint filenames contain their measured score, so the target is unambiguous.

```bash
mkdir -p data/ckpt
wget -P data/ckpt "https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0500-test_mean_score=0.884.ckpt"

python eval.py --checkpoint "data/ckpt/epoch=0500-test_mean_score=0.884.ckpt" \
  --output_dir data/eval_pusht_image --device cuda:0

python -c "import json;print(json.load(open('data/eval_pusht_image/eval_log.json'))['test/mean_score'])"
```

| Verified-live checkpoint | Target |
|---|---|
| `image/pusht/diffusion_policy_cnn` → `epoch=0500-test_mean_score=0.884.ckpt` | **0.884** |
| `low_dim/pusht/diffusion_policy_cnn` → `epoch=0550-test_mean_score=0.969.ckpt` | **0.969** |

Methods available per task: `diffusion_policy_cnn`, `diffusion_policy_transformer`, `ibc_dfo`,
`lstm_gmm`. Test seeds are fixed at `100000..100049`, so the number is deterministic; treat within
~±0.02 as a match. Note this spawns **56 processes** (`n_envs: null` → `n_train + n_test`).

## Track B — train from scratch

```bash
wandb login       # or append logging.mode=offline

python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.seed=42 training.device=cuda:0 \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

That is the authors' own resolved config for the 0.884 checkpoint (`horizon: 16`, `n_obs_steps: 2`,
`n_action_steps: 8`, `num_epochs: 3050`, `rollout_every: 50`, `batch_size: 64`).

Sizing for this box, measured rather than guessed:

| Concern | Finding |
|---|---|
| VRAM | `batch_size: 64` peaks at **6.47 GB of 12.3 GB**. Leave it alone. |
| If you ever do OOM | `dataloader.batch_size=32 training.gradient_accumulate_every=2` (identical effective batch; verified at `train_diffusion_unet_hybrid_workspace.py:167-174`). **AMP is not available** — no `autocast`/`GradScaler` anywhere in this codebase. |
| Rollout cost | 61 rollouts × 56 envs × 100 denoising steps. To cut it: `training.rollout_every=200 task.env_runner.n_test=20`. |
| System RAM (31 GB) | The binding constraint for Robomimic *image* eval (~1 GB/env, config asks for 64 GB). Lower `task.env_runner.n_envs`. |
| Multi-seed | `ray_train_multirun.py` assumes ≥3 GPUs. Run seeds 42/43/44 sequentially. |

Wall-clock for one PushT image seed is **days**, not hours (3050 epochs + 61 full rollouts).

## Track C — train on your own data

The template is the **real-robot** path, because it is the one with no simulator.

1. **Dataset** — imitate `diffusion_policy/dataset/pusht_image_dataset.py`. Back it with a zarr
   `ReplayBuffer` (`data/<key>` arrays concatenated over time + `meta/episode_ends`) and sample with
   `SequenceSampler(sequence_length=horizon, pad_before=n_obs_steps-1, pad_after=n_action_steps-1)`.
   `diffusion_policy/dataset/mujoco_image_dataset.py` shows the zarr-key pattern for
   TCP-pose + gripper data, but **do not copy it verbatim** — its `get_normalizer()` builds a 14-d
   `agent_pos` while `_sample_to_data()` emits 8-d.

2. **Image format** — `(To, 3, H, W)` float32 in `[0,1]`, **channel-first**. The README says
   `(To,H,W,3)` at lines 305/313; that is wrong. The code does
   `np.moveaxis(sample['img'], -1, 1) / 255`.

3. **`env_runner`** — with no simulator, reuse the no-op:
   ```yaml
   env_runner:
     _target_: diffusion_policy.env_runner.real_pusht_image_runner.RealPushTImageRunner
   ```
   Its `run()` returns `dict()`.

4. **Change the checkpoint monitor — mandatory.** A no-op runner emits no `test_mean_score`, so the
   default `topk` config raises `KeyError` at the first checkpoint epoch (50 epochs into your run).
   Copy what `train_diffusion_unet_real_image_workspace.yaml` does:
   ```yaml
   checkpoint:
     topk:
       monitor_key: train_loss
       mode: min
       format_str: 'epoch={epoch:04d}-train_loss={train_loss:.3f}.ckpt'
   ```

5. **Train:**
   ```bash
   python train.py --config-name=train_diffusion_unet_real_image_workspace \
     task.dataset_path=data/my_demos
   ```

Keep `shape_meta` consistent across the task yaml, the dataset and the policy — it drives the obs
keys, the vision encoders and the action dimension simultaneously.


---

## 7. wandb: the pin must be broken, and that is fine

The repo pins `wandb=0.13.3` (Sept 2022). **That version cannot accept a modern API key.**
`wandb/sdk/lib/apikey.py:221` enforces:

```python
prefix, suffix = key.split("-", 1) if "-" in key else ("", key)
if len(suffix) != 40:
    raise ValueError("API key must be 40 characters long, yours was %s" % len(key))
```

W&B's current `wandb_v1_...` keys are not 40 characters and use underscores, not the dash the
prefix-stripping expects — so the whole string is measured and rejected. This is not fixable by
getting a "shorter" key; only exactly 40 passes.

**Resolution: wandb was upgraded to 0.26.1.** Verified working on 2026-09-21 with a real online run.

The upgrade risk was the two *private* wandb APIs this repo reaches into. Both survived:

| Private access | Where | 0.26.1 |
|---|---|---|
| `wandb.sdk.data_types.video` | imported as `wv` in all 6 env_runners | exists |
| `wandb.sdk.data_types.video.Video` | `eval.py:56` isinstance check | exists, and `is wandb.Video` |
| `Video._path` | `eval.py:57` | exists |
| `wandb.util.generate_id` | runners, for video filenames | exists |

The pinned versions all held through the upgrade: `numpy 1.23.3`, `numba 0.56.4`, `gym 0.21.0`,
`zarr 2.12.0`, `torch 2.8.0+cu128`, `diffusers 0.11.1`.

**Credentials live in `~/.netrc` (perms 600), not in `.bashrc`.** wandb reads netrc natively and
applies no length validation on the read path, so this works regardless of key format. An env var
would also work but propagates into all 56 forked rollout workers and is readable via
`/proc/<pid>/environ`.

**Harmless warning you will see four times per rollout epoch:**
```
wandb: WARNING `format` argument was not provided, defaulting to `gif`.
```
All six runners call `wandb.Video(video_path)` with no `format=` (e.g.
`pusht_image_runner.py:242`). Verified cosmetic — the uploaded artifacts are real
`.mp4` files with `_type: video-file`. Ignore it, or pass `format="mp4"` in the runners.
