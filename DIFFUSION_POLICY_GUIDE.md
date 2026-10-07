# Diffusion Policy — Codebase Guide

A complete reference for this checkout: what the code is, how it is organised, what the
interfaces are, and where an external policy plugs in.

Companion document: **[`BENCHMARK_READINESS.md`](BENCHMARK_READINESS.md)** — whether this machine
can actually run any of it (short answer: not yet).

**Paper:** *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* — Chi, Feng, Du,
Xu, Cousineau, Burchfiel, Song (RSS 2023).

---

## 1. What this is — and the fork caveat

This is **not** pristine upstream.

```
origin  git@github.com:real-stanford/diffusion_policy.git

5ba07ac 2024-12-24 Yihuai Gao   Done adapting mujoco image dataset      <-- HEAD, fork-only
548a52b 2023-10-26 Cheng Chi    Merge pull request #27 from pointW/main <-- official upstream tip
de4384e 2023-10-25 Dian Wang    fix typo in rotation_transformer.py
7dd9dc4 2023-09-12 Cheng Chi    Merge PR #21 fix_cpu_affinity
```

HEAD sits one commit above official. That commit is **purely additive** (293 lines, 3 files) and
touches no benchmark path, so HEAD is functionally equivalent to upstream for all 12 standard
tasks. What it adds:

| File | What it is |
|---|---|
| `diffusion_policy/dataset/mujoco_image_dataset.py` | **Orphaned.** `grep` finds no config, env, or runner referencing it. See §10.13 — it has two real defects. Do not use it as a template. |
| `diffusion_policy/common/replay_buffer.py` (+2 lines) | In `copy_from_store`, skips nested `zarr.Group`s under `meta/`. Benign. |
| `image_pusht_diffusion_policy_cnn.yaml` (repo root) | The *resolved* upstream PushT experiment config, committed as-is. Consumed with `--config-dir=.` |

If you need canonical paper baselines, `git checkout 548a52b`.

### Core idea of the method

The policy models `p(A_t | O_t)` — a **sequence** of future actions conditioned on a short
observation history — as a denoising diffusion process, rather than regressing a single action.
This is what handles multimodal demonstrations (the standard failure mode of vanilla BC).

Three horizon quantities appear everywhere, in configs and in code:

| Symbol | Config key | Typical | Meaning |
|---|---|---|---|
| `To` | `n_obs_steps` | 2 | observation steps fed to the policy |
| `Ta` | `n_action_steps` | 8 | actions actually executed per inference (receding horizon) |
| `T`  | `horizon` | 16 | full predicted trajectory length |

```
 |o|o|                          To = 2 observations in
 | | |a|a|a|a|a|a|a|a|          Ta = 8 actions executed
 |<--------- T = 16 --------->| full prediction; the tail is discarded
```

---

## 2. Directory map

### Repo root

| Path | Role |
|---|---|
| `train.py` | Hydra training entrypoint (35 lines of glue) |
| `eval.py` | Click checkpoint-evaluation entrypoint |
| `eval_real_robot.py` | Real UR5 + RealSense + SpaceMouse closed-loop eval |
| `demo_pusht.py` | Collect Push-T demos with a mouse (pygame) into a zarr `ReplayBuffer` |
| `demo_real_robot.py` | Collect real-robot demos with a SpaceMouse |
| `ray_exec.py`, `ray_train_multirun.py` | Multi-seed launcher on a Ray cluster (assumes ≥3 GPUs) |
| `multirun_metrics.py` | Aggregates `logs.json.txt` across seeds → `metrics.json` |
| `image_pusht_diffusion_policy_cnn.yaml` | Flat, fully-resolved experiment config (see §1) |
| `conda_environment{,_macos,_real}.yaml` | Env specs (sim / macOS dev / real robot) |
| `setup.py` | 3-line stub, `find_packages()`, **no `install_requires`** |
| `tests/` | 12 pytest files |

`setup.py` declares no dependencies and the README never says `pip install -e .` — entrypoints rely
on **cwd being the repo root** for `import diffusion_policy` to resolve.

### `diffusion_policy/` — 13 subpackages

| Subdir | Contents |
|---|---|
| `codecs/` | Jpeg2000/JpegXl numcodecs wrappers for in-RAM compressed image datasets |
| `common/` | 16 utilities: `replay_buffer.py`, `sampler.py`, `normalize_util.py`, `json_logger.py`, `checkpoint_util.py`, `pytorch_util.py`, `robomimic_{config_,}util.py`, `pose_trajectory_interpolator.py`, … |
| `config/` | 22 workspace YAMLs + `task/` with 27 task YAMLs |
| `dataset/` | `base_dataset.py` + 9 concrete datasets |
| `env/` | Vendored simulators: `pusht/`, `block_pushing/`, `kitchen/` (incl. the full `relay_policy_learning/` tree with MuJoCo assets), `robomimic/` (gym wrappers only) |
| `env_runner/` | 2 base + 7 concrete vectorized rollout evaluators |
| `gym_util/` | `async_vector_env.py` (patched), `sync_vector_env.py`, `multistep_wrapper.py`, `video_recording_wrapper.py`, `video_wrapper.py` (unused by any runner) |
| `model/` | `diffusion/`, `vision/`, `common/`, `bet/` |
| `policy/` | 2 base + 11 concrete policies |
| `real_world/` | UR5/RealSense/SpaceMouse stack |
| `scripts/` | 9 dataset-conversion / metric CLIs |
| `shared_memory/` | Lock-free ring buffer + FIFO queue for the real-robot processes |
| `workspace/` | `base_workspace.py` + 11 concrete training workspaces |

> The `env/kitchen/relay_policy_learning/` tree is **vendored, not a submodule** — there is no
> `.gitmodules`. The Franka/kitchen MuJoCo XMLs and meshes ship in-repo.

### `diffusion_policy/scripts/`

| Script | Purpose |
|---|---|
| `robomimic_dataset_conversion.py` | Robomimic HDF5 → **absolute-action** HDF5 (`*_abs.hdf5`). Required for every `*_abs` task — these files are *generated*, not downloaded. |
| `robomimic_dataset_action_comparison.py` | Sanity-check delta vs absolute action replay |
| `bet_blockpush_conversion.py` | BET `.npy` → zarr `ReplayBuffer` |
| `blockpush_abs_conversion.py` | Delta → absolute EEF targets in a blockpush zarr |
| `generate_bet_blockpush.py` | Generate blockpush demos with a scripted oracle (tf_agents) |
| `real_dataset_conversion.py` | Real-robot recording dir → zarr |
| `real_pusht_metrics.py`, `real_pusht_successrate.py` | HSV-mask IoU metrics over eval videos |
| `episode_lengths.py` | Mean episode duration from a replay buffer |

---

## 3. The Hydra config system

### `train.py` in full (the important 6 lines)

```python
# train.py:18
OmegaConf.register_new_resolver("eval", eval, replace=True)

# train.py:20-24 — note: NO config_name
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('diffusion_policy','config')))
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)                       # :28 — freeze ${now:} to one timestamp
    cls = hydra.utils.get_class(cfg._target_)    # :30
    workspace: BaseWorkspace = cls(cfg)          # :31
    workspace.run()                              # :32
```

Four consequences worth internalising:

1. **`--config-name` is mandatory.** There is no default. `python train.py` alone errors.
2. `config_path` is absolute, so `train.py` works from any cwd (though imports still want repo root).
3. The top-level `_target_` is **not** `instantiate`d — `get_class` + manual construction, because
   the workspace takes the whole `cfg`.
4. The `eval:` resolver permits arbitrary Python inside YAML, e.g.
   `n_action_steps: ${eval:'${n_action_steps}+${n_latency_steps}'}`.

### Invocation forms

```bash
# named config from diffusion_policy/config/, swapping the task group
python train.py --config-name=train_diffusion_unet_lowdim_workspace task=pusht_lowdim

# a downloaded flat/resolved config from the current directory
python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.seed=42 training.device=cuda:0 \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# each workspace is also directly runnable
python diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py
```

The third form works because every concrete workspace carries its own entrypoint:

```python
# train_diffusion_unet_hybrid_workspace.py:287-296
@hydra.main(version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)     # <-- config_name = module filename stem
```

That is precisely why workspace `.py` and config `.yaml` names match 1:1.

### `_target_` instantiation — where each style is used

| Config node | How it is built |
|---|---|
| `cfg._target_` (top level) | `hydra.utils.get_class` + `cls(cfg)` — `train.py:30-31`, `eval.py:33-34` |
| `cfg.policy` | `hydra.utils.instantiate(cfg.policy)` — recursive into `noise_scheduler`, `obs_encoder`, `model`, `action_ae`, `state_prior`, `rgb_model` |
| `cfg.task.dataset` | `hydra.utils.instantiate(cfg.task.dataset)` |
| `cfg.task.env_runner` | `hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)` |
| `cfg.ema` | `hydra.utils.instantiate(cfg.ema, model=self.ema_model)` |
| `cfg.optimizer` | **Either** `instantiate(cfg.optimizer, params=...)` (UNet: `_target_: torch.optim.AdamW`) **or** `self.model.get_optimizer(**cfg.optimizer)` (transformer/BET: plain kwargs, no `_target_`) |
| `cfg.dataloader`, `cfg.logging`, `cfg.checkpoint.topk` | plain `**kwargs` splat, no `_target_` |

---

## 4. The config tree

`diffusion_policy/config/` — **22 workspace YAMLs + 27 task YAMLs = 49 files.**

### Workspace configs (22)

Naming: `train_<method>_<modality>[_<qualifier>]_workspace.yaml`. Several configs share one
workspace class.

| Config | Default task | Workspace class |
|---|---|---|
| `train_diffusion_unet_hybrid_workspace` | `lift_image_abs` | `TrainDiffusionUnetHybridWorkspace` |
| `train_diffusion_unet_ddim_hybrid_workspace` | `lift_image_abs` | same (DDIM, 8 inference steps) |
| `train_diffusion_unet_real_hybrid_workspace` | `real_pusht_image` | same |
| `train_diffusion_unet_image_workspace` | `lift_image_abs` | `TrainDiffusionUnetImageWorkspace` |
| `train_diffusion_unet_image_pretrained_workspace` | `lift_image_abs` | same (ImageNet weights, frozen) |
| `train_diffusion_unet_real_{image,pretrained}_workspace` | `real_pusht_image` | same |
| `train_diffusion_unet_{,ddim_}lowdim_workspace` | `pusht_lowdim` | `TrainDiffusionUnetLowdimWorkspace` |
| `train_diffusion_transformer_hybrid_workspace` | `lift_image_abs` | `TrainDiffusionTransformerHybridWorkspace` |
| `train_diffusion_transformer_real_hybrid_workspace` | `real_pusht_image` | same |
| `train_diffusion_transformer_lowdim_workspace` | `blockpush_lowdim_seed` | `TrainDiffusionTransformerLowdimWorkspace` |
| `train_diffusion_transformer_lowdim_{pusht,kitchen}_workspace` | `pusht_lowdim` / `kitchen_lowdim_abs` | same |
| `train_diffusion_unet_video_workspace` | `lift_image_abs` | `TrainDiffusionUnetVideoWorkspace` (prototype) |
| `train_ibc_dfo_{hybrid,lowdim,real_hybrid}_workspace` | `pusht_image` / `pusht_lowdim` / `real_pusht_image` | `TrainIbcDfo*Workspace` |
| `train_bet_lowdim_workspace` | `blockpush_lowdim_seed` | `TrainBETLowdimWorkspace` |
| `train_robomimic_{image,lowdim,real_image}_workspace` | `lift_image` / `pusht_lowdim` / `real_pusht_image` | `TrainRobomimic*Workspace` |

**Anatomy of a workspace config** (canonical: `train_diffusion_unet_hybrid_workspace.yaml`, 137 lines):

1. `defaults: [_self_, task: <default>]` — **`_self_` first**, so the task group loads after and
   can override. `task=<name>` on the CLI swaps the entire subtree.
2. `name`, `_target_` (the workspace class path).
3. Interpolation shims: `task_name: ${task.name}`, and either `shape_meta: ${task.shape_meta}`
   (image) or `obs_dim/action_dim/keypoint_dim: ${task.*}` (lowdim).
4. Horizon block: `horizon: 16`, `n_obs_steps: 2`, `n_action_steps: 8`, `n_latency_steps: 0`,
   `past_action_visible: False`, `obs_as_global_cond: True`.
   *(Robomimic BC-RNN instead uses `horizon: 10`, `n_obs_steps: 1`, `n_action_steps: 1`; BET uses
   `horizon: 3`, `n_obs_steps: 3`, `n_action_steps: 1`.)*
5. `policy:` — nested `_target_` tree.
6. `ema:` — absent for IBC / BET / robomimic.
7. `dataloader:` / `val_dataloader:` — splatted into `DataLoader`. Image `bs 64 / 8 workers`,
   lowdim `bs 256 / 1 worker`.
8. `optimizer:` — see the table in §3.
9. `training:` — `device, seed, debug, resume, lr_scheduler: cosine, lr_warmup_steps,
   num_epochs (3050 image / 5000 IBC-BET), gradient_accumulate_every, use_ema,
   rollout_every: 50, checkpoint_every: 50, val_every: 1, sample_every: 5`.
10. `logging:` — splatted into `wandb.init`; `project: diffusion_policy_debug`, `mode: online`.
11. `checkpoint:` — the `topk` block + `save_last_ckpt` / `save_last_snapshot`.
12. `multi_run:` — read by `ray_train_multirun.py`.
13. `hydra:` — `run.dir: data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}`.

### Task configs (27)

| Family | Files |
|---|---|
| Robomimic | 5 tasks × {`_image`, `_image_abs`, `_lowdim`, `_lowdim_abs`} = **20** (`lift`, `can`, `square`, `transport`, `tool_hang`) |
| Push-T | `pusht_image.yaml`, `pusht_lowdim.yaml` |
| Block Pushing | `blockpush_lowdim_seed.yaml`, `blockpush_lowdim_seed_abs.yaml` |
| Franka Kitchen | `kitchen_lowdim.yaml`, `kitchen_lowdim_abs.yaml` |
| Real robot | `real_pusht_image.yaml` |

`_abs` = absolute (rather than delta) end-effector actions.

A task config contains **exactly four things** and zero training hyperparameters:

1. `name:`
2. Shape declaration — `shape_meta: &shape_meta {...}` (image) or `obs_dim`/`action_dim`/
   `keypoint_dim` scalars (lowdim), plus extras like `dataset_type: &dataset_type ph`,
   `abs_action: &abs_action True`.
3. `env_runner:` — `_target_` + `n_train: 6`, `n_train_vis: 2`, `n_test: 50`, `n_test_vis: 4`,
   `test_start_seed: 100000`, `max_steps`, `fps`, `n_envs`.
4. `dataset:` — `_target_` + `zarr_path`/`dataset_path`, `horizon: ${horizon}`,
   `pad_before: ${eval:'${n_obs_steps}-1+${n_latency_steps}'}`,
   `pad_after: ${eval:'${n_action_steps}-1'}`, `val_ratio: 0.02`.

Task YAMLs reference the *parent* config freely (`${horizon}`, `${n_obs_steps}`), use `${task.*}`
self-references, and use `${eval:'...'}` arithmetic — e.g. `square_image_abs.yaml:39`:

```yaml
max_steps: ${eval:'500 if "${task.dataset_type}" == "mh" else 400'}
```

### The `shape_meta` convention

One declaration drives the dataset, the env, the vision encoder, and the action dimension.

```yaml
# diffusion_policy/config/task/square_image_abs.yaml:3-20
shape_meta: &shape_meta
  # acceptable types: rgb, low_dim
  obs:
    agentview_image:          {shape: [3, 84, 84], type: rgb}
    robot0_eye_in_hand_image: {shape: [3, 84, 84], type: rgb}
    robot0_eef_pos:           {shape: [3]}     # type defaults to low_dim
    robot0_eef_quat:          {shape: [4]}
    robot0_gripper_qpos:      {shape: [2]}
  action:
    shape: [10]                                 # 3 pos + 6 rot6d + 1 gripper
```

Rules:

- `obs` is an **ordered mapping** of key → `{shape, type}`.
- `type` is optional and **defaults to `'low_dim'`** (every consumer uses `attr.get('type','low_dim')`).
- `shape` excludes batch and time. RGB is **channel-first `[C,H,W]`**; lowdim is rank-1.
- `action.shape` must be rank-1 — policies assert `len(action_shape) == 1`.
- Only `rgb` and `low_dim` are handled by this repo's encoders.

The same ~12-line parse loop appears in 8 places: `diffusion_unet_hybrid_image_policy.py:43-65`,
`diffusion_transformer_hybrid_image_policy.py:50-72`, `ibc_dfo_hybrid_image_policy.py:34-58`,
`robomimic_image_policy.py:23-45`, `diffusion_unet_video_policy.py:41-66`,
`multi_image_obs_encoder.py:43-117`, `robomimic_replay_image_dataset.py:93-101`,
`robomimic_image_runner.py:27-30`.

Lowdim tasks do **not** use `shape_meta` — they declare flat `obs_dim` / `action_dim` scalars.

---

## 5. Workspaces — the training loop

### `BaseWorkspace` — `diffusion_policy/workspace/base_workspace.py`

```python
class BaseWorkspace:                                     # :13
    include_keys = tuple()                               # :14
    exclude_keys = tuple()                               # :15

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str]=None)   # :17
    @property
    def output_dir(self)                                 # :23  falls back to HydraConfig runtime dir
    def run(self)                                        # :29
    def save_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, include_keys=None, use_thread=True)       # :35
    def get_checkpoint_path(self, tag='latest')          # :73
    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs)  # :76
    def load_checkpoint(self, path=None, tag='latest', ...)              # :89
    @classmethod
    def create_from_checkpoint(cls, path, ...)           # :103
    def save_snapshot(self, tag='latest')                # :117
```

**The checkpoint mechanism is reflection-based** — this is the central design idea.
`save_checkpoint` walks `self.__dict__` (`:55`):

- anything with **both** `state_dict` and `load_state_dict` (models, optimizers, LR schedulers,
  the EMA-averaged model) → `payload['state_dicts'][key]`
- anything named in `include_keys` → `dill.dumps(value)` → `payload['pickles'][key]`
- `payload['cfg'] = self.cfg` — **always**

Every concrete workspace sets `include_keys = ['global_step', 'epoch']`, so resume restores the
counters. `pickle_module=dill` throughout is what lets the whole `cfg` round-trip inside the
`.ckpt` — which is exactly how `eval.py` reconstructs a run from a checkpoint alone.

`save_snapshot` pickles the *entire workspace object*; fast, but code-version-fragile. The
docstring at `:118` says to use `save_checkpoint` for anything long-lived.

### The 11 concrete workspaces

| File | Policy | EMA | Optimizer source |
|---|---|---|---|
| `train_diffusion_unet_hybrid_workspace.py` | `DiffusionUnetHybridImagePolicy` | ✅ repo `EMAModel` | `instantiate(cfg.optimizer, params=...)` |
| `train_diffusion_unet_image_workspace.py` | `DiffusionUnetImagePolicy` | ✅ | instantiate; adds `freeze_encoder` |
| `train_diffusion_unet_lowdim_workspace.py` | `DiffusionUnetLowdimPolicy` | ✅ **diffusers** `EMAModel` | instantiate |
| `train_diffusion_transformer_hybrid_workspace.py` | `DiffusionTransformerHybridImagePolicy` | ✅ | `model.get_optimizer(**cfg.optimizer)` |
| `train_diffusion_transformer_lowdim_workspace.py` | `DiffusionTransformerLowdimPolicy` | ✅ diffusers | `model.get_optimizer(...)` |
| `train_ibc_dfo_hybrid_workspace.py` | `IbcDfoHybridImagePolicy` | ❌ | instantiate |
| `train_ibc_dfo_lowdim_workspace.py` | `IbcDfoLowdimPolicy` | ❌ | instantiate |
| `train_bet_lowdim_workspace.py` | `BETLowdimPolicy` (attr is `self.policy`, not `self.model`) | ❌ | `policy.get_optimizer(...)` |
| `train_robomimic_image_workspace.py` | `RobomimicImagePolicy` | ❌, no LR sched | robomimic-internal |
| `train_robomimic_lowdim_workspace.py` | `RobomimicLowdimPolicy` | ❌, no LR sched | robomimic-internal |
| `train_diffusion_unet_video_workspace.py` | `DiffusionUnetVideoPolicy` | ✅ | instantiate — **prototype**: no JsonLogger, no val loop, no TopK |

These files are deliberate near-copies of one another (`O(N+M)`, not `O(N*M)`, per the README's
codebase tutorial). **Diffing two of them is the fastest way to see what actually varies.**

### The canonical `run()` loop

Reference: `train_diffusion_unet_hybrid_workspace.py:61-285`.

**Constructor (`:37-59`)** — seed `torch`/`np`/`random`; `self.model = instantiate(cfg.policy)`;
`self.ema_model = copy.deepcopy(self.model)` if `use_ema`; optimizer; `global_step = epoch = 0`.

**Setup inside `run()`:**

| Step | Lines | What |
|---|---|---|
| 1 | `:62` | `cfg = copy.deepcopy(self.cfg)` so debug mutations don't reach the saved cfg |
| 2 | `:65-69` | Resume from `checkpoints/latest.ckpt` if `cfg.training.resume` |
| 3 | `:73-80` | `instantiate(cfg.task.dataset)`, assert `isinstance(dataset, BaseImageDataset)`, build train + val `DataLoader`s |
| 4 | `:76,82-84` | `normalizer = dataset.get_normalizer()` → `model.set_normalizer()` **and** `ema_model.set_normalizer()` |
| 5 | `:87-97` | LR scheduler (see below) |
| 6 | `:100-104` | `ema = instantiate(cfg.ema, model=self.ema_model)` |
| 7 | `:108-111` | `env_runner = instantiate(cfg.task.env_runner, output_dir=self.output_dir)` |
| 8 | `:114-123` | `wandb.init(dir=output_dir, config=OmegaConf.to_container(cfg, resolve=True), **cfg.logging)` |
| 9 | `:126-129` | `TopKCheckpointManager(save_dir=.../checkpoints, **cfg.checkpoint.topk)` |
| 10 | `:132-136` | model / ema_model / `optimizer_to(optimizer, device)` |
| 11 | `:141-148` | If `cfg.training.debug`: `num_epochs=2`, `max_train_steps=3`, `max_val_steps=3`, all cadences → 1 |

> **Step 4 is load-bearing for anyone writing a new policy.** The normalizer is copied *into* the
> policy and therefore lives inside the policy's `state_dict`. Normalization statistics travel with
> the checkpoint; there is no separate normalizer file.

```python
# :87-97 — note the per-batch stepping and the resume-aware last_epoch
lr_scheduler = get_scheduler(
    cfg.training.lr_scheduler, optimizer=self.optimizer,
    num_warmup_steps=cfg.training.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) \
                       // cfg.training.gradient_accumulate_every,
    last_epoch=self.global_step-1)
```

**Epoch loop (`:152-285`)**, wrapped in `with JsonLogger(output_dir/'logs.json.txt') as json_logger:`

*Train phase (`:157-200`)* — per batch: move to device; cache the first batch as
`train_sampling_batch`; `raw_loss = self.model.compute_loss(batch)`;
`loss = raw_loss / gradient_accumulate_every`; `backward()`; on the accumulation boundary
`optimizer.step()`, `zero_grad()`, `lr_scheduler.step()`; then `ema.step(self.model)`.
`step_log = {train_loss, global_step, epoch, lr}` is logged **every batch except the last** — the
last batch's log is merged with validation and rollout results so everything lands on one wandb step.

*Eval phase (`:207-277`)*, with `policy = self.ema_model if use_ema else self.model`, `policy.eval()`:

| Trigger | Action |
|---|---|
| `epoch % rollout_every == 0` | `runner_log = env_runner.run(policy)`; `step_log.update(runner_log)` |
| `epoch % val_every == 0` | `no_grad` loop of `self.model.compute_loss(batch)` → `step_log['val_loss']` |
| `epoch % sample_every == 0` | `policy.predict_action(batch['obs'])` on the cached batch → `train_action_mse_error = MSE(result['action_pred'], batch['action'])` |
| `epoch % checkpoint_every == 0` | `save_checkpoint()` → `latest.ckpt`; sanitize metric keys `/`→`_`; `topk_manager.get_ckpt_path(metric_dict)` decides on a second named checkpoint |

Then `policy.train()`, final `wandb_run.log(step_log, step=self.global_step)`, `global_step += 1`,
`epoch += 1`.

> Note the asymmetry: **validation loss uses `self.model`, while rollouts and the sampling
> diagnostic use the EMA model.** And the sampling diagnostic requires the policy to return
> `'action_pred'`, not just `'action'`.

### Variants worth knowing

- **Transformer workspaces** use `self.model.get_optimizer(**cfg.optimizer)` so the minGPT
  decay/no-decay parameter groups are honoured — hence no `_target_` under `cfg.optimizer`.
- **`train_diffusion_unet_image_workspace.py:156-159`** inserts at the top of each epoch:
  ```python
  if cfg.training.freeze_encoder:
      self.model.obs_encoder.eval()
      self.model.obs_encoder.requires_grad_(False)
  ```
  This is the pretrained-ResNet / R3M path.
- **BET** (`train_bet_lowdim_workspace.py:62-140`) adds a K-Means fitting step before training:
  `self.policy.fit_action_ae(normalizer['action'].normalize(dataset.get_all_actions()))`, supports
  `cfg.training.enable_normalizer=False`, uses `grad_norm_clip`, and its `compute_loss` returns
  `(loss, loss_components)`.
- **Robomimic workspaces** own no optimizer at all:
  ```python
  info = self.model.train_on_batch(batch, epoch=self.epoch)
  loss_cpu = info['losses']['action_loss'].item()
  ```
  robomimic's `PolicyAlgo` does the backward pass internally.

### `TopKCheckpointManager` — `diffusion_policy/common/checkpoint_util.py`

```python
def __init__(self, save_dir, monitor_key: str, mode='min', k=1,
             format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt')   # :5
def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]                # :22
```

Pure bookkeeping: maintains `path_value_map`, returns a formatted path when the candidate beats the
worst retained one (deleting the displaced file at `:57-58`), returns `None` when `k == 0` or the
candidate doesn't qualify. All shipped configs use:

```yaml
checkpoint:
  topk:
    monitor_key: test_mean_score        # note: underscore, not slash — see §10.2
    mode: max
    k: 5
    format_str: 'epoch={epoch:04d}-test_mean_score={test_mean_score:.3f}.ckpt'
  save_last_ckpt: True
  save_last_snapshot: False
```

### `eval.py`

```python
payload = torch.load(open(checkpoint,'rb'), pickle_module=dill)   # :31
cfg = payload['cfg']                                              # :32
cls = hydra.utils.get_class(cfg._target_)                         # :33
workspace = cls(cfg, output_dir=output_dir)                       # :34
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
policy = workspace.ema_model if cfg.training.use_ema else workspace.model   # :39-41
env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)  # :48
runner_log = env_runner.run(policy)
```

The config rides inside the checkpoint, so eval needs no YAML. `wandb.Video` values are replaced by
their `_path` and everything is dumped to `<output_dir>/eval_log.json`.

> **This is not the entrypoint for an external policy** — it reconstructs a *workspace class named
> by the checkpoint*. See §11 for the seam you actually want.

---

## 6. Policies

### The base classes, verbatim

```python
# diffusion_policy/policy/base_image_policy.py
class BaseImagePolicy(ModuleAttrMixin):                       # :7
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:  # :10
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    def reset(self):                                          # :19  (stateful policies only)
        pass

    def set_normalizer(self, normalizer: LinearNormalizer):   # :24
        raise NotImplementedError()
```

```python
# diffusion_policy/policy/base_lowdim_policy.py
class BaseLowdimPolicy(ModuleAttrMixin):                      # :7
    def predict_action(self, obs_dict) -> Dict[str, torch.Tensor]:   # :10
        """
        obs_dict:
            obs: B,To,Do
        return:
            action: B,Ta,Da
        """
```

Both inherit `ModuleAttrMixin` (`model/common/module_attr_mixin.py:3`) — an `nn.Module` carrying a
`_dummy_variable = nn.Parameter()` so `.device` and `.dtype` always work, even with no real params.

### The contract in practice

| Aspect | Image policy | Lowdim policy |
|---|---|---|
| `obs_dict` | flat dict keyed by obs name, each `(B, To, *shape)` | single key `'obs'`, `(B, To, Do)` |
| RGB format | `(B, To, 3, H, W)` float32 in `[0,1]`, **channel-first** | — |
| Return | `{'action': (B, Ta, Da)}` | same |
| Also returned by diffusion policies | `'action_pred'` `(B, T, Da)` — the **full** horizon; required by the workspace sampling diagnostic | same, plus `'obs_pred'` / `'action_obs_pred'` in inpainting mode (`diffusion_unet_lowdim_policy.py:171-176`) |
| `past_action` | plumbed by the runners, but every policy asserts it is absent | same |
| `compute_loss(batch)` | takes `{'obs':…, 'action':…}` → scalar; asserts `'valid_mask' not in batch` | same |
| `set_normalizer` | uniformly `self.normalizer.load_state_dict(normalizer.state_dict())` | same |

### The 11 concrete policies

| File | Class | Summary |
|---|---|---|
| `diffusion_unet_lowdim_policy.py:13` | `DiffusionUnetLowdimPolicy` | DDPM over action (or action⊕obs) trajectories with a config-injected `ConditionalUnet1D`; supports `obs_as_local_cond` / `obs_as_global_cond` / pure inpainting, `pred_action_steps_only`, `oa_step_convention` |
| `diffusion_unet_image_policy.py:15` | `DiffusionUnetImagePolicy` | Same UNet, obs via an injected `MultiImageObsEncoder`. The end-to-end ResNet18+GroupNorm variant; what `freeze_encoder` targets |
| `diffusion_unet_hybrid_image_policy.py:22` | `DiffusionUnetHybridImagePolicy` | **The headline model.** Builds its vision encoder by instantiating a robomimic `bc_rnn` algo and lifting `policy.nets['policy'].nets['encoder'].nets['obs']` (`:102`), optionally swapping BatchNorm→GroupNorm and robomimic's `CropRandomizer` for the repo's; then `ConditionalUnet1D` |
| `diffusion_transformer_lowdim_policy.py:13` | `DiffusionTransformerLowdimPolicy` | DDPM with `TransformerForDiffusion` as denoiser; `obs_as_cond` selects cross-attention vs inpainting |
| `diffusion_transformer_hybrid_image_policy.py:22` | `DiffusionTransformerHybridImagePolicy` | Robomimic encoder + transformer; `get_optimizer` (`:296`) merges minGPT decay groups with a separate `obs_encoder` group |
| `diffusion_unet_video_policy.py:17` | `DiffusionUnetVideoPolicy` | Experimental: per-RGB-key video nets `(B,T,C,H,W)→(B,Do)` + a `TemporalAggregator` |
| `ibc_dfo_lowdim_policy.py:8` | `IbcDfoLowdimPolicy` | Implicit BC baseline: 5×1024 MLP energy `E(obs, action)`; inference runs `pred_n_iter` rounds of sample→softmax→resample→jitter over `pred_n_samples` candidates. Returns only `'action'` |
| `ibc_dfo_hybrid_image_policy.py:16` | `IbcDfoHybridImagePolicy` | Same, with the robomimic vision encoder |
| `bet_lowdim_policy.py:13` | `BETLowdimPolicy` | Behavior Transformer: `KMeansDiscretizer` action AE + minGPT prior; pads obs beyond `n_obs_steps` with sentinel `-2`; extra `fit_action_ae` / `get_latents` |
| `robomimic_image_policy.py:12` | `RobomimicImagePolicy` | Thin adapter over a robomimic `PolicyAlgo` (default `bc_rnn`); slices `x[:,0,...]`, returns `(B,1,Da)` |
| `robomimic_lowdim_policy.py:11` | `RobomimicLowdimPolicy` | Same for lowdim; flattens into a single robomimic obs key literally named `'obs'` |

### Diffusion internals — the pattern shared by every diffusion variant

```python
# conditional_sample, e.g. diffusion_unet_hybrid_image_policy.py:175-212
trajectory = torch.randn(size=condition_data.shape, ...)
scheduler.set_timesteps(self.num_inference_steps)
for t in scheduler.timesteps:
    trajectory[condition_mask] = condition_data[condition_mask]   # 1. apply conditioning (inpaint)
    model_output = model(trajectory, t, local_cond=..., global_cond=...)
    trajectory = scheduler.step(model_output, t, trajectory, generator=generator).prev_sample
trajectory[condition_mask] = condition_data[condition_mask]
```

Receding-horizon action slice (`:270-272`):

```python
start = To - 1
end = start + self.n_action_steps
action = action_pred[:, start:end]
```

*(The lowdim policy uses `start = To`, or `To-1` when `oa_step_convention=True` —
`diffusion_unet_lowdim_policy.py:161-164`.)*

`compute_loss` (`:284-351`): normalize → build `global_cond` from `obs_encoder(this_nobs)` reshaped
`(B, To*Do)` → `condition_mask = self.mask_generator(trajectory.shape)` → sample `noise` and
`timesteps ~ U[0, num_train_timesteps)` → `add_noise` → apply conditioning → predict → target is
`noise` for `prediction_type == 'epsilon'` or `trajectory` for `'sample'` → masked MSE:

```python
loss = F.mse_loss(pred, target, reduction='none')
loss = loss * loss_mask.type(loss.dtype)
loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
```

One subtlety: in global-cond mode `compute_loss` encodes only `x[:, :self.n_obs_steps]`, but in
inpainting mode it encodes **all** `T` steps (`:299-309`).

---

## 7. Model architectures

### `model/diffusion/`

| File | Key contents |
|---|---|
| `conditional_unet1d.py` | `ConditionalResidualBlock1D` (`:14`), `ConditionalUnet1D` (`:69`) |
| `transformer_for_diffusion.py` | `TransformerForDiffusion` (`:10`) |
| `conv1d_components.py` | `Downsample1d`, `Upsample1d`, `Conv1dBlock` (Conv1d→GroupNorm→Mish) |
| `positional_embedding.py` | `SinusoidalPosEmb` |
| `ema_model.py` | `EMAModel` (`:5`) |
| `mask_generator.py` | `LowdimMaskGenerator` (`:43`), `KeypointMaskGenerator` (`:107`), `DummyMaskGenerator` (`:32`) |

**`ConditionalUnet1D`** (`:70-78`):

```python
def __init__(self, input_dim, local_cond_dim=None, global_cond_dim=None,
    diffusion_step_embed_dim=256, down_dims=[256,512,1024],
    kernel_size=3, n_groups=8, cond_predict_scale=False)
```

- Diffusion-step encoder: `SinusoidalPosEmb → Linear(d,4d) → Mish → Linear(4d,d)`.
- `cond_dim = diffusion_step_embed_dim + global_cond_dim` — the global condition is **concatenated
  to the time embedding** and injected by FiLM in every residual block. With
  `cond_predict_scale=True` the block predicts both scale and bias: `out = scale*out + bias` (`:60-61`).
- `forward(sample, timestep, local_cond=None, global_cond=None)` (`:173-180`):
  `x: (B,T,input_dim)`, `global_cond: (B,global_cond_dim)`, output `(B,T,input_dim)`. Internally
  `einops.rearrange(sample, 'b h t -> b t h')` for Conv1d, and back at the end.

**`TransformerForDiffusion`** (`:11-26`):

```python
def __init__(self, input_dim, output_dim, horizon, n_obs_steps=None, cond_dim=0,
        n_layer=12, n_head=12, n_emb=768, p_drop_emb=0.1, p_drop_attn=0.1,
        causal_attn=False, time_as_cond=True, obs_as_cond=False, n_cond_layers=0)
```

- Token budget: `T = horizon` (`+1` if `not time_as_cond`); `T_cond = 1` (`+ n_obs_steps` if
  `obs_as_cond`). `obs_as_cond` is recomputed as `cond_dim > 0` at `:37`.
- Two architectures: encoder-decoder (`nn.TransformerDecoder` over the trajectory, memory = time +
  obs tokens) or encoder-only BERT when `T_cond == 0`.
- All layers use `norm_first=True` — commented "important for stability" (`:86`).
- `get_optim_groups(weight_decay)` (`:197`) implements the minGPT split (biases, LayerNorm,
  `pos_emb` / `cond_pos_emb` excluded from weight decay).

**`EMAModel`** (`ema_model.py:5`, `__init__` at `:10-18`):

```python
def __init__(self, model, update_after_step=0, inv_gamma=1.0,
             power=2/3, min_value=0.0, max_value=0.9999)
```

`get_decay(step)` (`:44`) = `1 - (1 + step/inv_gamma) ** -power`, clamped, zero before
`update_after_step`. `step(new_model)` (`:57`) walks `zip(new_model.modules(),
self.averaged_model.modules())` over immediate params only, with two exceptions: `_BatchNorm`
modules and `requires_grad=False` params are **copied, not averaged** (`:77-81`). Configs
universally set `power: 0.75`.

> Two different `EMAModel`s are in play: this one (used by the hybrid/image/transformer-hybrid
> workspaces and named by every `cfg.ema._target_`) and `diffusers.training_utils.EMAModel`
> (imported directly by the two lowdim workspaces).

### `model/vision/`

**`MultiImageObsEncoder`** (`multi_image_obs_encoder.py:12-29`):

```python
def __init__(self, shape_meta, rgb_model, resize_shape=None, crop_shape=None,
        random_crop=True, use_group_norm=False, share_rgb_model=False, imagenet_norm=False)
    """ Assumes rgb input: B,C,H,W ; Assumes low_dim input: B,D """
```

Per RGB key it composes `nn.Sequential(resizer, randomizer, normalizer)` where the randomizer is
`CropRandomizer` (train) or `CenterCrop`, and the normalizer is ImageNet `Normalize` when
`imagenet_norm`. `use_group_norm` runs
`replace_submodules(BatchNorm2d → GroupNorm(num_groups=C//16, num_channels=C))`.
`rgb_keys` / `low_dim_keys` are **sorted** (`:116-117`) so concatenation order is deterministic.
`output_shape()` (`:181-195`) runs a `torch.zeros` dummy forward to discover `obs_feature_dim` —
every policy calls this at construction.

> ⚠️ See §10.1 — `random_crop=False` silently applies **no crop at all**.

`crop_randomizer.py` is a TorchScript-friendly reimplementation of robomimic's, so `eval_fixed_crop`
gives deterministic centre crops at eval time. `model_getter.py:4` has
`get_resnet(name, weights=None)` → torchvision resnet with `fc = Identity`; `weights="r3m"` routes
to `get_r3m` (`:18`).

### `model/common/`

| File | Key contents |
|---|---|
| `normalizer.py` | `LinearNormalizer` (`:12`), `SingleFieldLinearNormalizer` (`:101`), `_fit` (`:182`) |
| `dict_of_tensor_mixin.py` | `DictOfTensorMixin` — nested `nn.ParameterDict` that survives `state_dict` with dynamic keys |
| `lr_scheduler.py` | `get_scheduler(...)` (`:6`) — diffusers' function plus `**kwargs` passthrough (needed for `last_epoch`) |
| `module_attr_mixin.py` | `ModuleAttrMixin` with `.device` / `.dtype` |
| `rotation_transformer.py` | `RotationTransformer` (`:7`) — **the only importer of `pytorch3d`** |
| `shape_util.py` | `get_output_shape(input_shape, net)` |
| `tensor_util.py` | Vendored robomimic tensor-tree helpers (960 lines) |

**`LinearNormalizer`** — `avaliable_modes = ['limits','gaussian']` *(sic)*.
`fit(data, last_n_dims=1, mode='limits', output_max=1., output_min=-1., range_eps=1e-4,
fit_offset=True)` (`:16`) takes a dict (fits per key) or a bare array (key `'_default'`).
`_fit` (`:182`) stores `scale`, `offset`, and `input_stats {min,max,mean,std}` as non-trainable
`nn.ParameterDict`s — **this is why normalization travels inside the policy's `state_dict`.**

Helpers in `common/normalize_util.py`: `get_image_range_normalizer` (`:23`, the fixed `[0,1]→[-1,1]`
image normalizer), `get_range_normalizer_from_stat`, `get_identity_normalizer_from_stat`,
`robomimic_abs_action_only_normalizer_from_stat` (`:110`, splits pos / rot6d / gripper and
normalizes each differently), `array_to_stats` (`:216`).

**`RotationTransformer`** (`rotation_transformer.py:7`): `valid_reps = ['axis_angle',
'euler_angles', 'quaternion', 'rotation_6d', 'matrix']`, always converting through `matrix`, built
on `pytorch3d.transforms`. This is what converts robomimic absolute actions from axis-angle to
rotation-6d — hence `*_abs` tasks declaring `action.shape: [10]` (3+6+1) or `[20]` for dual-arm.

**Mask generators.** `LowdimMaskGenerator(action_dim, obs_dim, max_n_obs_steps=2,
fix_obs_steps=True, action_visible=False)` (`:44`); `forward(shape)` (`:60`) asserts
`D == action_dim + obs_dim` and returns a `(B,T,D)` bool mask that is `True` where the value is
**given** (observed → inpainted → excluded from the loss). Diffusion policies pass `obs_dim=0` in
global-cond mode (mask all-`False`, whole trajectory denoised) and `obs_dim=obs_feature_dim` in
inpainting mode.

### `model/bet/`

Vendored from `notmahi/bet`: `action_ae/discretizers/k_means.py` (`KMeansDiscretizer`, `:10`),
`latent_generators/mingpt.py` (`MinGPT`, `:14`), `libraries/mingpt/`, `libraries/loss_fn.py`
(`FocalLoss`).

---

## 8. Datasets and the replay buffer

### The interface — `diffusion_policy/dataset/base_dataset.py`

```python
class BaseImageDataset(torch.utils.data.Dataset):             # :30
    def get_validation_dataset(self) -> 'BaseLowdimDataset'   # :31  default: empty dataset
    def get_normalizer(self, **kwargs) -> LinearNormalizer    # :35  NotImplementedError
    def get_all_actions(self) -> torch.Tensor                 # :38  NotImplementedError
    def __len__(self) -> int                                  # :41
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]     # :44
        """ output: obs: {key: T, *} ; action: T, Da """
```

`BaseLowdimDataset` (`:7`) is identical with `obs: (T, Do)`.

### The 9 concrete datasets

| File | Class | Source | Sample |
|---|---|---|---|
| `pusht_dataset.py:12` | `PushTLowdimDataset` | zarr `keypoint`,`state`,`action` | `{'obs': (T,20), 'action': (T,2)}` |
| `pusht_image_dataset.py:13` | `PushTImageDataset` | zarr `img`,`state`,`action` | `{'obs': {'image': (T,3,96,96), 'agent_pos': (T,2)}, 'action': (T,2)}` |
| `blockpush_lowdim_dataset.py:11` | `BlockPushLowdimDataset` | zarr `obs`,`action` | `{'obs': (T,16), 'action': (T,2)}` |
| `kitchen_lowdim_dataset.py:12` | `KitchenLowdimDataset` | `observations_seq.npy` + `actions_seq.npy` + `existence_mask.npy` | `{'obs': (T,60), 'action': (T,9)}` |
| `kitchen_mjl_lowdim_dataset.py:14` | `KitchenMjlLowdimDataset` | raw `*/*.mjl` MuJoCo logs | same; requires `abs_action=True`; injects `robot_noise_ratio` proprio noise |
| `robomimic_replay_lowdim_dataset.py:21` | `RobomimicReplayLowdimDataset` | hdf5 `data/demo_i/obs/*` | `{'obs': (T,Do), 'action': (T,Da)}` |
| `robomimic_replay_image_dataset.py:34` | `RobomimicReplayImageDataset` | hdf5 → zarr with `.hdf5.zarr.zip` cache, Jpeg2k codec | `{'obs': {rgb: (To,3,H,W), lowdim: (To,D)}, 'action': (T,Da)}` |
| `real_pusht_image_dataset.py:28` | `RealPushTImageDataset` | directory of real demos → zarr keyed by md5(shape_meta) | `{'obs': {camera_i: (To,3,240,320), 'robot_eef_pose': (To,2)}, 'action': (T,2)}` |
| `mujoco_image_dataset.py:13` | `MujocoImageDataset` | fork-only, orphaned — see §10.13 | `{'obs': {'image': (T,3,224,224), 'agent_pos': (T,8)}, 'action': (T,8)}` |

### The shared 6-step construction pattern

Every dataset is ~90 lines of the same shape. Copy this for a new task:

```python
# 1. load
self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=['img','state','action'])
# 2. split episodes
val_mask   = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
train_mask = downsample_mask(mask=~val_mask, max_n=max_train_episodes, seed=seed)
# 3. window sampler
self.sampler = SequenceSampler(replay_buffer=self.replay_buffer, sequence_length=horizon,
                               pad_before=pad_before, pad_after=pad_after, episode_mask=train_mask)
# 4. validation split = shallow copy sharing the ReplayBuffer, inverted mask
# 5. get_normalizer(mode='limits') -> LinearNormalizer().fit(...); normalizer['image'] = get_image_range_normalizer()
# 6. __getitem__ -> sampler.sample_sequence(idx) -> reshape/scale -> dict_apply(data, torch.from_numpy)
```

Image-specific efficiency: `robomimic_replay_image_dataset.py:104-108` sets
`key_first_k[key] = n_obs_steps` so the sampler only materializes the first `To` frames, and
`__getitem__` (`:193`) opens with `threadpool_limits(1)` then slices before the `/255.` conversion.

### `ReplayBuffer` — `diffusion_policy/common/replay_buffer.py:84`

Zarr-backed, **time-major**, two backends (zarr group or plain numpy dict). On-disk layout:

```
data/pusht/pusht_cchi_v7_replay.zarr
 ├── data
 │   ├── action   (25650, 2)      float32
 │   ├── img      (25650, 96, 96, 3) float32
 │   ├── keypoint (25650, 9, 2)   float32
 │   ├── n_contacts (25650, 1)    float32
 │   └── state    (25650, 5)      float32
 └── meta
     └── episode_ends (206,)      int64
```

Episodes are delimited purely by the cumulative `episode_ends` array. Key methods:
`copy_from_path` (`:211`), `save_to_path` (`:281`), `n_steps` (`:423`), `n_episodes` (`:429`),
`episode_lengths` (`:439`), `add_episode` (`:445`), `get_episode` (`:532`), and the chunking
heuristic `get_optimal_chunks(shape, dtype, target_chunk_bytes=2e6)` (`:48`).

### `SequenceSampler` — `diffusion_policy/common/sampler.py:77`

```python
def __init__(self, replay_buffer, sequence_length, pad_before=0, pad_after=0,
             keys=None, key_first_k=dict(), episode_mask=None)
```

`create_indices(...)` (`:7`, `@numba.jit(nopython=True)`) precomputes an `(N,4)` int array of
`(buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)`. `pad_before`/`pad_after`
are clamped to `[0, sequence_length-1]`.

`sample_sequence(idx)` (`:121`) slices each key and, when a window runs off an episode boundary,
**edge-pads by repeating the first/last available frame** (`data[:sample_start_idx] = sample[0]`).
Episodes never bleed into each other. `key_first_k` fills the unused tail with `np.nan`
deliberately, "to catch bugs".

Config convention: `pad_before = n_obs_steps - 1 (+ n_latency_steps)`, `pad_after = n_action_steps - 1`.

---

## 9. Environments and the evaluation harness

### 9.1 The environments — `diffusion_policy/env/`

All are **`gym==0.21.0` style**: `reset() -> obs` (no `info`, no `seed=` kwarg),
`step(a) -> (obs, reward, done, info)` — a **4-tuple**, not the gymnasium 5-tuple.

**Push-T** — `env/pusht/`, backend **pymunk + pygame** (2D, no OpenGL).

| Class | Obs space | Action space |
|---|---|---|
| `pusht_env.py:28` `PushTEnv` | `Box(5,)` — `[agent_x, agent_y, block_x, block_y, block_angle]`, bounds `[0…512, 0…2π]` | `Box(2,)`, `[0,0]`→`[512,512]` |
| `pusht_image_env.py:6` `PushTImageEnv` | `Dict{'image': Box(3,96,96) f32 [0,1], 'agent_pos': Box(2,) [0,512]}` | inherited `Box(2,)` |
| `pusht_keypoints_env.py:7` `PushTKeypointsEnv` | `Box(40,)` = (9 block kps ×2 + agent xy) ⧺ visibility mask | `Box(2,)` |

Control is a **positional target with a PD controller**: `sim_hz=100`, `control_hz=10`, so 10
pymunk substeps per `step()`, `k_p, k_v = 100, 20` (`:45`). The action is an absolute XY goal in
pixel coordinates. Reward (`:124-133`): `coverage = intersection_area / goal_area` between the
T-block and its goal pose, `reward = clip(coverage/0.95, 0, 1)`, `done = coverage > 0.95`. So
`test/mean_score ∈ [0,1]` is a **coverage score, not a binary success rate**. `legacy=True` (set by
every test config as `legacy_test: True`) changes state-setting order for compatibility with the
released zarr.

**Block Pushing** — `env/block_pushing/`, backend **pybullet** (xArm). `block_pushing.py:179`
`BlockPush(gym.Env)`; the runner actually uses `block_pushing_multimodal.py:103`
`BlockPushMultimodal`. Action `Box(-0.1, 0.1, (2,))` — **delta XY** of the end effector. Obs dict
with 2 blocks + 2 targets = **16 dims flattened** (via `gym.wrappers.FlattenObservation`).
`done=True` when `reward >= 0.5`. Assets (`*.urdf`) are vendored in `env/block_pushing/assets/`.

**Franka Kitchen** — `env/kitchen/`, backend **MuJoCo via `free-mujoco-py` / adept_envs**.
`base.py:34` `KitchenBase`; 7 subtasks with fixed obs indices (`OBS_ELEMENT_INDICES`, `:12`):
bottom burner, top burner, light switch, slide cabinet, hinge cabinet, microwave, kettle.
`BONUS_THRESH = 0.3`; `+1` per newly completed element. **obs_dim 60, action_dim 9** (7 arm joints
+ 2 gripper). `max_episode_steps=280`. MuJoCo XMLs are vendored under
`env/kitchen/relay_policy_learning/adept_models/`.

**Robomimic** — `env/robomimic/` contains *no simulator*; these are gym adapters over
`robomimic.envs.env_robosuite.EnvRobosuite` (MuJoCo via robosuite).

- `robomimic_image_wrapper.py:8` `RobomimicImageWrapper(env, shape_meta, init_state=None,
  render_obs_key='agentview_image')`. Action space is always `Box(-1, 1, ...)` from
  `shape_meta['action']['shape']`. **The observation space is built by matching key *suffixes*** —
  see §10.9.
- `robomimic_lowdim_wrapper.py:7` concatenates
  `['object','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']` into one flat `Box`.

### 9.2 The runner interface

`diffusion_policy/env_runner/base_image_runner.py` — the file in full:

```python
class BaseImageRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BaseImagePolicy) -> Dict:
        raise NotImplementedError()
```

`base_lowdim_runner.py` is identical with `BaseLowdimPolicy`. **That is the entire contract:
construct with `output_dir`, call `run(policy)`, receive a flat `Dict` of wandb-loggable values.**

The 9 runner files:

| File | Class |
|---|---|
| `pusht_image_runner.py:20` | `PushTImageRunner(BaseImageRunner)` |
| `pusht_keypoints_runner.py:20` | `PushTKeypointsRunner(BaseLowdimRunner)` |
| `blockpush_lowdim_runner.py:21` | `BlockPushLowdimRunner(BaseLowdimRunner)` |
| `kitchen_lowdim_runner.py:25` | `KitchenLowdimRunner(BaseLowdimRunner)` |
| `robomimic_image_runner.py:42` | `RobomimicImageRunner(BaseImageRunner)` |
| `robomimic_lowdim_runner.py:42` | `RobomimicLowdimRunner(BaseLowdimRunner)` |
| `real_pusht_image_runner.py:4` | `RealPushTImageRunner` — **stub, `run()` returns `dict()`** |
| `base_image_runner.py`, `base_lowdim_runner.py` | the two bases |

### 9.3 What a runner actually does

Taking `PushTImageRunner` (`:21`) as the reference:

```python
def __init__(self, output_dir, n_train=10, n_train_vis=3, train_start_seed=0,
        n_test=22, n_test_vis=6, legacy_test=False, test_start_seed=10000,
        max_steps=200, n_obs_steps=8, n_action_steps=8, fps=10, crf=22,
        render_size=96, past_action=False, tqdm_interval_sec=5.0, n_envs=None)
```

1. **Parallelism.** `if n_envs is None: n_envs = n_train + n_test` (`:41-42`). With
   `pusht_image.yaml` (`n_train: 6`, `n_test: 50`, `n_envs: null`) that is **56 forked processes**.
2. **Wrapper stack** (`:45-66`), per env:
   `MultiStepWrapper(VideoRecordingWrapper(PushTImageEnv(...)), n_obs_steps, n_action_steps, max_episode_steps=max_steps)`
3. **Seed split.** Train seeds `train_start_seed + i`, prefix `train/`; test seeds
   `test_start_seed + i`, prefix `test/`. Shipped config: train `0..5`, test `100000..100049`.
   Per-env setup closures are **`dill.dumps`'d** and executed inside each worker via
   `env.call_each('run_dill_function', args_list=...)` (`:173`).
4. **Video.** Only the first `n_train_vis` / `n_test_vis` envs get
   `env.env.file_path = <output_dir>/media/<wandb_id>.mp4`; the rest keep `file_path=None`, which
   disables recording entirely.
5. **`run(policy)`** (`:145`) — chunk over `ceil(n_inits / n_envs)`, then per chunk:

```python
obs = env.reset(); policy.reset()
while not done:
    np_obs_dict = dict(obs)                       # keys 'image','agent_pos'; each (n_envs, To, ...)
    obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device))
    with torch.no_grad():
        action_dict = policy.predict_action(obs_dict)
    action = np_action_dict['action']             # (n_envs, Ta, Da)
    obs, reward, done, info = env.step(action)
    done = np.all(done)
all_video_paths = env.render()                    # mp4 paths, not pixels
all_rewards     = env.call('get_attr', 'reward')  # per-substep reward lists
```

`RobomimicImageRunner` (`:47`) adds four things: it reads `env_meta` and per-demo init states from
the hdf5 **inside `__init__`** (`:79`, `:166`); train inits are **demo indices**, test inits are
**seeds**; `abs_action` installs a `RotationTransformer('axis_angle','rotation_6d')` and
`undo_transform_action` (`:356`) converts the policy's 10-d output back to 7-d before stepping; and
it passes a `dummy_env_fn` (`:126`) with rendering disabled so no OpenGL context exists in the
parent process (see §10.7).

### 9.4 The wrappers — `diffusion_policy/gym_util/`

**`MultiStepWrapper`** (`multistep_wrapper.py:67`) is what makes the env speak in *sequences*.
`repeated_space(space, n)` (`:18`) turns `Box(shape)` into `Box((n,)+shape)` and recurses through
`spaces.Dict` — **only `Box` and `Dict` are supported**. `step(action)` (`:101`) iterates
`for act in action:`, calling the inner env once per element and breaking early on termination,
then returns `(stacked_obs, aggregate(reward,'max'), aggregate(done,'max'), last_n_info)`.

Observation stacking is `stack_last_n_obs` (`:54`): fill with the last `min(n_steps, len(all_obs))`
observations and **pad the beginning by repeating the earliest frame**. That is how `To=2` works on
the very first step of an episode.

**`VideoRecordingWrapper`** (`video_recording_wrapper.py:5`) — "when `file_path` is None, don't
record"; recording is toggled per-episode by the runner mutating `env.env.file_path`. Note that
**`render()` returns `self.file_path` (a string), not pixels** — hence `env.render()` in the runners
yields a list of mp4 paths.

**`AsyncVectorEnv`** (`async_vector_env.py:43`) is a fork of `gym.vector.AsyncVectorEnv` with three
additions:

1. `dummy_env_fn` (`:97-102`) — used only to probe `metadata` / spaces in the parent. The code
   comment: *"Added dummy_env_fn to fix OpenGL error in Mujoco — disable any OpenGL rendering in
   dummy_env_fn, since it will conflict with OpenGL context in the forked child process."*
2. `call_each(name, args_list=...)` (`:475`) — sends a *different* payload to each worker. This is
   the per-env seeding and video mechanism.
3. `render(*args)` (`:557`) = `self.call('render', ...)`.

**`SyncVectorEnv`** (`sync_vector_env.py:11`) mirrors the same API in-process. **Every runner has a
commented-out `# env = SyncVectorEnv(env_fns)` line next to the Async construction** — that is the
single-process debugging switch you want while bringing up a new policy.

### 9.5 The 12 benchmark tasks

| # | Task | Modality | Simulator | Obs | Action | Runner | Dataset path |
|---|---|---|---|---|---|---|---|
| 1 | Push-T | image | pymunk/pygame | `image (3,96,96)` + `agent_pos (2)` | 2 (abs XY) | `PushTImageRunner` | `data/pusht/pusht_cchi_v7_replay.zarr` |
| 2 | Push-T | lowdim/keypoints | pymunk/pygame | 20 | 2 | `PushTKeypointsRunner` | same |
| 3 | Lift | image / lowdim (+`_abs`) | robosuite/MuJoCo | 2×`(3,84,84)` + 9 lowdim | 7 (delta) / 10 (abs) | `RobomimicImageRunner` | `data/robomimic/datasets/lift/ph/image.hdf5` |
| 4 | Can | image / lowdim (+`_abs`) | robosuite/MuJoCo | as above | 7 / 10 | `RobomimicImageRunner` | `…/can/{ph,mh}/…` |
| 5 | Square | image / lowdim (+`_abs`) | robosuite/MuJoCo | as above | 7 / 10 | `RobomimicImageRunner` | `…/square/{ph,mh}/…` |
| 6 | Transport | image / lowdim (+`_abs`) | robosuite/MuJoCo | dual-arm | 14 / 20 | `RobomimicImageRunner` | `…/transport/{ph,mh}/…` |
| 7 | ToolHang | image / lowdim (+`_abs`) | robosuite/MuJoCo | as above | 7 / 10 | `RobomimicImageRunner` | `…/tool_hang/ph/…` |
| 8 | BlockPush | lowdim (+`_abs`) | pybullet | 16 | 2 (delta XY) | `BlockPushLowdimRunner` | `data/block_pushing/multimodal_push_seed.zarr` |
| 9 | Kitchen | lowdim (+`_abs`) | MuJoCo (mujoco-py) | 60 | 9 | `KitchenLowdimRunner` | `data/kitchen` |
| 10 | Real Push-T | image | real UR5 | 2 cameras + eef pose | 2 | `RealPushTImageRunner` (**stub**) | `data/pusht_real/real_pusht_20230105` |

*(That is 5 robomimic tasks × {image, lowdim} plus PushT ×2, BlockPush, Kitchen — the "12 tasks"
count in the paper. Each also has an `_abs` variant, giving 27 task configs in total.)*

`ph` = proficient-human demos, `mh` = multi-human. `KitchenLowdimRunner` additionally requires
`data/kitchen/all_init_qpos.npy` and `all_init_qvel.npy` (`kitchen_lowdim_runner.py:84-85`).

### 9.6 Metric keys — what `run()` returns

| Runner | Aggregate keys | Per-episode keys |
|---|---|---|
| PushT image/keypoints, Robomimic image/lowdim | `train/mean_score`, `test/mean_score` | `{prefix}sim_max_reward_{seed}`, `{prefix}sim_video_{seed}` |
| Kitchen lowdim | `train|test/mean_score`, `train|test/p_1` … `p_7` | `{prefix}sim_video_{seed}` |
| BlockPush lowdim | `train|test/mean_score`, `p1`, `p2`, plus `REACH_0`, `TARGET_0_0`, … event probabilities | `{prefix}sim_max_reward_{seed}`, `{prefix}sim_video_{seed}` |

Semantics differ by task, which matters when you tabulate results:

- **Push-T** `test/mean_score` = mean over episodes of the **max goal coverage** reached, in `[0,1]`.
- **Robomimic** `test/mean_score` = mean of per-episode max sparse reward = literally **success rate**.
- **Kitchen** `mean_score` = mean of `sum(rewards)/7` (fraction of the 7 subtasks completed);
  `p_n` = `P(n_completed >= n)`.
- **BlockPush** `total_reward = np.unique(this_rewards).sum()`; `p1 = total_reward > 0.4`,
  `p2 = total_reward > 0.9` (`blockpush_lowdim_runner.py:249-253`).

`test/mean_score` is the TopK monitor key for every shipped config (as `test_mean_score` — §10.2).

---

## 10. Gotchas

### 10.1 `random_crop=False` silently applies no crop

`model/vision/multi_image_obs_encoder.py:85-105`:

```python
this_randomizer = nn.Identity()
if crop_shape is not None:
    if random_crop:
        this_randomizer = CropRandomizer(...)
    else:
        this_normalizer = torchvision.transforms.CenterCrop(size=(h,w))   # <-- wrong variable
# configure normalizer
this_normalizer = nn.Identity()                                           # <-- overwrites it
```

The `else` branch assigns `this_normalizer`, which the next statement unconditionally overwrites.
`this_randomizer` stays `nn.Identity()`, so **no crop is applied at all**. Affects
`DiffusionUnetImagePolicy` (which uses `MultiImageObsEncoder`). The hybrid policy is unaffected —
it goes through the robomimic encoder and its own `eval_fixed_crop` path.

### 10.2 `test/mean_score` vs `test_mean_score`

`train_diffusion_unet_hybrid_workspace.py:264-273` sanitizes metric names before checkpoint
selection:

```python
metric_dict = {key.replace('/', '_'): value for key, value in step_log.items()}
topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
```

So wandb shows `test/mean_score` while `checkpoint.topk.monitor_key` must be written
`test_mean_score`. Not a bug — just a trap when writing a new config.

### 10.3 Validation and rollout use different weights

Validation loss is computed with `self.model`; rollouts and the sampling diagnostic use
`self.ema_model`. Comparing `val_loss` against `test/mean_score` compares two different networks.

### 10.4 `EMAModel` is not checkpointed

`model/diffusion/ema_model.py` has **no `state_dict`**, so the `ema` object itself is never saved —
only `self.ema_model` (a real `nn.Module`) is. On resume, `optimization_step` and therefore the
decay warmup restart from zero. Averaged weights persist; the decay schedule does not.

### 10.5 `latest.ckpt` can be momentarily empty

`save_checkpoint` writes on a background thread by default (`base_workspace.py:66`). The workspace
acknowledges this in a comment at `:270-272`: *"We can't copy the last checkpoint here since
save_checkpoint uses threads. therefore at this point the file might have been empty!"* Don't read
`latest.ckpt` from another process immediately after an epoch boundary.

### 10.6 `n_latency_steps` is only implemented in two runners

Only `robomimic_lowdim_runner.py` and `pusht_keypoints_runner.py`. **No image runner supports it.**
The mechanism (`robomimic_lowdim_runner.py:94-96, 272-295`):

```python
env_n_obs_steps = n_obs_steps + n_latency_steps
...
'obs': obs[:, :self.n_obs_steps]                         # discard the LAST n_latency observations
action = np_action_dict['action'][:, self.n_latency_steps:]   # discard the FIRST n_latency actions
```

The env keeps a longer buffer, the policy sees only the stale prefix, and the leading predicted
actions are thrown away — simulating inference latency. In configs it couples to
`n_action_steps: ${eval:'${n_action_steps}+${n_latency_steps}'}` and
`pad_before: ${eval:'${n_obs_steps}-1+${n_latency_steps}'}`.

### 10.7 `n_envs: null` means 56 processes, and robosuite can segfault

`n_envs = n_train + n_test` when unset — 56 forked workers for the default PushT config. The
robomimic configs carry the comment *"costs 1GB per env"* and *"evaluation at this config requires
a 16 core 64GB instance"*, and use `n_envs: 28` (2 chunks of 28).

Separately, the README warns that environments creating an OpenGL context during initialization
(robosuite) *"often causes obscure bugs like segmentation fault"* under forking. The mitigation is
`dummy_env_fn` (`robomimic_image_runner.py:126`), but this remains a live risk.

For a first smoke test, override:
`task.env_runner.n_envs=4 task.env_runner.n_test=4 task.env_runner.n_train=1`, or swap in the
commented-out `SyncVectorEnv` line.

### 10.8 Two runners read datasets in `__init__`

`RobomimicImageRunner` (`:79` for `env_meta`, `:166` for per-demo init states) and
`KitchenLowdimRunner` (`:84-85` for `all_init_qpos.npy` / `all_init_qvel.npy`) both touch the
filesystem during construction. **You cannot instantiate them without the dataset present.**
`PushTImageRunner` is the only runner that constructs data-free — which makes Push-T the natural
first target when bringing up a new policy.

### 10.9 `RobomimicImageWrapper` whitelists obs keys by suffix

`robomimic_image_wrapper.py` builds the observation space by matching key **suffixes**:
`*image` → `[0,1]`, `*quat` / `*qpos` / `*pos` → `[-1,1]`, and anything else raises
`RuntimeError(f"Unsupported type {key}")`. If you add a custom observation key, name it to match
one of those suffixes or patch the wrapper.

### 10.10 `use_cache: True` writes a large sidecar

Robomimic image datasets convert HDF5 → zarr on first use and cache it as `<name>.hdf5.zarr.zip`
next to the source. Expect a very long first epoch and significant extra disk.

### 10.11 `av` is required for *simulation* eval, not just the real robot

`gym_util/video_recording_wrapper.py:3` imports `VideoRecorder` from
`real_world/video_recorder.py:3`, which does `import av`. Rollout video recording is on by default,
so PyAV is a hard dependency of the sim benchmark path.

### 10.12 wandb blocks unless configured

Shipped configs set `logging.mode: online`, `project: diffusion_policy_debug`. Either run
`wandb login` or override `logging.mode=offline`, otherwise training stalls on an interactive prompt.

### 10.13 `mujoco_image_dataset.py` is orphaned and has two defects

Added by the fork commit (§1). Nothing references it: `grep -rn "mujoco_image" --include=*.py
--include=*.yaml .` returns **zero** matches — no config, no env, no runner, and no import.
Two real problems if you copy it as a template:

1. **`get_normalizer()` doesn't match `_sample_to_data()`.**
   ```python
   # get_normalizer  ->  14-d
   'agent_pos': np.concatenate([rb['robot_0_tcp_xyz_wxyz'], rb['robot_0_tcp_xyz_wxyz']], axis=-1)
   # _sample_to_data ->  8-d
   agent_pos = np.concatenate([sample['robot_0_tcp_xyz_wxyz'], sample['robot_0_gripper_width']], -1)
   ```
   The TCP pose is duplicated instead of being concatenated with the gripper width, so the
   normalizer's statistics describe a different vector than the samples it normalizes.
2. `get_all_actions()` is left unimplemented (inherits `NotImplementedError`), unlike every other
   dataset.

Its `test()` also hardcodes `/home/yihuai/robotics/...`, a path from another machine. Treat this
file as dead code; use `pusht_image_dataset.py` as your template instead.

### 10.14 `diffusion_unet_video_policy.py` uses the wrong type string

`:54` reads `attr.get('type', 'lowdim')` — everywhere else in the codebase the value is `'low_dim'`
with an underscore. Confined to the experimental video policy, but it means that file's shape_meta
handling diverges from the documented convention.

### 10.15 `*_abs.hdf5` files are generated, not downloaded

The robomimic archives ship `low_dim.hdf5` and `image.hdf5` only. Every `*_abs` task config points
at `low_dim_abs.hdf5` / `image_abs.hdf5`, which you produce locally with
`diffusion_policy/scripts/robomimic_dataset_conversion.py`. A fresh download will fail on any
`_abs` task until that conversion has run.

### 10.16 There is no `pip install -e .`

`setup.py` declares no dependencies and the README never installs the package. Imports resolve
because **cwd is the repo root**. Each workspace module compensates with a `sys.path.append(ROOT_DIR)`
+ `os.chdir(ROOT_DIR)` prologue in its `if __name__ == "__main__":` block.

---

## 11. Plugging in your own policy

Full walkthrough with runnable code: **[`BENCHMARK_READINESS.md` §5](BENCHMARK_READINESS.md)**.
The short version — an image policy needs exactly four things:

```python
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

class MyPolicy(BaseImagePolicy):
    # .device / .dtype come free from ModuleAttrMixin

    def reset(self):
        """Called once before each rollout batch. Clear any temporal state."""

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # obs_dict[key]: (B, To, *shape), already on self.device
        #   rgb keys  -> (B, To, 3, H, W) float32 in [0,1], channel-first
        #   lowdim    -> (B, To, D)
        return {
            'action':      ...,   # (B, Ta, Da)  <- required by every env_runner
            'action_pred': ...,   # (B, T,  Da)  <- required only by the training workspaces
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch) -> torch.Tensor:
        """Only needed if you also train inside this repo's workspaces."""
```

Then `PushTImageRunner(output_dir=...).run(my_policy)['test/mean_score']` is your benchmark number.

To add a whole new task, the README's recipe is to imitate four files, keeping `shape_meta`
consistent across all of them:
`dataset/pusht_image_dataset.py`, `env_runner/pusht_image_runner.py`,
`config/task/pusht_image.yaml`, and an env under `env/`.

---

## 12. The real-world stack (brief)

`diffusion_policy/real_world/` — 11 modules. Hardware expected: **1× UR5-CB3 or UR5e** (RTDE
interface), **2× Intel RealSense D415**, **1× 3Dconnexion SpaceMouse**, a Millibar tool changer, and
a 3D-printed end effector + T-block.

- `real_env.py` — `RealEnv` is **not** a gym env: `step()` is split into `get_obs()` and
  `exec_actions(actions, timestamps)`.
- `rtde_interpolation_controller.py` — UR5 servoL controller in its own process, fed by a
  `SharedMemoryQueue`.
- `single_realsense.py` / `multi_realsense.py` — one process per camera writing into a
  `SharedMemoryRingBuffer`.
- `spacemouse.py` — needs `libspnav-dev` + `spacenavd`.
- `video_recorder.py` — **the one real_world module the simulation path depends on** (see §10.11).

Entrypoints: `demo_real_robot.py`, `eval_real_robot.py`. Separate env: `conda_environment_real.yaml`.
Not needed for simulation benchmarking.
