"""
Evaluate a PushT checkpoint (baseline or per-step) on more test seeds than the in-training rollouts.

Usage:
    python eval_checkpoint.py -c data/outputs/dp-perstep-b6_seed42/checkpoints/latest.ckpt \
        -o data/eval/dp-perstep-b6_seed42 --n_test 500

Uses the env runner stored in the checkpoint's config (so per-step checkpoints are executed with
their per-step executor), overriding only the number/seeds of test episodes. Videos are disabled.
"""
import json
import os
import pathlib
import click
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--n_test', default=500, type=int, help='test episodes (seeds test_start_seed ...)')
@click.option('--test_start_seed', default=100000, type=int, help='same seeds as the in-training rollouts')
@click.option('--n_envs', default=16, type=int)
def main(checkpoint, output_dir, device, n_test, test_start_seed, n_envs):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device(device)).eval()

    runner_cfg = OmegaConf.to_container(cfg.task.env_runner, resolve=True)
    runner_cfg.update(n_train=0, n_train_vis=0, n_test=n_test, n_test_vis=0,
                      test_start_seed=test_start_seed, n_envs=n_envs)
    env_runner = hydra.utils.instantiate(runner_cfg, output_dir=output_dir)
    log = env_runner.run(policy)

    scores = np.array([v for k, v in log.items() if k.startswith('test/sim_max_reward_')])
    summary = {
        'checkpoint': checkpoint,
        'runner': runner_cfg['_target_'],
        'n_test': int(len(scores)),
        'test_mean_score': float(scores.mean()),
        'test_std_err': float(scores.std(ddof=1) / np.sqrt(len(scores))),
        'test_success_rate_0.95': float(np.mean(scores >= 1.0)),
    }
    json.dump({'summary': summary, 'per_seed': {k: float(v) for k, v in log.items() if 'sim_max_reward' in k}},
              open(os.path.join(output_dir, 'eval_summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
