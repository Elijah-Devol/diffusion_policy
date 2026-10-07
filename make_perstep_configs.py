"""
Write image_pusht_perstep_b{2,4,6}.yaml from the dp-04 baseline config, changing only what the
per-step action needs (action shape, dataset, policy, env runner, wandb run id/name).

Usage:
    python make_perstep_configs.py
"""
import copy
from omegaconf import OmegaConf

BASE = 'image_pusht_diffusion_policy_cnn.yaml'
ZARR = '/home/elijahong/robot-learning/data/pusht/pusht_cchi_v7_replay_perstep_b{b}.zarr'


def main():
    base = OmegaConf.load(BASE)
    for b in (2, 4, 6):
        cfg = copy.deepcopy(base)
        shape = [2 * b]
        cfg.shape_meta.action.shape = shape
        cfg.policy.shape_meta.action.shape = shape
        cfg.task.shape_meta.action.shape = shape
        cfg.policy._target_ = ('diffusion_policy.policy.diffusion_unet_hybrid_image_perstep_policy.'
                               'DiffusionUnetHybridImagePerStepPolicy')
        cfg.task.dataset._target_ = ('diffusion_policy.dataset.pusht_perstep_image_dataset.'
                                     'PushTPerStepImageDataset')
        cfg.task.dataset.zarr_path = ZARR.format(b=b)
        cfg.task.dataset.budget = b
        cfg.task.env_runner._target_ = ('diffusion_policy.env_runner.pusht_perstep_image_runner.'
                                        'PushTPerStepImageRunner')
        cfg.task.env_runner.budget = b
        cfg.name = 'train_diffusion_unet_hybrid_perstep'
        cfg.logging.id = f'dp-perstep-b{b}'
        cfg.logging.name = f'dp-perstep-b{b}'
        cfg.logging.tags = ['train_diffusion_unet_hybrid_perstep', 'pusht_image', f'budget_{b}']
        out = f'image_pusht_perstep_b{b}.yaml'
        OmegaConf.save(cfg, out)
        print('wrote', out)


if __name__ == '__main__':
    main()
