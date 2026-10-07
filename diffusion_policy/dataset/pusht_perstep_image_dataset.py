from typing import Dict
import torch
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.perstep_motion import stay_label
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset


class PushTPerStepImageDataset(PushTImageDataset):
    """
    PushT image dataset whose 'action' holds per-step polynomial labels (make_perstep_labels.py).
    Index 0 of every window is the step that just finished; at the start of an episode there is none,
    so the padded entries become 'hold the initial position' -- the same thing the runner pins at reset.
    """
    def __init__(self, zarr_path, budget, **kwargs):
        super().__init__(zarr_path, **kwargs)
        self.budget = budget
        dim = self.replay_buffer['action'].shape[-1]
        assert dim == 2 * budget, (
            f'{zarr_path} has action dim {dim}, expected {2 * budget}; '
            f'run: python make_perstep_labels.py --budget {budget}')

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        n_front_pad = int(self.sampler.indices[idx][2])
        if n_front_pad > 0:
            x0 = data['obs']['agent_pos'][n_front_pad]
            data['action'][:n_front_pad] = stay_label(x0, self.budget)
        return dict_apply(data, torch.from_numpy)
