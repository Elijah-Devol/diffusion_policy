from typing import Dict, Optional
import torch
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy


class DiffusionUnetHybridImagePerStepPolicy(DiffusionUnetHybridImagePolicy):
    """
    Diffusion Policy for per-step polynomial actions.
    Chunk index 0 is the step that just finished: during training it is given (not denoised, not in
    the loss); at inference the runner passes it as prev_action and it is pinned (inpainted), so every
    new chunk continues exactly from what was executed.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.obs_as_global_cond, 'per-step policy expects obs_as_global_cond=True'
        assert self.n_obs_steps >= 2, 'index 0 is visible only when n_obs_steps >= 2'
        self.mask_generator = LowdimMaskGenerator(
            action_dim=self.action_dim,
            obs_dim=0,
            max_n_obs_steps=self.n_obs_steps,
            fix_obs_steps=True,
            action_visible=True)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor],
                       prev_action: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """prev_action: (B, action_dim) label of the step that just finished, in env units."""
        assert 'past_action' not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        T, Da, To = self.horizon, self.action_dim, self.n_obs_steps

        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        global_cond = self.obs_encoder(this_nobs).reshape(B, -1)

        cond_data = torch.zeros(size=(B, T, Da), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        if prev_action is not None:
            prev = prev_action.to(device=self.device, dtype=self.dtype)
            cond_data[:, 0] = self.normalizer['action'].normalize(prev)
            cond_mask[:, 0] = True

        nsample = self.conditional_sample(
            cond_data, cond_mask, local_cond=None, global_cond=global_cond, **self.kwargs)
        action_pred = self.normalizer['action'].unnormalize(nsample[..., :Da])

        start = To - 1
        end = start + self.n_action_steps
        return {'action': action_pred[:, start:end], 'action_pred': action_pred}
