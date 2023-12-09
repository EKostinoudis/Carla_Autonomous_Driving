import gc
import torch

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner

class PPOTorchLearnerClearCache(PPOTorchLearner):
    def additional_update_for_module(self, *, module_id, hps, timestep, sampled_kl_values):
        results = super().additional_update_for_module(
            module_id=module_id,
            hps=hps,
            timestep=timestep,
            sampled_kl_values=sampled_kl_values,
        )

        # clear gpu memory cache
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

        return results

class PPOTorchLearnerPretrainedKLLoss(PPOTorchLearner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pt_kl_coeff = 1.
        self.pt_kl_coeff_decay = 0.9995

    def compute_loss_for_module(self, *, module_id, hps, batch, fwd_out):
        ret = super().compute_loss_for_module(
            module_id=module_id,
            hps=hps,
            batch=batch,
            fwd_out=fwd_out,
        )

        # KL divergence loss between the RL model and the frozen pretrained policy
        action_dist_class = (
            self.module[module_id].unwrapped().get_exploration_action_dist_cls()
        )
        pt_action_dist = action_dist_class.from_logits(
            batch['pretrained_action_dist'],
        )
        curr_action_dist = action_dist_class.from_logits(
            fwd_out[SampleBatch.ACTION_DIST_INPUTS]
        )
        pt_kl = torch.mean(pt_action_dist.kl(curr_action_dist))
        pt_kl_loss = self.pt_kl_coeff * pt_kl

        self.register_metrics(
            module_id,
            {
                'pretrained_kl_loss': pt_kl,
                'pretrained_kl_loss_scaled': pt_kl_loss,
            },
        )
        return ret + pt_kl_loss

    def additional_update_for_module(self, *, module_id, hps, timestep, sampled_kl_values):
        results = super().additional_update_for_module(
            module_id=module_id,
            hps=hps,
            timestep=timestep,
            sampled_kl_values=sampled_kl_values,
        )

        # update the pretrained kl coefficient
        self.pt_kl_coeff *= self.pt_kl_coeff_decay

        # clear gpu memory cache
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        return results

