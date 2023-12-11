import gc
import torch

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner

from ray.rllib.utils.torch_utils import sequence_mask
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.core.learner.learner import POLICY_LOSS_KEY, VF_LOSS_KEY, ENTROPY_KEY
from ray.rllib.utils.torch_utils import explained_variance
from ray.rllib.algorithms.ppo.ppo_learner import (
    LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY,
    LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY,
    PPOLearner,
)



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

class PPOTorchLearnerPretrainedKLLoss(PPOLearner, TorchLearner):
    ''' 
    Chaned version of the PPOTorchLearner, removed the KL loss and added a new
    KL loss from the frozen pretrained model.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pt_kl_coeff = 1.
        self.pt_kl_coeff_decay = 0.9995

    def compute_loss_for_module(self, *, module_id, hps, batch, fwd_out):
        if self.module[module_id].is_stateful():
            # In the RNN case, we expect incoming tensors to be padded to the maximum
            # sequence length. We infer the max sequence length from the actions
            # tensor.
            maxlen = torch.max(batch[SampleBatch.SEQ_LENS])
            mask = sequence_mask(batch[SampleBatch.SEQ_LENS], maxlen=maxlen)
            num_valid = torch.sum(mask)

            def possibly_masked_mean(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            possibly_masked_mean = torch.mean

        action_dist_class_train = (
            self.module[module_id].unwrapped().get_train_action_dist_cls()
        )
        action_dist_class_exploration = (
            self.module[module_id].unwrapped().get_exploration_action_dist_cls()
        )

        pt_action_dist = action_dist_class_exploration.from_logits(
            fwd_out['pretrained_action_dist'],
        )

        curr_action_dist = action_dist_class_train.from_logits(
            fwd_out[SampleBatch.ACTION_DIST_INPUTS]
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(batch[SampleBatch.ACTIONS])
            - batch[SampleBatch.ACTION_LOGP]
        )

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = possibly_masked_mean(curr_entropy)

        surrogate_loss = torch.min(
            batch[Postprocessing.ADVANTAGES] * logp_ratio,
            batch[Postprocessing.ADVANTAGES]
            * torch.clamp(logp_ratio, 1 - hps.clip_param, 1 + hps.clip_param),
        )

        # Compute a value function loss.
        if hps.use_critic:
            value_fn_out = fwd_out[SampleBatch.VF_PREDS]
            vf_loss = torch.pow(value_fn_out - batch[Postprocessing.VALUE_TARGETS], 2.0)
            vf_loss_clipped = torch.clamp(vf_loss, 0, hps.vf_clip_param)
            mean_vf_loss = possibly_masked_mean(vf_loss_clipped)
            mean_vf_unclipped_loss = possibly_masked_mean(vf_loss)
        # Ignore the value function.
        else:
            value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            mean_vf_unclipped_loss = torch.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)

        # pretrained kl loss
        pt_kl = torch.mean(pt_action_dist.kl(curr_action_dist))
        pt_kl_loss = self.pt_kl_coeff * pt_kl

        total_loss = possibly_masked_mean(
            -surrogate_loss
            + hps.vf_loss_coeff * vf_loss_clipped
            - (
                self.entropy_coeff_schedulers_per_module[module_id].get_current_value()
                * curr_entropy
            )
            + pt_kl_loss
        )

        # Register important loss stats.
        self.register_metrics(
            module_id,
            {
                POLICY_LOSS_KEY: -possibly_masked_mean(surrogate_loss),
                VF_LOSS_KEY: mean_vf_loss,
                LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY: mean_vf_unclipped_loss,
                LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY: explained_variance(
                    batch[Postprocessing.VALUE_TARGETS], value_fn_out
                ),
                ENTROPY_KEY: mean_entropy,
                'pretrained_kl_loss': pt_kl,
                'pretrained_kl_loss_scaled': pt_kl_loss,
            },
        )
        # Return the total loss.
        return total_loss

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


class PPOTorchLearnerPretrainedKLLoss_OLD(PPOTorchLearner):
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
            fwd_out['pretrained_action_dist'],
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

