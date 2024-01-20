import torch
import gc
from collections import defaultdict
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    Optional,
    Union,
)

from ray.rllib.core.learner.reduce_result_dict_fn import _reduce_mean_results
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.minibatch_utils import (
    MiniBatchDummyIterator,
    MiniBatchCyclicIterator,
)
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import ResultDict
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.core.learner.learner import POLICY_LOSS_KEY, VF_LOSS_KEY, ENTROPY_KEY
from ray.rllib.utils.torch_utils import explained_variance
from ray.rllib.algorithms.ppo.ppo_learner import (
    LEARNER_RESULTS_KL_KEY,
    LEARNER_RESULTS_CURR_KL_COEFF_KEY,
    LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY,
    LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY,
)
from typing import Any, Dict, List, Optional, Union
from ray.rllib.core.rl_module.rl_module import ModuleID
from ray.rllib.utils.lambda_defaultdict import LambdaDefaultDict
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.torch_utils import sequence_mask
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner

class PPGTorchLearner(PPOTorchLearner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sleep_mode_this_iter = False

    def build(self) -> None:
        super().build()
        
        self.curr_pt_kl_coeffs_per_module: Dict[ModuleID, Scheduler] = LambdaDefaultDict(
            lambda module_id: self._get_tensor_variable(
                self.hps.get_hps_for_module(module_id).pt_kl_coeff
            )
        )

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
        curr_action_dist = action_dist_class_train.from_logits(
            fwd_out[SampleBatch.ACTION_DIST_INPUTS]
        )

        # if the batch doesn't have rewards then we are in the sleep phase of ppg
        if not SampleBatch.REWARDS in batch:
            self.sleep_mode_this_iter = True
            prev_action_dist = action_dist_class_exploration.from_logits(
                batch[SampleBatch.ACTION_DIST_INPUTS]
            )
            kl_loss = possibly_masked_mean(prev_action_dist.kl(curr_action_dist))
            kl_loss_scaled = kl_loss * hps.aux_kl_coef

            value_fn_out = fwd_out[SampleBatch.VF_PREDS]
            value_policy_out = fwd_out['vf_policy']

            vf_loss = possibly_masked_mean(torch.pow(value_fn_out - batch[Postprocessing.VALUE_TARGETS], 2.0))
            vf_loss_p = possibly_masked_mean(torch.pow(value_policy_out - batch[Postprocessing.VALUE_TARGETS], 2.0))

            self.register_metrics(
                module_id,
                {
                    'sleep_kl_loss': kl_loss,
                    'sleep_kl_loss_scaled': kl_loss_scaled,
                    'sleep_vf_loss': vf_loss,
                    'sleep_vf_policy_loss': vf_loss_p,
                },
            )

            return torch.mean(
                vf_loss +
                vf_loss_p +
                kl_loss_scaled
            )
        else:
            # normal phase of ppg (ppo)
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

            # we must use all the layers in the loss, else we get an error
            vf_policy_loss = fwd_out['vf_policy'] * 0.

            total_loss = possibly_masked_mean(
                - surrogate_loss
                + hps.vf_loss_coeff * vf_loss_clipped
                - (
                    self.entropy_coeff_schedulers_per_module[module_id].get_current_value()
                    * curr_entropy
                )
                + vf_policy_loss
            )

            if hps.use_pt_kl_loss:
                pt_action_dist = action_dist_class_exploration.from_logits(
                    fwd_out['pretrained_action_dist'],
                )
                pt_kl = torch.mean(pt_action_dist.kl(curr_action_dist))
                pt_kl_loss = self.curr_pt_kl_coeffs_per_module[module_id] * pt_kl
                self.register_metrics(
                    module_id,
                    {
                        'pretrained_kl_loss': pt_kl,
                        'pretrained_kl_loss_scaled': pt_kl_loss,
                    },
                )
                total_loss += pt_kl_loss


            if hps.use_kl_loss:
                prev_action_dist = action_dist_class_exploration.from_logits(
                    batch[SampleBatch.ACTION_DIST_INPUTS]
                )
                action_kl = prev_action_dist.kl(curr_action_dist)
                mean_kl_loss = possibly_masked_mean(action_kl)
                total_loss += self.curr_kl_coeffs_per_module[module_id] * mean_kl_loss
                self.register_metrics(module_id, {LEARNER_RESULTS_KL_KEY: mean_kl_loss})

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

        if not self.sleep_mode_this_iter:
            # update the pretrained kl coefficient
            if hps.use_pt_kl_loss:
                curr_var = self.curr_pt_kl_coeffs_per_module[module_id]
                curr_var.data *= hps.pt_kl_coeff_decay
                results.update({'curr_pt_kl_coeff': curr_var.item()})
            self.sleep_mode_this_iter = False

        if hps.use_kl_loss:
            sampled_kl = sampled_kl_values[module_id]
            curr_var = self.curr_kl_coeffs_per_module[module_id]
            if sampled_kl > 2.0 * self.hps.kl_target:
                # TODO (Kourosh) why not 2?
                curr_var.data *= 1.5
            elif sampled_kl < 0.5 * self.hps.kl_target:
                curr_var.data *= 0.5
            results.update({LEARNER_RESULTS_CURR_KL_COEFF_KEY: curr_var.item()})

        # clear gpu memory cache
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

        return results

    def update(
        self,
        batch: MultiAgentBatch,
        *,
        minibatch_size: Optional[int] = None,
        num_iters: int = 1,
        reduce_fn: Callable[[List[Mapping[str, Any]]], ResultDict] = (
            _reduce_mean_results
        ),
    ) -> Union[Mapping[str, Any], List[Mapping[str, Any]]]:
        """Do `num_iters` minibatch updates given the original batch.

        Given a batch of episodes you can use this method to take more
        than one backward pass on the batch. The same minibatch_size and num_iters
        will be used for all module ids in MultiAgentRLModule.

        Args:
            batch: A batch of data.
            minibatch_size: The size of the minibatch to use for each update.
            num_iters: The number of complete passes over all the sub-batches
                in the input multi-agent batch.
            reduce_fn: reduce_fn: A function to reduce the results from a list of
                minibatch updates. This can be any arbitrary function that takes a
                list of dictionaries and returns a single dictionary. For example you
                can either take an average (default) or concatenate the results (for
                example for metrics) or be more selective about you want to report back
                to the algorithm's training_step. If None is passed, the results will
                not get reduced.
        Returns:
            A dictionary of results, in numpy format or a list of such dictionaries in
            case `reduce_fn` is None and we have more than one minibatch pass.
        """
        self._check_is_built()

        missing_module_ids = set(batch.policy_batches.keys()) - set(self.module.keys())
        if len(missing_module_ids) > 0:
            raise ValueError(
                "Batch contains module ids that are not in the learner: "
                f"{missing_module_ids}"
            )

        if num_iters < 1:
            # We must do at least one pass on the batch for training.
            raise ValueError("`num_iters` must be >= 1")

        if minibatch_size:
            batch_iter = MiniBatchCyclicIterator
        elif num_iters > 1:
            # `minibatch_size` was not set but `num_iters` > 1.
            # Under the old training stack, users could do multiple sgd passes
            # over a batch without specifying a minibatch size. We enable
            # this behavior here by setting the minibatch size to be the size
            # of the batch (e.g. 1 minibatch of size batch.count)
            minibatch_size = batch.count
            batch_iter = MiniBatchCyclicIterator
        else:
            # `minibatch_size` and `num_iters` are not set by the user.
            batch_iter = MiniBatchDummyIterator

        results = []
        # Convert input batch into a tensor batch (MultiAgentBatch) on the correct
        # device (e.g. GPU). We move the batch already here to avoid having to move
        # every single minibatch that is created in the `batch_iter` below.
        batch = self._convert_batch_type(batch)
        batch = self._set_slicing_by_batch_id(batch, value=True)

        # if we are in the speel mode (no reward in the batch), populate the
        # action dist field of the batch
        key = list(batch.policy_batches.keys())[0]
        if not SampleBatch.REWARDS in batch[key]:
            batch_len = batch.env_steps()
            idx = 0
            while idx < batch_len:
                minibatch = {}
                end_idx = min(idx+minibatch_size, batch_len)

                for module_id, module_batch in batch.policy_batches.items():
                    minibatch[module_id] = module_batch[idx:end_idx]

                minibatch = MultiAgentBatch(minibatch, end_idx - idx + 1)
                nested_tensor_minibatch = NestedDict(minibatch.policy_batches)
                with torch.no_grad():
                    fwd_out = self.module.forward_train(nested_tensor_minibatch)

                for policy_id in fwd_out.keys():
                    batch[policy_id][SampleBatch.ACTION_DIST_INPUTS][idx:end_idx, :] = \
                        fwd_out[policy_id][SampleBatch.ACTION_DIST_INPUTS]

                idx += minibatch_size

        for tensor_minibatch in batch_iter(batch, minibatch_size, num_iters):
            # Make the actual in-graph/traced `_update` call. This should return
            # all tensor values (no numpy).
            nested_tensor_minibatch = NestedDict(tensor_minibatch.policy_batches)
            (
                fwd_out,
                loss_per_module,
                metrics_per_module,
            ) = self._update(nested_tensor_minibatch)

            result = self.compile_results(
                batch=tensor_minibatch,
                fwd_out=fwd_out,
                loss_per_module=loss_per_module,
                metrics_per_module=defaultdict(dict, **metrics_per_module),
            )
            self._check_result(result)
            # TODO (sven): Figure out whether `compile_metrics` should be forced
            #  to return all numpy/python data, then we can skip this conversion
            #  step here.
            results.append(convert_to_numpy(result))

        batch = self._set_slicing_by_batch_id(batch, value=False)

        # Reduce results across all minibatches, if necessary.

        # If we only have one result anyways, then the user will not expect a list
        # to be reduced here (and might not provide a `reduce_fn` therefore) ->
        # Return single results dict.
        if len(results) == 1:
            return results[0]
        # If no `reduce_fn` provided, return list of results dicts.
        elif reduce_fn is None:
            return results
        # Pass list of results dicts through `reduce_fn` and return a single results
        # dict.
        return reduce_fn(results)

