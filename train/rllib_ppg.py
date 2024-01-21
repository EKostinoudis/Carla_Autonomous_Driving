import numpy as np
from collections import defaultdict
from copy import deepcopy
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo.ppo_learner import LEARNER_RESULTS_KL_KEY
from ray.rllib.execution.rollout_ops import (
    standardize_fields,
    synchronous_parallel_sample,
)
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
    SAMPLE_TIMER,
    ALL_MODULES,
)
from ray.rllib.utils.typing import ResultDict
from ray.rllib.policy.sample_batch import concat_samples, SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.core.learner.learner import LearnerHyperparameters
from ray.rllib.algorithms.ppo.ppo_learner import PPOLearnerHyperparameters


def update_stats(existing_mean, existing_var, existing_count, new_mean, new_var, new_count):
    combined_count = existing_count + new_count

    # Update mean
    combined_mean = (existing_count * existing_mean + new_count * new_mean) / combined_count

    # Update variance
    combined_variance = (
        (existing_count * (existing_var + existing_mean**2) +
         new_count * (new_var + new_mean**2)) / combined_count
        - combined_mean**2
    )
    combined_variance = max(combined_variance, 1e-6)

    return combined_mean, combined_variance, combined_count

@dataclass
class PPGLearnerHyperparameters(PPOLearnerHyperparameters):
    use_pt_kl_loss: bool = None
    pt_kl_coeff: float = None
    pt_kl_coeff_decay: float = None
    aux_kl_coef: float = None
    aux_vf_coef: float = None
    aux_policy_vf_coef: float = None

class PPGConfig(PPOConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or PPG)
        self.use_pt_kl_loss = True
        self.pt_kl_coeff = 0.1
        self.pt_kl_coeff_decay = 0.999
        self.value_tartget_norm = True
        self.auxiliary_epochs = 5
        self.normal_phase_iters = 2
        self.aux_kl_coef = 1.
        self.aux_vf_coef = 1.
        self.aux_policy_vf_coef = 1.

    def get_learner_hyperparameters(self) -> PPGLearnerHyperparameters:
        base_hps = super().get_learner_hyperparameters()
        return PPGLearnerHyperparameters(
            use_pt_kl_loss=self.use_pt_kl_loss,
            pt_kl_coeff=self.pt_kl_coeff,
            pt_kl_coeff_decay=self.pt_kl_coeff_decay,
            aux_kl_coef=self.aux_kl_coef,
            aux_vf_coef=self.aux_vf_coef,
            aux_policy_vf_coef=self.aux_policy_vf_coef,
            **dataclasses.asdict(base_hps),
        )

class PPG(PPO):
    def __init__(self, config, env=None, logger_creator=None, **kwargs):
        super().__init__(config, env, logger_creator, **kwargs)
        self.mean_vf_target = defaultdict(lambda: 0.)
        self.var_vf_target = defaultdict(lambda: 1.)
        self.std_vf_target = defaultdict(lambda: 1.)
        self.num_values_vf_target = defaultdict(lambda: 0)
        self.train_batches = []

    @classmethod
    def get_default_config(cls) -> AlgorithmConfig:
        return PPGConfig()

    def training_step(self) -> ResultDict:
        if not self.config._enable_learner_api:
            raise Exception('This PPG implementation only works with the learner api.')

        # Collect SampleBatches from sample workers until we have a full batch.
        with self._timers[SAMPLE_TIMER]:
            if self.config.count_steps_by == "agent_steps":
                train_batch = synchronous_parallel_sample(
                    worker_set=self.workers,
                    max_agent_steps=self.config.train_batch_size,
                )
            else:
                train_batch = synchronous_parallel_sample(
                    worker_set=self.workers, max_env_steps=self.config.train_batch_size
                )

        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        for key in train_batch.policy_batches:
            self.train_batches.append(SampleBatch({
                SampleBatch.OBS: train_batch[key][SampleBatch.OBS],
                SampleBatch.ACTION_DIST_INPUTS: np.zeros_like(train_batch[key][SampleBatch.ACTION_DIST_INPUTS]),
                Postprocessing.VALUE_TARGETS: deepcopy(train_batch[key][Postprocessing.VALUE_TARGETS]),
            }))

        # Standardize value_targets
        if self.config.value_tartget_norm:
            for key in train_batch.policy_batches:
                self.mean_vf_target[key], self.var_vf_target[key], self.num_values_vf_target[key] = update_stats(
                    self.mean_vf_target[key],
                    self.var_vf_target[key],
                    self.num_values_vf_target[key],
                    np.mean(train_batch[key][Postprocessing.VALUE_TARGETS]),
                    np.var(train_batch[key][Postprocessing.VALUE_TARGETS]),
                    train_batch[key][Postprocessing.VALUE_TARGETS].shape[0],
                )
                self.std_vf_target[key] = self.var_vf_target[key]**0.5
                train_batch[key][Postprocessing.VALUE_TARGETS] = \
                    (train_batch[key][Postprocessing.VALUE_TARGETS] - self.mean_vf_target[key]) / \
                    (self.std_vf_target[key])

        # Standardize advantages
        train_batch = standardize_fields(train_batch, ["advantages"])

        # Train
        is_module_trainable = self.workers.local_worker().is_policy_to_train
        self.learner_group.set_is_module_trainable(is_module_trainable)
        train_results = self.learner_group.update(
            train_batch,
            minibatch_size=self.config.sgd_minibatch_size,
            num_iters=self.config.num_sgd_iter,
        )
        policies_to_update = set(train_results.keys()) - {ALL_MODULES}

        # auxiliary phase
        if len(self.train_batches) == self.config.normal_phase_iters:
            self.train_batches = concat_samples(self.train_batches).as_multi_agent()

            # normalize value targets if needed
            if self.config.value_tartget_norm:
                for key in train_batch.policy_batches:
                    self.std_vf_target[key] = self.var_vf_target[key]**0.5
                    self.train_batches[key][Postprocessing.VALUE_TARGETS] = \
                        (self.train_batches[key][Postprocessing.VALUE_TARGETS] - self.mean_vf_target[key]) / \
                        (self.std_vf_target[key])

            train_results_sleep = self.learner_group.update(
                self.train_batches,
                minibatch_size=self.config.sgd_minibatch_size,
                num_iters=self.config.auxiliary_epochs,
            )

            self.train_batches = []

        # TODO (Kourosh): num_grad_updates per each policy should be accessible via
        # train_results
        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
            "num_grad_updates_per_policy": {
                pid: self.workers.local_worker().policy_map[pid].num_grad_updates
                for pid in policies_to_update
            },
        }

        if self.config.value_tartget_norm:
            # update the mean and std vf targets in the workers
            mean, std = self.mean_vf_target, self.std_vf_target
            self.workers.foreach_worker(
                lambda w: w.set_global_vars({
                    "mean_vf_target": mean,
                    "std_vf_target": std,
                    **global_vars,
                }),
            )

        # Update weights - after learning on the local worker - on all remote
        # workers.
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            if self.workers.num_remote_workers() > 0:
                from_worker_or_learner_group = None
                if self.config._enable_learner_api:
                    # sync weights from learner_group to all rollout workers
                    from_worker_or_learner_group = self.learner_group
                self.workers.sync_weights(
                    from_worker_or_learner_group=from_worker_or_learner_group,
                    policies=policies_to_update,
                    global_vars=global_vars,
                )
            elif self.config._enable_learner_api:
                weights = self.learner_group.get_weights()
                self.workers.local_worker().set_weights(weights)

        kl_dict = {}
        if self.config.use_kl_loss:
            for pid in policies_to_update:
                kl = train_results[pid][LEARNER_RESULTS_KL_KEY]
                kl_dict[pid] = kl
                if np.isnan(kl):
                    logger.warning(
                        f"KL divergence for Module {pid} is non-finite, this will "
                        "likely destabilize your model and the training process. "
                        "Action(s) in a specific state have near-zero probability. "
                        "This can happen naturally in deterministic environments "
                        "where the optimal policy has zero mass for a specific "
                        "action. To fix this issue, consider setting `kl_coeff` to "
                        "0.0 or increasing `entropy_coeff` in your config."
                    )

        # triggers a special update method on RLOptimizer to update the KL values.
        additional_results = self.learner_group.additional_update(
            module_ids_to_update=policies_to_update,
            sampled_kl_values=kl_dict,
            timestep=self._counters[NUM_AGENT_STEPS_SAMPLED],
        )
        for pid, res in additional_results.items():
            train_results[pid].update(res)

        return train_results

