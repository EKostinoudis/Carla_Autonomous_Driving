from collections import defaultdict
from typing import Dict
import numpy as np
import torch

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy.sample_batch import SampleBatch

class LogInfoCallback(DefaultCallbacks):
    '''
    Log every variable in the info dict.
    NOTE: the info dict must have only scalars, else this will crash.
    '''
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
        **kwargs,
    ):
        episode.user_data = defaultdict(list)

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
        **kwargs,
    ):
        data = episode.last_info_for('agent0')
        for name, value in data.items():
            episode.user_data[name].append(value)

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
        **kwargs,
    ):
        for name, value in episode.user_data.items():
            episode.custom_metrics[name + "_avg"] = np.mean(value)
            episode.custom_metrics[name + "_sum"] = np.sum(value)
            episode.hist_data[name] = value


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
    return combined_mean, combined_variance, combined_count


class NormValueInfoCallback(LogInfoCallback):
    def on_create_policy(self, *, policy_id, policy: Policy) -> None:
        policy.mean_vf_target = 0.
        policy.var_vf_target = 1.
        policy.num_values = 0

    '''
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
        **kwargs,
    ):
        super().on_episode_start(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            env_index=env_index,
            **kwargs,
        )
    '''

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
        **kwargs,
    ):
        super().on_episode_step(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            env_index=env_index,
            **kwargs,
        )
        if hasattr(worker, 'after_end_sample'):
            if worker.after_end_sample:
                worker.mean_vf_target = policies['default_policy'].mean_vf_target
                worker.var_vf_target = policies['default_policy'].var_vf_target
                worker.after_end_sample = False
        else:
            if not hasattr(worker, 'mean_vf_target'):
                worker.mean_vf_target = policies['default_policy'].mean_vf_target
                worker.var_vf_target = policies['default_policy'].var_vf_target

        list_len = len(episode._agent_collectors["agent0"].buffers[SampleBatch.VF_PREDS][0])
        if list_len > 2:
            episode._agent_collectors["agent0"].buffers[SampleBatch.VF_PREDS][0][-1] = \
                worker.mean_vf_target + (worker.var_vf_target**0.5) * episode._agent_collectors["agent0"].buffers[SampleBatch.VF_PREDS][0][-1]
        elif list_len == 2:
            episode._agent_collectors["agent0"].buffers[SampleBatch.VF_PREDS][0][0] = \
                worker.mean_vf_target + (worker.var_vf_target**0.5) * episode._agent_collectors["agent0"].buffers[SampleBatch.VF_PREDS][0][0]
            episode._agent_collectors["agent0"].buffers[SampleBatch.VF_PREDS][0][-1] = \
                worker.mean_vf_target + (worker.var_vf_target**0.5) * episode._agent_collectors["agent0"].buffers[SampleBatch.VF_PREDS][0][-1]

    def on_postprocess_trajectory(
        self,
        *,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs,
    ) -> None:
        policy = policies[policy_id]

        policy.mean_vf_target, policy.var_vf_target, policy.num_values = update_stats(
            policy.mean_vf_target,
            policy.var_vf_target,
            policy.num_values,
            np.mean(postprocessed_batch['value_targets']),
            np.var(postprocessed_batch['value_targets']),
            postprocessed_batch['value_targets'].shape[0],
        )

        worker.mean_vf_target_last = policy.mean_vf_target
        worker.var_vf_target_last = policy.var_vf_target

        episode.custom_metrics['mean_vf_target'] = policy.mean_vf_target
        episode.custom_metrics['var_vf_target'] = policy.var_vf_target
        episode.custom_metrics['decay_vf_target'] = policy.decay

    def on_sample_end(self, *, worker, samples, **kwargs):
        samples['default_policy']['value_targets'] = \
         (samples['default_policy']['value_targets'] - worker.mean_vf_target_last) / (worker.var_vf_target_last**0.5)
        worker.after_end_sample = True
