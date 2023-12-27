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

class NormValueInfoCallback(LogInfoCallback):
    def on_create_policy(self, *, policy_id, policy: Policy) -> None:
        policy.mean_vf_target = 0.
        policy.var_vf_target = 1.
        policy.decay = 0.9

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

        with torch.no_grad():
            policy.mean_vf_target = policy.decay * policy.mean_vf_target + \
                (1 - policy.decay) * np.mean(postprocessed_batch["value_targets"])
            policy.var_vf_target = policy.decay * policy.var_vf_target + \
                (1 - policy.decay) * np.var(postprocessed_batch["value_targets"])

            postprocessed_batch["value_targets"] = \
             (postprocessed_batch["value_targets"] - policy.mean_vf_target) / (policy.var_vf_target + 1e-8)

        episode.custom_metrics['mean_vf_target'] = policy.mean_vf_target
        episode.custom_metrics['var_vf_target'] = policy.var_vf_target
        episode.custom_metrics['decay_vf_target'] = policy.decay
