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
        mean_vf_target = worker.global_vars.get('mean_vf_target', defaultdict(lambda: 0.))['default_policy']
        std_vf_target = worker.global_vars.get('std_vf_target', defaultdict(lambda: 1.))['default_policy']
        episode.custom_metrics['mean_vf_target'] = mean_vf_target
        episode.custom_metrics['std_vf_target'] = std_vf_target

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
        mean_vf_target = worker.global_vars.get('mean_vf_target', defaultdict(lambda: 0.))['default_policy']
        std_vf_target = worker.global_vars.get('std_vf_target', defaultdict(lambda: 1.))['default_policy']

        list_len = len(episode._agent_collectors["agent0"].buffers[SampleBatch.VF_PREDS][0])
        if list_len > 2:
            episode._agent_collectors["agent0"].buffers[SampleBatch.VF_PREDS][0][-1] = \
                mean_vf_target + (std_vf_target) * episode._agent_collectors["agent0"].buffers[SampleBatch.VF_PREDS][0][-1]
        elif list_len == 2:
            episode._agent_collectors["agent0"].buffers[SampleBatch.VF_PREDS][0][0] = \
                mean_vf_target + (std_vf_target) * episode._agent_collectors["agent0"].buffers[SampleBatch.VF_PREDS][0][0]
            episode._agent_collectors["agent0"].buffers[SampleBatch.VF_PREDS][0][-1] = \
                mean_vf_target + (std_vf_target) * episode._agent_collectors["agent0"].buffers[SampleBatch.VF_PREDS][0][-1]

