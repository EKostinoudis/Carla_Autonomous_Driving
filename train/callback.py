from collections import defaultdict
from typing import Dict
import numpy as np

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2

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

