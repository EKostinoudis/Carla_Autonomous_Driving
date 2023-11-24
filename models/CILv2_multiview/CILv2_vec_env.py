import random
from omegaconf import DictConfig
from typing import Optional

from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.env.env_context import EnvContext

from CILv2_env import CILv2_env

class CILv2_vec_env(VectorEnv):
    def __init__(self,
                 env_config: DictConfig | dict,
                 path_to_conf_file: str,
                 rllib_config: Optional[EnvContext] = None,
                 ):
        num_envs = env_config.get('environments', 1)
        self.envs = []
        for offset in range(num_envs):
            seed = env_config.get('seed', random.randint(0, 10000)) + offset
            port = env_config.get('port', 2000) + 4*offset
            tm_port = env_config.get('tm_port', 8000) + offset
            new_env_config = {**env_config, 'port': port, 'tm_port': tm_port, 'seed': seed}
            self.envs.append(CILv2_env(new_env_config, path_to_conf_file))

        super().__init__(
            observation_space=self.envs[0].observation_space,
            action_space=self.envs[0].action_space,
            num_envs=num_envs,
        )

    def vector_reset(self, *, seeds=None, options=None):
        seeds = seeds or [None for _ in range(self.num_envs)]
        options = options or [None for _ in range(self.num_envs)]
        temp = [self.envs[i].reset(seed=seeds[i], options=options[i]) for i in range(self.num_envs)]
        obs, info = zip(*temp)
        obs, info = list(obs), list(info)
        return obs, info

    def reset_at(self, index, *, seed=None, options=None):
        return self.envs[index].reset(seed=seed, options=options)

    def vector_step(self, actions):
        temp = [env.step(action) for env, action in zip(self.envs, actions)]
        obs_batch, rew_batch, terminated_batch, truncated_batch, info_batch = zip(*temp)
        obs_batch, rew_batch, terminated_batch, truncated_batch, info_batch = \
            list(obs_batch), list(rew_batch), list(terminated_batch), list(truncated_batch), list(info_batch)
        return obs_batch, rew_batch, terminated_batch, truncated_batch, info_batch

    def get_sub_environments(self):
        return self.envs

