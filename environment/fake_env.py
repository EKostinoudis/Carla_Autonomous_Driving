import gymnasium as gym

class FakeEnv(gym.Env):
    '''
    This will be used on multiagent VectorEnv, because we don't have individual
    environments.
    '''
    def __init__(self, obs_space, act_space):
        self.action_space = act_space
        self.observation_space = obs_space

    def reset(self, *, seed=None, options=None):
        raise Exception(
            "Shouldn't call reset after getting the sub env with "
            "'get_sub_environments'. This is NOT supported"
        )

    def step(self, action):
        raise Exception(
            "Shouldn't call step after getting the sub env with "
            "'get_sub_environments'. This is NOT supported"
        )

    def close(self):
        pass
