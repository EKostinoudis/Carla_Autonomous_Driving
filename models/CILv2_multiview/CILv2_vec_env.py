from __future__ import annotations

import random
from omegaconf import DictConfig
from typing import Optional
from multiprocessing.connection import Listener, Client
from multiprocessing import Process
from enum import Enum
import time

from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.env.env_context import EnvContext

from .CILv2_env import CILv2_env

class EnvCommand(Enum):
    init = 0
    reset = 1
    step = 2
    close = 3


def handle_env(env_config: DictConfig | dict, path_to_conf_file: str):
    port = env_config.get('port', 2000) + 3

    listener = Listener(('localhost', port))
    conn = listener.accept()

    env = CILv2_env(env_config, path_to_conf_file)

    # sent the observation_space and action_space
    conn.send((env.observation_space, env.action_space))

    while True:
        commmand, msg = conn.recv()

        if commmand == EnvCommand.reset:
            ret = env.reset(**msg)
            conn.send(ret)
        elif commmand == EnvCommand.step:
            ret = env.step(msg)
            conn.send(ret)
        elif commmand == EnvCommand.close:
            env.close()
            conn.send(True)
            conn.close()
            break

    listener.close()


def connect_to_port(port, timeout=5):
    start = time.perf_counter()

    while True:
        try:
            conn = Client(('localhost', port))
            return conn
        except:
            if time.perf_counter() - start < timeout:
                time.sleep(.1)
            else:
                Exception(f"Timeout ({timeout}) for port: {port}")


class EnvWrapper():
    def __init__(self, conn, process: Process, observation_space, actions_space):
        self.conn = conn
        self.process = process
        self.observation_space = observation_space
        self.action_space = actions_space

    def reset(self, *, seed=None, options=None):
        self.conn.send((EnvCommand.reset, {'seed': seed, 'options': options}))
        return self.conn.recv()

    def step(self, action):
        self.conn.send((EnvCommand.step, action))
        return self.conn.recv()

    def close(self):
        self.conn.send((EnvCommand.close, None))
        self.conn.recv()
        self.conn.close()

        # wait for process to end before closing
        self.process.join()
        self.process.close()

class CILv2_vec_env(VectorEnv):
    def __init__(self,
                 env_config: DictConfig | dict,
                 path_to_conf_file: str,
                 rllib_config: Optional[EnvContext] = None,
                 ):
        num_envs = env_config.get('environments', 1)

        self.proceses = []
        ports = []
        for offset in range(num_envs):
            seed = env_config.get('seed', random.randint(0, 10000)) + offset
            port = env_config.get('port', 2000) + 4*offset
            tm_port = env_config.get('tm_port', 8000) + offset
            new_env_config = {**env_config, 'port': port, 'tm_port': tm_port, 'seed': seed}

            # start new process to handle the env
            p = Process(target=handle_env, args=(new_env_config, path_to_conf_file))
            p.start()
            self.proceses.append(p)
            ports.append(port + 3)

        # connect to the processes
        self.connections = [connect_to_port(port) for port in ports]

        # get the observation and action spaces
        msg = [conn.recv() for conn in self.connections]

        # Create the fake envs
        self.envs = [EnvWrapper(conn, p, msg[0][0], msg[0][1]) \
            for conn, p in zip(self.connections, self.proceses)]

        super().__init__(
            observation_space=msg[0][0],
            action_space=msg[0][1],
            num_envs=num_envs,
        )

    def vector_reset(self, *, seeds=None, options=None):
        seeds = seeds or [None for _ in range(self.num_envs)]
        options = options or [None for _ in range(self.num_envs)]
        for conn, seed, option in zip(self.connections, seeds, options):
            conn.send((EnvCommand.reset, {'seed': seed, 'options': option}))

        temp = [conn.recv() for conn in self.connections]
        obs, info = zip(*temp)
        obs, info = list(obs), list(info)
        return obs, info

    def reset_at(self, index, *, seed=None, options=None):
        return self.envs[index].reset(seed=seed, options=options)

    def vector_step(self, actions):
        for conn, action in zip(self.connections, actions):
            conn.send((EnvCommand.step, action))

        temp = [conn.recv() for conn in self.connections]
        obs_batch, rew_batch, terminated_batch, truncated_batch, info_batch = zip(*temp)
        obs_batch, rew_batch, terminated_batch, truncated_batch, info_batch = \
            list(obs_batch), list(rew_batch), list(terminated_batch), list(truncated_batch), list(info_batch)
        return obs_batch, rew_batch, terminated_batch, truncated_batch, info_batch

    def get_sub_environments(self):
        return self.envs

