from __future__ import annotations

import random
from omegaconf import DictConfig
from typing import Optional
from multiprocessing.connection import Listener, Client
from multiprocessing import Process
from enum import Enum
import time
import gymnasium as gym
import logging
import traceback

from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.env.env_context import EnvContext

from .CILv2_env import CILv2_env

from environment.carla_launcher import CarlaLauncher

logger = logging.getLogger(__name__)

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
    def __init__(self, port, process):
        # connect to the processes
        self.conn = connect_to_port(port)

        # get the observation and action spaces
        msg = self.conn.recv()

        self.process = process
        self.observation_space = msg[0]
        self.action_space = msg[1]

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

class CILv2_sub_env(gym.Env):
    def __init__(self,
                 env_config: DictConfig | dict,
                 path_to_conf_file: str,
                 rllib_config: Optional[EnvContext] = None,
                 ):
        use_launcher = env_config.get('use_carla_launcher', False)
        self.port = env_config.get('port', 2000)

        # update the env for the environment
        if rllib_config is not None:
            offset = rllib_config.worker_index - 1
            seed = env_config.get('seed', random.randint(0, 10000)) + offset
            port = env_config.get('port', 2000) + 4*offset
            tm_port = env_config.get('tm_port', 8000) + offset
            env_config.update({
                'port': port,
                'tm_port': tm_port,
                'seed': seed,
                'use_carla_launcher': False,
            })
            self.port = port
        self.env_config = env_config
        self.path_to_conf_file = path_to_conf_file

        # create the carla launcher if needed
        if use_launcher:
            launch_script = env_config.get('carla_launch_script', None)
            if launch_script is None:
                raise ValueError('Must provide "carla_launch_script" in the environment config')
            self.carla_launcher = CarlaLauncher(
                env_config.get('port', 2000),
                launch_script,
                env_config.get('carla_restart_after', -1),
                launch_on_init=False,
            )
        else:
            self.carla_launcher = None

        self.port += 3
        self.env = None
        self.env_proc = None

        self.restart_env()

    def restart_env(self):
        if self.carla_launcher is not None: self.carla_launcher.lauch()

        while True:
            try:
                if self.env_proc is not None: self.env_proc.kill()
                try:
                    if self.env is not None: self.env.close()
                except Exception as e:
                    logger.warning(f'Restart env: env wrapper close: Got exception: {e}')
                    logger.warning(traceback.format_exc())
                self.env = None

                self.env_proc = Process(
                    target=handle_env,
                    args=(self.env_config, self.path_to_conf_file),
                )
                self.env_proc.start()
                self.env = EnvWrapper(self.port, self.env_proc)
                self.observation_space = self.env.observation_space
                self.action_space = self.env.action_space
            except Exception as e:
                logger.warning(f'Restart env: Got exception: {e}')
                logger.warning(traceback.format_exc())
            else:
                break


    def reset(self, *, seed=None, options=None):
        while True:
            try:
                state, info = self.env.reset(seed=seed, options=options)
            except Exception as e:
                logger.warning(f'Reset: Got exception: {e}')
                logger.warning(traceback.format_exc())
                # self.env.restart_server = True
                self.restart_env()
            else:
                break
        return state, info

    def step(self, action):
        try:
            state, reward, terminated, truncated, info = self.env.step(action)
        except Exception as e:
            logger.warning(f'Step: Got exception: {e}')
            logger.warning(traceback.format_exc())

            self.restart_env()

            state = self.observation_space.sample() 
            reward = 0.
            terminated = False
            truncated = True
            info = {}
        return state, reward, terminated, truncated, info

    def close(self):
        try:
            if self.carla_launcher is not None: self.carla_launcher.kill()
            try:
                if self.env is not None: self.env.close()
            except Exception as e:
                logger.warning(f'Close env: env wrapper close: Got exception: {e}')
                logger.warning(traceback.format_exc())
            self.env = None
            if self.env_proc is not None: self.env_proc.kill()
        except Exception as e:
            logger.warning(f'Close env: Got exception: {e}')
            logger.warning(traceback.format_exc())

