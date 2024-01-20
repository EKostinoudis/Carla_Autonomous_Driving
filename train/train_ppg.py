import os
import argparse
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
import math

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from train.utils import get_config_path, update_to_abspath
from train.callback import (
    LogInfoCallback,
    NormValueInfoCallback,
)
from models.CILv2_multiview import g_conf, merge_with_yaml
from models.CILv2_multiview.CILv2_env import CILv2_env
from models.CILv2_multiview.CILv2_vec_env import CILv2_vec_env
from models.CILv2_multiview.CILv2_sub_env import CILv2_sub_env
from models.CILv2_multiview.CILv2_RLModule import CILv2_RLModule_PPG

from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.utils.from_config import NotProvided
from ray.tune.registry import register_trainable
from train.rllib_ppg import PPG
from train.ppg_trainer import PPGTorchLearner

register_trainable('PPG', PPG)

def main(args):
    conf_file = get_config_path(args.config)
    env_conf_file = get_config_path(args.env_config)

    conf = OmegaConf.load(conf_file)
    env_conf = OmegaConf.load(env_conf_file)

    # update to absolute paths
    conf_path_fields = ['path_to_conf', 'checkpoint_file']
    env_conf_path_fields = ['route', 'scenario_file']
    update_to_abspath(conf, conf_path_fields)
    update_to_abspath(env_conf, env_conf_path_fields)

    checkpoint_file = conf.checkpoint_file

    train_iters = conf.train_iters
    num_workers = conf.num_workers
    training_gpus = conf.training_gpus
    worker_gpus = conf.worker_gpus
    checkpoint_freq = conf.checkpoint_freq
    num_cpus_per_worker = conf.get('num_cpus_per_worker', 1)

    # these are useless because we have set:
    # _enable_rl_module_api=False and _enable_learner_api=False
    num_learner_workers = conf.get('num_learner_workers', 0)
    num_cpus_per_learner_worker = conf.get('num_cpus_per_learner_worker', 1)
    num_gpus_per_learner_worker = conf.get('num_gpus_per_learner_worker', 0)

    output_distribution = conf.get('output_distribution', 'gaussian')
    use_separate_vf = conf.get('use_separate_vf', False)
    use_stacked_model = conf.get('use_stacked_model', False)

    trainer = PPGTorchLearner
    rl_module = CILv2_RLModule_PPG


    extra_params = dict(conf.extra_params)
    list_params = ['lr', 'lr_schedule', 'entropy_coeff']
    for list_param in list_params:
        coeff = extra_params.get(list_param, None)
        if coeff is not None and isinstance(coeff, ListConfig):
            coeff = [tuple(item) for item in coeff]
            extra_params[list_param] = coeff

    path_to_conf = conf.path_to_conf

    env_name = 'CILv2_env'
    rollout_fragment_length = 'auto'
    if conf.get('use_vec_env', False):
        env_name = 'CILv2_vec_env'
        env_conf.update({'environments': num_workers})
        batch_size = conf.get('extra_params', {}).get('train_batch_size', None)
        if batch_size is None:
            raise Exception(
                'Set the number of "train_batch_size" in the "extra_params"'
                'in order to calculate the roolout fragment length.'
            )
        rollout_fragment_length = math.ceil(batch_size / num_workers)

        if conf.get('no_rollout_workers', False):
            # use same process for sampling and traing
            num_workers = 0
        else:
            num_workers = 1
    elif conf.get('use_sub_env', False): 
        env_name = 'CILv2_sub_env'

    # callback for the value target norm
    if conf.get('norm_target_value', False):
        callback = NormValueInfoCallback
        extra_params.update({'value_tartget_norm': True})
    else:
        extra_params.update({'value_tartget_norm': False})
        callback = LogInfoCallback

    tune.register_env(
        'CILv2_env',
        lambda rllib_conf: CILv2_env(env_conf, path_to_conf, rllib_conf),
    )
    tune.register_env(
        'CILv2_vec_env',
        lambda rllib_conf: CILv2_vec_env(env_conf, path_to_conf, rllib_conf),
    )
    tune.register_env(
        'CILv2_sub_env',
        lambda rllib_conf: CILv2_sub_env(env_conf, path_to_conf, rllib_conf),
    )

    # update g_conf
    merge_with_yaml(os.path.join(os.path.dirname(path_to_conf), 'CILv2.yaml'))


    ray.init()

    spec = SingleAgentRLModuleSpec(
        module_class=rl_module,
        model_config_dict={
            'g_conf': g_conf,
            'checkpoint': checkpoint_file,
            'pretrain_value': False,
            'output_distribution': output_distribution,
            'use_separate_vf': use_separate_vf,
            'use_stacked_model': use_stacked_model,
        },
    )
    _enable_rl_module_api = True
    _enable_learner_api = True
    training_model = NotProvided 

    config = (
        PPOConfig()
        .environment(env_name)
        .framework('torch')
        .training(
            model=training_model,
            _enable_learner_api=_enable_learner_api,
            learner_class=trainer,
        )
        .rollouts(
            num_rollout_workers=num_workers,
            rollout_fragment_length=rollout_fragment_length,
        )
        .resources(
            num_gpus=training_gpus,
            num_cpus_per_worker=num_cpus_per_worker,
            num_gpus_per_worker=worker_gpus,
            num_learner_workers=num_learner_workers,
            num_cpus_per_learner_worker=num_cpus_per_learner_worker,
            num_gpus_per_learner_worker=num_gpus_per_learner_worker,
        )
        .rl_module(
            rl_module_spec=spec,
            _enable_rl_module_api=_enable_rl_module_api,
        )
        .callbacks(callback)
    )
    config.update_from_dict(extra_params)

    results = tune.run(
        'PPG',
        name='ppg_train',
        config=config.to_dict(),
        stop={"training_iteration": train_iters},
        local_dir=os.path.abspath("ray_results"),
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the policy head of the model with IL.',
    )
    parser.add_argument('-c', '--config',
                        default='train_ppg_config.yaml',
                        help='Filename or whole path to the config',
                        type=str,
                        )
    parser.add_argument('-e', '--env_config',
                        default='environment_conf.yaml',
                        help='Filename or whole path to the environment config',
                        type=str,
                        )
    args = parser.parse_args()

    main(args)
