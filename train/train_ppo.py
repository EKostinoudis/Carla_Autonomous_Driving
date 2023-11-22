import os
import torch
import argparse
from omegaconf import OmegaConf

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import Policy
from ray.rllib.models import ModelCatalog

from train.utils import get_config_path, update_to_abspath

from models.CILv2_multiview import CIL_multiview_rllib, CIL_multiview_rllib_stack
from models.CILv2_multiview import g_conf, merge_with_yaml
from models.CILv2_multiview.CILv2_env import CILv2_env
from ray.rllib.models.torch.torch_action_dist import TorchBeta

VALID_MODELS = ["CIL_multiview_rllib", "CIL_multiview_rllib_stack"]
ModelCatalog.register_custom_model("CIL_multiview_rllib", CIL_multiview_rllib)
ModelCatalog.register_custom_model("CIL_multiview_rllib_stack", CIL_multiview_rllib_stack)
ModelCatalog.register_custom_action_dist("beta", TorchBeta)


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

    path_to_conf = conf.path_to_conf
    tune.register_env(
        'CILv2_env',
        lambda rllib_conf: CILv2_env(env_conf, path_to_conf, rllib_conf),
    )

    model = conf.model
    if model not in VALID_MODELS:
        raise ValueError(f'model: {model} is not a valid model. Valid models: {VALID_MODELS}')

    checkpoint_file = conf.checkpoint_file

    # set the pretrained file to the checkpoint plus the "_value_pretrained"
    # string in the end (before the extension)
    base, ext = os.path.splitext(checkpoint_file)
    pretrained_file = base + '_value_pretrained' + ext

    # for pretraining the value function
    pretrain_value = conf.pretrain_value
    pretrain_iters = conf.pretrain_iters

    train_iters = conf.train_iters
    num_workers = conf.num_workers
    training_gpus = conf.training_gpus
    worker_gpus = conf.worker_gpus
    checkpoint_freq = conf.checkpoint_freq

    extra_params = conf.extra_params

    # update g_conf
    merge_with_yaml(os.path.join(os.path.dirname(path_to_conf), 'CILv2.yaml'))

    num_cpus = conf.get('num_cpus', None)
    num_gpus = conf.get('num_gpus', None)

    # specify the number of cpus because on a SLURM env it crashes
    if num_cpus is None:
        num_cpus = num_workers + 1

    # ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
    ray.init()

    if pretrain_value:
        config = (
            PPOConfig()
            .environment('CILv2_env')
            .framework('torch')
            .training(
                model={ 
                    "custom_model": model,
                    "custom_action_dist": "beta",
                    "custom_model_config": {
                        'g_conf': g_conf,
                        'checkpoint': checkpoint_file,
                        'pretrain_value': True,
                    },
                },
            )
            .rollouts(num_rollout_workers=num_workers)
            .resources(
                num_gpus=training_gpus,
                num_gpus_per_worker=worker_gpus,
            )
            .rl_module(_enable_rl_module_api=False)
            .training(_enable_learner_api=False)
        )
        config.update_from_dict(extra_params)

        results = tune.run(
            'PPO',
            name='ppo_train/value_pretrain',
            config=config.to_dict(),
            stop={"training_iteration": pretrain_iters},
            local_dir=os.path.abspath("ray_results"),
            checkpoint_freq=checkpoint_freq,
            checkpoint_at_end=True,
        )

        # create the proper path for the policy checkpoint
        checkpoint = os.path.join(
            results.get_last_checkpoint().path,
            'policies',
            'default_policy',
        )
        model = Policy.from_checkpoint(checkpoint).model
        torch.save({'model': model.state_dict()}, pretrained_file)

        # update the checkpoint file
        checkpoint_file = pretrained_file

    config = (
        PPOConfig()
        .environment('CILv2_env')
        .framework('torch')
        .training(model={ 
            "custom_model": model,
            "custom_action_dist": "beta",
            "custom_model_config": {
                'g_conf': g_conf,
                'checkpoint': checkpoint_file,
                'pretrain_value': False,
            },
        })
        .rollouts(num_rollout_workers=num_workers)
        .resources(
            num_gpus=training_gpus,
            num_gpus_per_worker=worker_gpus,
        )
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
    )
    config.update_from_dict(extra_params)

    results = tune.run(
        'PPO',
        name='ppo_train/main',
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
                        default='train_ppo_config.yaml',
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
