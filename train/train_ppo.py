import os
import torch
import argparse
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
import math

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import Policy
from ray.rllib.models import ModelCatalog

from train.utils import get_config_path, update_to_abspath
from train.callback import LogInfoCallback, NormValueInfoCallback

from models.CILv2_multiview import CIL_multiview_rllib, CIL_multiview_rllib_stack
from models.CILv2_multiview import g_conf, merge_with_yaml
from models.CILv2_multiview.CILv2_env import CILv2_env
from models.CILv2_multiview.CILv2_vec_env import CILv2_vec_env
from models.CILv2_multiview.CILv2_RLModule import CILv2_RLModule

from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.models.torch.torch_action_dist import TorchBeta
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.from_config import NotProvided

# TorchBeta had a bug, this a fix for the bug
class TorchBetaFixed(TorchBeta):
    def entropy(self) -> TensorType:
        return super().entropy().sum(-1)

VALID_MODELS = ["CIL_multiview_rllib", "CIL_multiview_rllib_stack"]
ModelCatalog.register_custom_model("CIL_multiview_rllib", CIL_multiview_rllib)
ModelCatalog.register_custom_model("CIL_multiview_rllib_stack", CIL_multiview_rllib_stack)
ModelCatalog.register_custom_action_dist("beta", TorchBetaFixed)


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

    # set the pretrained file to the checkpoint plus the "_value_pretrained"
    # string in the end (before the extension)
    base, ext = os.path.splitext(checkpoint_file)
    pretrain_file = base + '_value_pretrained' + ext
    pretrain_complete_episodes = conf.get('pretrain_complete_episodes', False)

    # for pretraining the value function
    pretrain_value = conf.pretrain_value
    pretrain_iters = conf.pretrain_iters

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

    use_rl_module = conf.get('use_rl_module', False)

    output_distribution = conf.get('output_distribution', 'gaussian')

    # callback
    if conf.get('norm_target_value', False):
        callback = NormValueInfoCallback
    else:
        callback = LogInfoCallback

    extra_params = dict(conf.extra_params)
    entropy_coeff = extra_params.get('entropy_coeff', None)
    if entropy_coeff is not None and isinstance(entropy_coeff, ListConfig):
        entropy_coeff = [tuple(item) for item in entropy_coeff]
        extra_params['entropy_coeff'] = entropy_coeff

    path_to_conf = conf.path_to_conf

    model_name: str = conf.model
    if model_name not in VALID_MODELS:
        raise ValueError(f'model: {model_name} is not a valid model. Valid models: {VALID_MODELS}')

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

    tune.register_env(
        'CILv2_env',
        lambda rllib_conf: CILv2_env(env_conf, path_to_conf, rllib_conf),
    )
    tune.register_env(
        'CILv2_vec_env',
        lambda rllib_conf: CILv2_vec_env(env_conf, path_to_conf, rllib_conf),
    )

    # update g_conf
    merge_with_yaml(os.path.join(os.path.dirname(path_to_conf), 'CILv2.yaml'))


        

    ray.init()

    #################################################################
    # Pretrain code
    #################################################################
    if pretrain_value:
        if use_rl_module:
            spec = SingleAgentRLModuleSpec(
                module_class=CILv2_RLModule,
                model_config_dict={
                    'g_conf': g_conf,
                    'checkpoint': checkpoint_file,
                    'pretrain_value': pretrain_value,
                },
            )
            _enable_rl_module_api = True
            _enable_learner_api = True
            training_model = NotProvided 
        else:
            spec = NotProvided
            _enable_rl_module_api = False
            _enable_learner_api = False
            training_model = { 
                "custom_model": model_name,
                "custom_action_dist": "beta",
                "custom_model_config": {
                    'g_conf': g_conf,
                    'checkpoint': checkpoint_file,
                    'pretrain_value': pretrain_value,
                    'output_distribution': output_distribution,
                },
            }

        config = (
            PPOConfig()
            .environment(env_name)
            .framework('torch')
            .training(
                model=training_model,
                _enable_learner_api=_enable_learner_api,
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
        if pretrain_complete_episodes:
            config.update_from_dict({'batch_mode': 'complete_episodes'})


        # update the learning rate if given
        lr_pretrain = conf.get('lr_pretrain', None)
        if lr_pretrain is not None:
            config.update_from_dict({'lr': lr_pretrain})

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
        state_dict = {k[6:] if k.startswith('model.') else k: v for k, v in model.state_dict().items()}
        torch.save({'model': state_dict}, pretrain_file)

        # update the checkpoint file
        checkpoint_file = pretrain_file

    #################################################################
    # Enf of pretrain code
    #################################################################
    if use_rl_module:
        spec = SingleAgentRLModuleSpec(
            module_class=CILv2_RLModule,
            model_config_dict={
                'g_conf': g_conf,
                'checkpoint': checkpoint_file,
                'pretrain_value': False,
            },
        )
        _enable_rl_module_api = True
        _enable_learner_api = True
        training_model = NotProvided 
    else:
        spec = NotProvided
        _enable_rl_module_api = False
        _enable_learner_api = False
        training_model = { 
            "custom_model": model_name,
            "custom_action_dist": "beta",
            "custom_model_config": {
                'g_conf': g_conf,
                'checkpoint': checkpoint_file,
                'pretrain_value': False,
                'output_distribution': output_distribution,
            },
        }

    config = (
        PPOConfig()
        .environment(env_name)
        .framework('torch')
        .training(
            model=training_model,
            _enable_learner_api=_enable_learner_api,
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
