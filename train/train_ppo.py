import os
import torch
import argparse
from omegaconf import OmegaConf

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import Policy
from ray.rllib.models import ModelCatalog

from models.CILv2_multiview import CILv2_env, CIL_multiview_rllib, CIL_multiview_rllib_stack

VALID_MODELS = ["CIL_multiview_rllib", "CIL_multiview_rllib_stack"]
ModelCatalog.register_custom_model("CIL_multiview_rllib", CIL_multiview_rllib)
ModelCatalog.register_custom_model("CIL_multiview_rllib_stack", CIL_multiview_rllib_stack)

def main(args):
    if os.path.sep in args.config:
        conf_file = args.config
    else:
        conf_file = os.path.join(*'./train/configs'.split('/'), args.config)

    conf = OmegaConf.load(conf_file)

    model = conf.model
    if model not in VALID_MODELS:
        raise ValueError(f'model: {model} is not a valid model. Valid models: {VALID_MODELS}')

    checkpoint_file = os.path.abspath(conf.checkpoint_file)

    # for pretraining the value function
    pretrain_value = conf.pretrain_value
    pretrained_file = os.path.abspath(conf.pretrained_file)
    pretrain_iters = conf.pretrain_iters

    train_iters = conf.train_iters
    num_workers = conf.num_workers
    training_gpus = conf.training_gpus
    worker_gpus = conf.worker_gpus
    checkpoint_freq = conf.checkpoint_freq

    extra_params = conf.extra_params


    # TODO: add cpus and gpus???
    ray.init()

    if pretrain_value:
        config = (
            PPOConfig()
            .environment(CILv2_env)
            .framework('torch')
            .training(
                model={ 
                    "custom_model": "my_custom_model",
                    "custom_model_config": {
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
        .environment(CILv2_env)
        .framework('torch')
        .training(model={ 
            "custom_model": "my_custom_model",
            "custom_model_config": {
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
    args = parser.parse_args()

    main(args)
