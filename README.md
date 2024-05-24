# Carla Autonomous Driving

This repository contains code to train a agent in the
[Carla](https://carla.org/) simulator. Specifically, we use both Imitation
Learning (IL) and Reinforcement Learning (RL) techniques to train an agent. The
objective it to use IL to train a basic agent and the RL to fine tune it in
order to achieve better performance.

## Model Architectures

This repository supports only the
[CIL++](https://github.com/yixiao1/CILv2_multiview) architecture for the agent
(with many variations). To use this model, you must download it from the link
([_results.tar.gz](https://drive.google.com/file/d/1GLo5mVrmyNsb5pLqksYnjR8fN1-ZptHE/view?usp=sharing))
that is provided in the GitHub repo. After that you must extract the data in the
`models/CILv2_multiview/_results` folder. You can use the commands:

```bash
mkdir -p models/CILv2_multiview/_results
tar -zxvf _results.tar.gz -C models/CILv2_multiview/_results
```

### Using other model

If you want to use the environment to train your model,
take a look at [CILv2_env](models/CILv2_multiview/CILv2_env.py) which is the
environment for the CIL++ architecture.

We recommend to use a wrapper class (like in the
[CILv2_sub_env](models/CILv2_multiview/CILv2_sub_env.py)) to run it a separate
process to avoid ending the training when the Carla server crashes. To do this
just copy this file and change the environment class.

# Training

As we previously mentioned, there are two training faces:

1. Imitation Learning
2. Reinforcement Leaning

In both cases you must install the requirements that we provide. In order to do
this, we recommend to create first a Python 3.10 virtual environment (with the
program of chaise). Then use pip install the requirements with the command:

```bash
pip install -r requirements.txt
```

## Imitation Learning

There are two options for the IL training. These options are:

1. [policy_head_training.py](train/policy_head_training.py) which uses the
   [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)
   to train the model.
2. [policy_head_training_accelerate.py](train/policy_head_training_accelerate.py)
which uses the [Accelerate](https://huggingface.co/docs/accelerate/index)
library to train the model. These library uses Distributed Data Parallel in the
background. This is the recommended option!

Both these option can be configured using the configuration file
[IL_CIL_multiview_actor_critic.yaml](train/configs/IL_CIL_multiview_actor_critic.yaml).

We provide the [script](train/launch_policy_head_training_accelerate.sh) for an
easier execution of the training (works only for the 2nd option with the
Accelerate library). In order to launch the training run:

```bash
bash train/launch_policy_head_training_accelerate.sh
```

Also you can provide flags of the python script in this bash script, for
example:

```bash
bash train/launch_policy_head_training_accelerate.sh --clean --all-weights
```

The data that we used in the phase are the data from the
[CIL++](https://github.com/yixiao1/CILv2_multiview). Specicaly, we used the part
7-14 for the training. We used to part 6 for the evaluation, but we
recommend to use other or more data for a proper evaluation.

## Reinforcement Learning

For the RL we used Carla 0.9.15. You can use the
[GitHub releases](https://github.com/carla-simulator/carla/releases/tag/0.9.15) in order to
download this version, or take a look at the [Carla
website](https://carla.readthedocs.io/en/latest/start_quickstart/#carla-installation)
for installation instructions.

**NOTE**: To avoid adding files to the path for the Carla, we use the pip
package for the Carla and we copied the `agents` folder in this repository.

For the RL training we use the
[RLlib](https://docs.ray.io/en/latest/rllib/index.html) library. For the RL
training we support two algorithms:

1. Proximal Policy Optimization (PPO) provided by the RLlib and enhanced with
   more options by us.
2. Phasic Policy Gradient (PPG) we implemented this algorithm on to of the PPO
   implementation of the RLlib. If we want to see how we impemented it, take a
   look at the PPG related files in the `train` folder.

### PPO

The PPO algorithm can be configured from the
[train_ppo_config.yaml](train/configs/train_ppo_config.yaml) file. This file
controls the algorithm related option for the training. Take a look at the file
for the training options.

In order to run the training you can use the command:

```bash
RAY_DEDUP_LOGS=0 PYTHONPATH=. python3 train/train_ppo.py
```

**NOTE**: Take a look bellow for option for the environment.

### PPG

As in PPO algorithm, the PPG algorithm can be configured from the
[train_ppg_config.yaml](train/configs/train_ppg_config.yaml) file. This file
controls the algorithm related option for the training. Take a look at the file
for the training options. 

We should mention that this is a "hacked" version of the algorithm just to work
in our needs for this project.

In order to run the training you can use the command:

```bash
RAY_DEDUP_LOGS=0 PYTHONPATH=. python3 train/train_ppg.py
```

**NOTE**: Take a look bellow for option for the environment.

### Environment configuration

For the environment configuration we use a different file
[environment_conf.yaml](train/configs/environment_conf.yaml). This configuration
file controls options such as the reward, the scenarios or routes the we train
the agent, server options and more.

We support 3 different option for routes-scenarios for training the agent. These
are:

1. **Routes**: These are the routes that the [scenario
   runner](https://github.com/carla-simulator/scenario_runner) provides. These
   are predefined routes with various challenges for the agent. It can
   configured using the option `run_type: route`
2. **Scenarios**: These are also provided by the [scenario
   runner](https://github.com/carla-simulator/scenario_runner). It can be used
   by the option `run_type: scenario`. We didn't used this option for training,
   but if you are interested you can use it.
3. **Free ride**: In these case the agent ride in the from a random generated
   route. This can be configured with `run_type: free ride` (or any value except
   `route` and `scenario`).

For all these option, in the configuration file you find many option in order to
configured the training in your needs.

# Acknowledgements

This repository contains code from various sources:

* [scenario runner](https://github.com/carla-simulator/scenario_runner)
* [CIL++](https://github.com/yixiao1/CILv2_multiview)
* [Roach](https://github.com/zhejz/carla-roach)
* [agents](https://github.com/carla-simulator/carla)
