import torch  # Load the PyTorch library for loading the Tensor model and defining the computing network
from easydict import EasyDict  # Load EasyDict for instantiating configuration files
# Load configuration related components in DI-engine config module
from ding.config import compile_config
# Load environment related components in DI-engine env module
from ding.envs import DingEnvWrapper
# Load policy-related components in DI-engine policy module
from ding.policy import DQNPolicy, single_env_forward_wrapper
from ding.model import DQN  # Load model related components in DI-engine model module
# Load DI-zoo lunarlander environment and DQN algorithm related configurations
from dizoo.box2d.lunarlander.config.lunarlander_dqn_config import main_config, create_config
import gymnasium as gym
from director import Director
from gymnasium import logger
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
from gymnasium.wrappers.resize_observation import ResizeObservation
from coef import Coef
from facade import Facade
from stable_baselines3 import *
from guise import Guise
import numpy as np
import argparse
from copy import deepcopy
from ding.framework.middleware import online_logger
from ding.envs.env import DingEnvWrapper
from ding.policy import create_policy
from ding.envs import create_env_manager
from tensorboardX import SummaryWriter
from easydict import EasyDict

from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer

import os


def not_die_eval_o口O(main_config: EasyDict, create_config: EasyDict, ckpt_path: str):
    # Set the name of the experiment to be run in this deployment, which is the name of the project folder to be created
    main_config.exp_name += '_eval'
    # Compile and generate all configurations
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)

    coef = Coef("config/single_space_invader.toml")
    director = Director(coef)
    envs = director.birth_envs()
    env = DingEnvWrapper(gym.make("Facade/container-v0", envs=deepcopy(
        envs), director=deepcopy(director)), EasyDict(env_wrapper='default'))

    # evaluator_env_fn = [lambda: DingEnvWrapper(gym.make("Facade/container-v0", envs=deepcopy(
    #     envs), director=deepcopy(director))) for _ in range(cfg.env.evaluator_env_num)]
    # env = create_env_manager(
    #     cfg.env.manager, env_fn=evaluator_env_fn)
    # Enable the video recording of the environment and set the video saving folder
    # env.enable_save_replay(replay_path=f'./{main_config.exp_name}/video')
    policy = create_policy(cfg.policy, model=None, enable_field=[
        'eval'])
    policy.eval_mode.load_state_dict(torch.load(
        ckpt_path, map_location='cpu'))
    # tb_logger = SummaryWriter(os.path.join(
    #     './{}/log/'.format(cfg.exp_name), 'serial'))
    # evaluator = InteractionSerialEvaluator(
    #     cfg.policy.eval.evaluator, env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    # )
    # evaluator.eval()

    # Import model configuration, instantiate DQN model
    # Use the strategy decorator of the simple environment to decorate the decision method of the DQN strategy
    forward_fn = single_env_forward_wrapper(policy.eval_mode.forward)
    obs = env.reset()  # Reset the initialization environment to get the initial observations
    print(env)
    print(obs, obs.shape)
    returns = 0.  # Initialize total reward
    while True:  # Let the agent's strategy and environment interact cyclically until the end
        # According to the observed state, make a decision and generate action
        print(obs, obs.shape)
        action = forward_fn(obs)
        # Execute actions, interact with the environment, get the next observation state, the reward of this interaction, the signal of whether to end, and other information
        obs, rew, done, info = env.step(action)
        returns += rew  # Cumulative reward return
        if done:
            break
    print(f'Deploy is finished, final epsiode return is: {returns}')


def not_die_eval(main_config: EasyDict, create_config: EasyDict, ckpt_path: str):
    # Set the name of the experiment to be run in this deployment, which is the name of the project folder to be created
    main_config.exp_name += '_eval'
    # Compile and generate all configurations
    create_config.policy.type = create_config.policy.type + '_command'
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)

    coef = Coef(main_config.conf_path)
    for i in range(coef.n_envs):
        director = Director(coef)
        envs = director.birth_envs(eval=True)
        # env = DingEnvWrapper(gym.make("Facade/container-v0", envs=deepcopy(
        #     envs), director=deepcopy(director)))

        director.blend = False
        facade = gym.make("Facade/container-v0", envs=envs, director=director)
        env_name = coef.env_ids[i]
        logger.info(f"evaluating env {env_name}")
        director.set_eval(i)
        evaluator_env_fn = [lambda: DingEnvWrapper(
            facade) for _ in range(1)]
        env = create_env_manager(
            cfg.env.manager, env_fn=evaluator_env_fn)
        # Enable the video recording of the environment and set the video saving folder
        env.enable_save_replay(
            replay_path=f'./not_die_logs/{main_config.exp_name}/{env_name}_video')
        policy = create_policy(cfg.policy, model=None, enable_field=[
            'learn', 'collect', 'eval', 'command'])
        policy.eval_mode.load_state_dict(torch.load(
            ckpt_path, map_location='cpu'))
        evaluator = InteractionSerialEvaluator(
            cfg.policy.eval.evaluator, env, policy.eval_mode, exp_name=cfg.exp_name
        )
        evaluator.eval()
    return
    evaluator_env_fn = [lambda: DingEnvWrapper(gym.make("Facade/container-v0", envs=deepcopy(
        envs), director=deepcopy(director))) for _ in range(cfg.env.evaluator_env_num)]
    env = create_env_manager(
        cfg.env.manager, env_fn=evaluator_env_fn)
    # Enable the video recording of the environment and set the video saving folder
    env.enable_save_replay(
        replay_path=f'./not_die_logs/{main_config.exp_name}/video')
    policy = create_policy(cfg.policy, model=None, enable_field=[
                           'learn', 'collect', 'eval', 'command'])
    policy.eval_mode.load_state_dict(torch.load(
        ckpt_path, map_location='cpu'))
    tb_logger = SummaryWriter(os.path.join(
        './{}/log/'.format(cfg.exp_name), 'serial'))
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator.eval()

    return
    # Import model configuration, instantiate DQN model
    # Use the strategy decorator of the simple environment to decorate the decision method of the DQN strategy
    forward_fn = single_env_forward_wrapper(policy.forward)
    obs = env.reset()  # Reset the initialization environment to get the initial observations
    returns = 0.  # Initialize total reward
    while True:  # Let the agent's strategy and environment interact cyclically until the end
        # According to the observed state, make a decision and generate action
        action = forward_fn(obs)
        # Execute actions, interact with the environment, get the next observation state, the reward of this interaction, the signal of whether to end, and other information
        obs, rew, done, info = env.step(action)
        returns += rew  # Cumulative reward return
        if done:
            break
    print(f'Deploy is finished, final epsiode return is: {returns}')
