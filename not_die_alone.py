from typing import Union, Optional, List, Any, Tuple
import os
from pkg_resources import evaluate_marker
import torch
from ditk import logging
from functools import partial
from tensorboardX import SummaryWriter
from copy import deepcopy

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector, create_serial_evaluator
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed, get_rank
from ding.entry.utils import random_collect
from ditk import logging
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
from ding.envs import SubprocessEnvManagerV2, BaseEnvManagerV2
from ding.model import GTrXLDQN
from ding.policy import R2D2GTrXLPolicy
from config.die_r2d2_gtrxl import spaceinvaders_r2d2_gtrxl_config, spaceinvaders_r2d2_gtrxl_create_config
from ding.config import compile_config
from ding.data import DequeBuffer
import env_container
from ding.framework import ding_init
from tensorboardX import SummaryWriter
from ding.utils import set_pkg_seed, get_rank
from ding.worker import create_buffer
import os
from ding.policy import create_policy


def not_alone(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
        dynamic_seed: Optional[bool] = True,
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry for off-policy RL.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.
        - max_env_step (:obj:`Optional[int]`): Maximum collected environment interaction steps.
        - dynamic_seed(:obj:`Optional[bool]`): set dynamic seed for collector.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(cfg, seed=seed, env=env_fn,
                         auto=True, create_cfg=create_cfg, save_cfg=True)
    # Create main components: env, policy
    # if env_setting is None:
    #     env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    # else:
    #     env_fn, collector_env_cfg, evaluator_env_cfg = env_setting
    # collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    # evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    coef = Coef(cfg.conf_path)
    director = Director(coef)
    envs = director.birth_envs()
    collector_env_fn = [lambda: DingEnvWrapper(gym.make("Facade/container-v0", envs=deepcopy(
        envs), director=deepcopy(director))) for _ in range(cfg.env.collector_env_num)]
    evaluator_env_fn = [lambda: DingEnvWrapper(gym.make("Facade/container-v0", envs=deepcopy(
        envs), director=deepcopy(director))) for _ in range(cfg.env.evaluator_env_num)]
    collector_env = create_env_manager(
        cfg.env.manager, env_fn=collector_env_fn)
    evaluator_env = create_env_manager(
        cfg.env.manager, env_fn=evaluator_env_fn)
    collector_env.seed(cfg.seed, dynamic_seed=dynamic_seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=[
                           'learn', 'collect', 'eval', 'command'])
    if cfg.policy.get('load_path', None) is not None:
        print('🪅')
        policy._load_state_dict_learn(torch.load(cfg.policy.load_path))
    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join(
        './logs/{}/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None
    learner = BaseLearner(cfg.policy.learn.learner,
                          policy.learn_mode, tb_logger, exp_name=f"not_die_logs/{cfg.exp_name}")
    collector = create_serial_collector(
        cfg.policy.collect.collector,
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name
    )
    evaluator = create_serial_evaluator(
        cfg.policy.eval.evaluator,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name
    )
    replay_buffer = create_buffer(
        cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)
    commander = BaseSerialCommander(
        cfg.policy.other.commander, learner, collector, evaluator, replay_buffer, policy.command_mode
    )
    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')

    # Accumulate plenty of data at the beginning of training.
    if cfg.policy.get('random_collect_size', 0) > 0:
        random_collect(cfg.policy, policy, collector,
                       collector_env, commander, replay_buffer)
    while True:
        collect_kwargs = commander.step()
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, eval_info = evaluator.eval(
                learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        # Collect data by default config n_sample/n_episode
        new_data = collector.collect(
            train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # Learn policy from collected data
        for i in range(cfg.policy.learn.update_per_collect):
            # Learner will train ``update_per_collect`` times in one iteration.
            train_data = replay_buffer.sample(
                learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is None:
                # It is possible that replay buffer's data count is too few to train ``update_per_collect`` times
                logging.warning(
                    "Replay buffer's data can only train for {} steps. ".format(i) +
                    "You can modify data collect config, e.g. increasing n_sample, n_episode."
                )
                break
            learner.train(train_data, collector.envstep)
            if learner.policy.get_attribute('priority'):
                replay_buffer.update(learner.priority_info)
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    # Learner's after_run hook.
    learner.call_hook('after_run')
    if get_rank() == 0:
        import time
        import pickle
        import numpy as np
        with open(os.path.join(cfg.exp_name, 'result.pkl'), 'wb') as f:
            eval_value_raw = eval_info['eval_episode_return']
            final_data = {
                'stop': stop,
                'env_step': collector.envstep,
                'train_iter': learner.train_iter,
                'eval_value': np.mean(eval_value_raw),
                'eval_value_raw': eval_value_raw,
                'finish_time': time.ctime(),
            }
            pickle.dump(final_data, f)
    return policy


if __name__ == '__main__':
    not_alone((spaceinvaders_r2d2_gtrxl_config,
               spaceinvaders_r2d2_gtrxl_create_config), seed=0, max_env_step=30000000)
