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
import os
from ding.policy import create_policy


def main():
    """Main function to run the experiment
    """

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, required=True)
    # parser.add_argument("--config", type=str, required=True)
    # args = parser.parse_args()

    # coef = Coef("config/" + args.config)
    coef = Coef("config/single_space_invader.toml")
    logger.set_level(logger.INFO)
    logger.info("👻")
    director = Director(coef)
    envs = director.birth_envs()
    # facade = gym.make("Facade/container-v0", envs=envs, director=director)
    # ding = DingEnvWrapper(facade)
    cfg = compile_config(spaceinvaders_r2d2_gtrxl_config,
                         create_cfg=spaceinvaders_r2d2_gtrxl_create_config, auto=True)
    ding_init(cfg)
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = SubprocessEnvManagerV2(
        env_fn=[lambda: DingEnvWrapper(gym.make("Facade/container-v0", envs=deepcopy(envs), director=deepcopy(director))) for _ in range(collector_env_num)], cfg=cfg.env.manager
    )
    print("🫸")
    evaluator_director = Director(coef)
    evaluator_envs = evaluator_director.birth_envs()
    evaluator_env = SubprocessEnvManagerV2(
        env_fn=[lambda: DingEnvWrapper(gym.make("Facade/container-v0", envs=deepcopy(evaluator_envs), director=deepcopy(evaluator_director))) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )
    # model = DQN(ding.observation_space.shape[::-1], int(ding.action_space.n))
    buffer_ = DequeBuffer(
        size=cfg.policy.other.replay_buffer.replay_buffer_size)
    # policy = R2D2GTrXLPolicy(cfg.policy, model=model)
    policy = create_policy(cfg.policy, enable_field=[
                           'learn', 'collect', 'eval'])
    from ding.framework import task
    from ding.framework.context import OnlineRLContext
    from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, eps_greedy_handler, CkptSaver, nstep_reward_enhancer
    filename = '{}/log.txt'.format(cfg.exp_name)
    logging.getLogger(with_files=[filename]).setLevel(logging.INFO)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        # Evaluating, we place it on the first place to get the score of the random model as a benchmark value
        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        # Decay probability of explore-exploit
        task.use(eps_greedy_handler(cfg))
        # Collect environmental data
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        # Prepare nstep reward for training
        task.use(nstep_reward_enhancer(cfg))
        task.use(data_pusher(cfg, buffer_))  # Push data to buffer
        # Train the model
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
        # Save the model
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=3000000))
        task.use(online_logger(train_show_freq=1000))
        # In the evaluation process, if the model is found to have exceeded the convergence value, it will end early here
        task.run()
    # ding.enable_save_replay("replays/")
    # obs = ding.reset()
    # while True:
    #     action = ding.random_action()
    #     timestep = ding.step(int(action))
    #     if timestep.done:
    #         break

    # policy = SQLPolicy(model, ding.action_space)
    # model = coef.algorithm(policy=coef.policy, env=facade,
    #                        tensorboard_log="logs/", seed=coef.seed,)

    # director.set_model(model)
    # director.learn()
    print("Learning done")
    # director.save("models/" + args.model)


if __name__ == '__main__':
    main()
