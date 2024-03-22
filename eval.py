import re
from stable_baselines3 import *
import gymnasium as gym
from gymnasium import logger
from director import Director
from exp import birth_envs
from facade import Facade
import argparse
from stable_baselines3.common.evaluation import evaluate_policy
from guise import Guise
from coef import Coef
# load model and evaluate
if __name__ == '__main__':
    coef = Coef(
        n_timestep=10000,
        c_lr=0.0001,
        cap=1000,
        env_weights=[0.5, 0.5],
        n_envs=2,
        env_ids=["LunarLander-v2"],
        act_mapping=[{0: "NOOP", 1: "LEFT", 2: "UP",
                      3: "RIGHT"}],
        c_transition_loss=0.5,
        policy="MlpPolicy",
        eval_freq=1000,
        eval_episodes=1000,
        seed=123,
        device="cuda"
    )
    logger.set_level(logger.INFO)
    logger.info("👻")
    director = Director(coef)
    envs = birth_envs(coef.env_ids, coef.act_mapping)
    facade = Facade(envs, director=director)
    model = PPO(coef.policy, facade)
    model.load("models/v0.1")
    vec_env = model.get_env()
    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
        if dones:
            break
    # no fancy evaluation for now
    # std, mean = evaluate_policy(model, envs[0], n_eval_episodes=1000)
    # print(f"mean: {mean}, std: {std}")

    # no parser for now
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, required=True)
    # parser.add_argument("--env", type=str, required=True)
    # args = parser.parse_args()
    # model = PPO.load(args.model)
    # env = gym.make(args.env, render_mode= "human")
    # observation, info = env.reset()
    # mean, std = evaluate_policy(model, env, n_eval_episodes=1000)
    # print(f"mean: {mean}, std: {std}")
