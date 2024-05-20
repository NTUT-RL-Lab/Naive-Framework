from copy import deepcopy
import re
from stable_baselines3 import *
import gymnasium as gym
from gymnasium import logger
from director import Director
from facade import Facade
import argparse
from stable_baselines3.common.evaluation import evaluate_policy
from guise import Guise
from coef import Coef
from typing import Any, Dict
import numpy as np
# load model and evaluate


def eval_exp(config_path, model_path, env_id=-1, episodes=1000,  render=False):
    """Evaluates the experiment
    """
    logger.set_level(logger.INFO)
    coef = Coef(config_path)
    director = Director(coef)
    envs = director.birth_envs()
    facade = Facade(envs, director=director)
    model = coef.algorithm(policy=coef.policy, env=facade, seed=coef.seed)
    model.load(model_path)
    if env_id == -1:
        logger.info("evaluating all envs")
        for i in range(coef.n_envs):
            env_name = coef.env_ids[i]
            logger.info(f"evaluating env {env_name}")
            director.set_eval(i)
            eval_model(model, re.sub(
                '[^0-9a-zA-Z]+', '_', model_path), re.sub('[^0-9a-zA-Z]+', '_', env_name), facade, episodes, render)
    else:
        # WONTFIX
        logger.info(f"evaluating env {env_id}")
        coef.env_ids = [coef.env_ids[env_id]]
        coef.act_mapping = [coef.act_mapping[env_id]]
        coef.n_envs = 1
        eval_model(coef, model_path, envs[i], episodes, render)


def eval_model(model, model_name, env_name, facade: Facade, episodes=1000,  render=False):
    """Evaluates the model
    """
    screens = []

    def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        screen = facade.render()
        screens.append(screen)
    if render:
        render_env(model, facade, env_name, model_name, episodes=1000)
    return
    # yes fancy evaluation for now
    vec_env = model.get_env()
    vec_env.reset()
    std, mean = evaluate_policy(
        model, facade, n_eval_episodes=episodes, callback=grab_screens)
    if render:
        logger.info(f"rendering video for {env_name}")
        path = f"logs/videos/{model_name}"
        import os
        import cv2
        if not os.path.exists(path):
            os.makedirs(path)
        # save video
        height, width, _ = screens[0].shape
        out = cv2.VideoWriter(
            f"{path}/{env_name}.avi", cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
        for screen in screens:
            out.write(cv2.cvtColor(screen, cv2.COLOR_RGB2BGR))
        out.release()

    print(f"mean: {mean}, std: {std}")
    return mean, std


def render_env(model, facade: Facade, env_name, model_name, episodes=1000):
    """Renders the environment
    """
    rewards = []
    screens = []
    # obs = vec_env.reset()
    obs, info = facade.reset()
    temp = 0
    while True:
        action, _states = model.predict(obs, )
        # logger.info(f"action: {action}")
        # obs, reward, dones, info = vec_env.step(action)
        obs, reward, dones, truncated, info = facade.step(action)
        temp += reward
        # vec_env.render("human")
        screens.append(facade.render())
        if dones or truncated:
            # obs = vec_env.reset()
            facade.reset()
            rewards.append(temp)
            temp = 0
            break
    # rewards.append(temp)
    # vec_env.close()
    facade.close()

    logger.info(f"rendering video for {env_name}")
    path = f"logs/videos/{model_name}"
    import os
    import cv2
    if not os.path.exists(path):
        os.makedirs(path)
    # save video
    height, width, _ = screens[0].shape
    out = cv2.VideoWriter(
        f"{path}/{env_name}.avi", cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
    for screen in screens:
        out.write(cv2.cvtColor(screen, cv2.COLOR_RGB2BGR))
    out.release()
    logger.info(
        f"mean reward: {sum(rewards)/len(rewards)}, std: {np.std(rewards)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--env_id", type=int, default=-1,
                        help="environment id to evaluate", required=False)
    parser.add_argument("--episodes", type=int, default=10,
                        help="number of episodes to evaluate", required=False)
    parser.add_argument("--render", type=bool, default=False,
                        help="render the evaluation", required=False)
    args = parser.parse_args()
    eval_exp("config/"+args.config, "models/"+args.model, args.env_id,
             args.episodes, args.render)
