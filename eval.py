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
# load model and evaluate


def eval_exp(config_path, model_path, env_id=-1, episodes=1000,  render=False):
    """Evaluates the experiment
    """
    coef = Coef(config_path)
    logger.set_level(logger.INFO)
    if env_id == -1:
        logger.info("evaluating all envs")
        for i in range(coef.n_envs):
            _coef = deepcopy(coef)
            _coef.env_ids = [coef.env_ids[i]]
            _coef.act_mapping = [coef.act_mapping[i]]
            _coef.n_envs = 1
            eval_model(_coef, model_path, episodes, render)
    else:
        logger.info(f"evaluating env {env_id}")
        coef.env_ids = [coef.env_ids[env_id]]
        coef.act_mapping = [coef.act_mapping[env_id]]
        coef.n_envs = 1
        eval_model(coef, model_path, episodes, render)


def eval_model(coef: Coef, model_path, episodes=1000,  render=False):
    """Evaluates the model
    """
    director = Director(coef)
    envs = director.birth_envs()
    facade = Facade(envs, director=director)
    model = coef.algorithm(coef.policy, facade)
    model.load(model_path)
    env_name = re.sub('[^0-9a-zA-Z]+', '_', coef.env_ids[0])
    screens = []

    def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        screen = facade.render()
        screens.append(screen)
    # if render:
    #     render_env(model, episodes)
    vec_env = model.get_env()
    vec_env.reset()
    # render_env(model, 1000)
    # return
    # yes fancy evaluation for now
    std, mean = evaluate_policy(
        model, vec_env, n_eval_episodes=episodes, callback=grab_screens)
    if render:
        path = f"logs/videos/{re.sub('[^0-9a-zA-Z]+', '_', model_path)}"
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


def render_env(model, episodes=1000):
    """Renders the environment
    """
    vec_env = model.get_env()
    obs = vec_env.reset()
    for _ in range(episodes):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
        if dones:
            obs = vec_env.reset()
            # break
    vec_env.close()


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
    eval_exp("config/"+args.config, "models/"+args.model,
             args.env_id, args.episodes, args.render)
