from copy import deepcopy
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
    envs = birth_envs(coef.env_ids, coef.act_mapping)
    facade = Facade(envs, director=director)
    model = PPO(coef.policy, facade)
    model.load(model_path)
    vec_env = model.get_env()
    vec_env.reset()
    # yes fancy evaluation for now
    std, mean = evaluate_policy(
        model, vec_env, n_eval_episodes=episodes, render=True)
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
            break
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
