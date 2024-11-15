import gymnasium as gym
from coef import Coef
import argparse
import numpy as np


def eval(id):
    env = gym.make(id, render_mode='rgb_array')
    observation, info = env.reset(seed=420)
    scores = []
    for _ in range(1000):

        score = 0
        for __ in range(1000000):
            action = env.action_space.sample()  # this is where you would insert your policy
            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            if terminated or truncated:
                observation, info = env.reset()
                scores.append(score)
                break

    print(
        f'Env {id} mean score: {sum(scores) / len(scores)}, std: {np.std(scores)}')
    env.close()


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=False)
parser.add_argument("--env_id", type=int,
                    help="environment id to evaluate", required=False)
args = parser.parse_args()
env = None
if args.config:
    coef = Coef(f'config/{args.config}')
    for i in range(coef.n_envs):
        eval(coef.env_ids[i])
