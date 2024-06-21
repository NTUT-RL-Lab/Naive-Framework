from logging import config
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


def main():
    """Main function to run the experiment
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    coef = Coef("config/" + args.config)
    logger.set_level(logger.INFO)
    logger.info("👻")
    director = Director(coef)
    envs = director.birth_envs()
    facade = Facade(envs, director=director)
    model = coef.algorithm(policy=coef.policy, env=facade,
                           tensorboard_log="logs/", seed=coef.seed, learning_rate=coef.c_lr)
    director.set_model(model)
    director.learn()
    print("Learning done")
    director.save("models/" + args.model)


if __name__ == '__main__':
    main()
