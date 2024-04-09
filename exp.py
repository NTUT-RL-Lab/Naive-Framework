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
    envs = birth_envs(coef.env_ids, coef.act_mapping)
    facade = Facade(envs, director=director)
    model = PPO(coef.policy, facade, tensorboard_log="logs/")
    director.set_model(model)

    director.learn()
    print("Learning done")
    director.save("models/" + args.model)


def birth_envs(env_ids, action_mappings: dict[int, str]) -> list[Guise]:
    """Births the environments
    """
    max_w, max_h = 0, 0
    envs = [gym.make(env_id, render_mode="rgb_array") for env_id in env_ids]
    disguises: list[Guise] = []
    for i in range(envs.__len__()):
        disguises.append(Guise(envs[i]))
    #     obs = disguises[i].observation_space
    #     max_w = max(max_w, obs.shape[0])
    #     max_h = max(max_h, obs.shape[1])
    all_actions = {}
    cid = 0
    # hardcode shape for now
    # max_w, max_h = 84, 84
    for i in range(disguises.__len__()):
        #     # magic(observation normalize) happen here 🪄🕊️
        #     # disguises[i].rescale_observation((max_w, max_h))
        for action in action_mappings[i].values():
            if action not in all_actions:
                all_actions[action] = cid
                cid += 1
        logger.info(
            f"disguises[{i}].observation_space.shape: {disguises[i].observation_space.shape}")
    logger.info(f"all_actions: {all_actions}")
    for i in range(disguises.__len__()):
        mapping = {}
        for key, value in all_actions.items():
            for k, v in action_mappings[i].items():
                if v == key:
                    mapping[value] = k
                    break
            else:
                mapping[value] = all_actions["NOOP"]
        logger.info(f"mapping for disguises[{i}]: {mapping}")
        disguises[i].init_action_mapping(
            mapping, origin_space=disguises[i].action_space.n)
    return disguises


if __name__ == '__main__':
    main()
