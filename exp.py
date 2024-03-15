import gymnasium as gym
from director import Director
from gymnasium import logger
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
from gymnasium.wrappers.resize_observation import ResizeObservation
from coef import Coef
from facade import Facade
from stable_baselines3 import *
from guise import Guise


def main():
    """Main function to run the experiment
    """
    coef = Coef(
        n_timestep=10000,
        c_lr=0.0001,
        cap=1000,
        env_weights=[0.5, 0.5],
        n_envs=2,
        env_ids=["LunarLander-v2", "CartPole-v1"],
        # env_ids = ["CartPole-v1" ],
        # n_envs=1,
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
    envs = birth_envs(coef.env_ids)
    facade = Facade(envs, director=director)
    model = PPO(coef.policy, facade)
    director.set_model(model)

    director.learn()
    print("Learning done")


def birth_envs(env_ids) -> list[gym.Env]:
    """Births the environments
    """
    max_w, max_h = 0, 0
    envs = [gym.make(env_id, render_mode="rgb_array") for env_id in env_ids]
    for i in range(envs.__len__()):
        envs[i] = Guise(envs[i])
        obs = envs[i].observation_space
        max_w = max(max_w, obs.shape[0])
        max_h = max(max_h, obs.shape[1])
    for i in range(envs.__len__()):
        # magic(observation normalize) happen here 🪄🕊️
        envs[i].rescale_observation((max_w, max_h))
        logger.info(
            f"envs[{i}].observation_space.shape: {envs[i].observation_space.shape}")
    return envs


if __name__ == '__main__':
    main()
