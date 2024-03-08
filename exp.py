import gymnasium as gym
from director import Director
from gymnasium import logger
from coef import Coef
from facade import Facade
from stable_baselines3 import *


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
    envs = [gym.make(env_id) for env_id in coef.env_ids]
    facade = Facade(envs, director=director)
    model = PPO(coef.policy, facade)
    director.SetModel(model)

    director.Learn()
    print("Learning done")


if __name__ == '__main__':
    main()
