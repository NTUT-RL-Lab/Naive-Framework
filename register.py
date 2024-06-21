from gymnasium.envs.registration import register

register(
    id="Facade/test",
    entry_point="Naive-Framework:Facade",
    max_episode_steps=300,
)
