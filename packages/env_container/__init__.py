from gymnasium.envs.registration import register

register(
    id="Facade/container-v0",
    entry_point="env_container.envs:Facade",
)
