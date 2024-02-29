import gymnasium
from gymnasium import Env, spaces, logger
from typing import TYPE_CHECKING, List, TypeVar, Any, Generic, SupportsFloat
from copy import deepcopy
import numpy as np

from gymnasium.utils import RecordConstructorArgs, seeding


if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec, WrapperSpec

WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")
class Wrapper(
    Env[WrapperObsType, WrapperActType],
    Generic[WrapperObsType, WrapperActType, ObsType, ActType],
):
    """Wraps a :class:`gymnasium.Env` to allow a modular transformation of the :meth:`step` and :meth:`reset` methods.

    This class is the base class of all wrappers to change the behavior of the underlying environment.
    Wrappers that inherit from this class can modify the :attr:`action_space`, :attr:`observation_space`,
    :attr:`reward_range` and :attr:`metadata` attributes, without changing the underlying environment's attributes.
    Moreover, the behavior of the :meth:`step` and :meth:`reset` methods can be changed by these wrappers.

    Some attributes (:attr:`spec`, :attr:`render_mode`, :attr:`np_random`) will point back to the wrapper's environment
    (i.e. to the corresponding attributes of :attr:`env`).

    Note:
        If you inherit from :class:`Wrapper`, don't forget to call ``super().__init__(env)``
    """

    def __init__(self, envs: List(Env[ObsType, ActType])):
        """Wraps an environment to allow a modular transformation of the :meth:`step` and :meth:`reset` methods.

        Args:
            env: The environment to wrap
        """
        self.selector = 0
        self.envs = envs

        self._action_space: spaces.Space[WrapperActType] | None = None
        self._observation_space: spaces.Space[WrapperObsType] | None = None
        self._reward_range: tuple[SupportsFloat, SupportsFloat] | None = None
        self._metadata: dict[str, Any] | None = None

        self._cached_spec: EnvSpec | None = None

    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore.

        Args:
            name: The variable name

        Returns:
            The value of the variable in the wrapper stack

        Warnings:
            This feature is deprecated and removed in v1.0 and replaced with `env.get_attr(name})`
        """
        if name == "_np_random":
            raise AttributeError(
                "Can't access `_np_random` of a wrapper, use `self.unwrapped._np_random` or `self.np_random`."
            )
        elif name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        logger.warn(
            f"env.{name} to get variables from other wrappers is deprecated and will be removed in v1.0, "
            f"to get this variable you can do `env.unwrapped.{name}` for environment variables or `env.get_attr('{name}')` that will search the reminding wrappers."
        )
        return getattr(self.env, name)

    def get_wrapper_attr(self, name: str) -> Any:
        """Gets an attribute from the wrapper and lower environments if `name` doesn't exist in this object.

        Args:
            name: The variable name to get

        Returns:
            The variable with name in wrapper or lower environments
        """
        if hasattr(self, name):
            return getattr(self, name)
        else:
            try:
                return self.env.get_wrapper_attr(name)
            except AttributeError as e:
                raise AttributeError(
                    f"wrapper {self.class_name()} has no attribute {name!r}"
                ) from e

    @property
    def spec(self) -> EnvSpec | None:
        """Returns the :attr:`Env` :attr:`spec` attribute with the `WrapperSpec` if the wrapper inherits from `EzPickle`."""
        if self._cached_spec is not None:
            return self._cached_spec

        env_spec = self.env.spec
        if env_spec is not None:
            # See if the wrapper inherits from `RecordConstructorArgs` then add the kwargs otherwise use `None` for the wrapper kwargs. This will raise an error in `make`
            if isinstance(self, RecordConstructorArgs):
                kwargs = getattr(self, "_saved_kwargs")
                if "env" in kwargs:
                    kwargs = deepcopy(kwargs)
                    kwargs.pop("env")
            else:
                kwargs = None

            from gymnasium.envs.registration import WrapperSpec

            wrapper_spec = WrapperSpec(
                name=self.class_name(),
                entry_point=f"{self.__module__}:{type(self).__name__}",
                kwargs=kwargs,
            )

            # to avoid reference issues we deepcopy the prior environments spec and add the new information
            env_spec = deepcopy(env_spec)
            env_spec.additional_wrappers += (wrapper_spec,)

        self._cached_spec = env_spec
        return env_spec

    @classmethod
    def wrapper_spec(cls, **kwargs: Any) -> WrapperSpec:
        """Generates a `WrapperSpec` for the wrappers."""
        from gymnasium.envs.registration import WrapperSpec

        return WrapperSpec(
            name=cls.class_name(),
            entry_point=f"{cls.__module__}:{cls.__name__}",
            kwargs=kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def action_space(
        self,
    ) -> spaces.Space[ActType] | spaces.Space[WrapperActType]:
        """Return the :attr:`Env` :attr:`action_space` unless overwritten then the wrapper :attr:`action_space` is used."""
        if self._action_space is None:
            return self.env.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space: spaces.Space[WrapperActType]):
        self._action_space = space

    @property
    def observation_space(
        self,
    ) -> spaces.Space[ObsType] | spaces.Space[WrapperObsType]:
        """Return the :attr:`Env` :attr:`observation_space` unless overwritten then the wrapper :attr:`observation_space` is used."""
        if self._observation_space is None:
            return self.env.observation_space
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space: spaces.Space[WrapperObsType]):
        self._observation_space = space

    @property
    def reward_range(self) -> tuple[SupportsFloat, SupportsFloat]:
        """Return the :attr:`Env` :attr:`reward_range` unless overwritten then the wrapper :attr:`reward_range` is used."""
        if self._reward_range is None:
            return self.env.reward_range
        logger.warn("The `reward_range` is deprecated and will be removed in v1.0")
        return self._reward_range

    @reward_range.setter
    def reward_range(self, value: tuple[SupportsFloat, SupportsFloat]):
        self._reward_range = value

    @property
    def metadata(self) -> dict[str, Any]:
        """Returns the :attr:`Env` :attr:`metadata`."""
        if self._metadata is None:
            return self.env.metadata
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict[str, Any]):
        self._metadata = value

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the :attr:`Env` :attr:`np_random` attribute."""
        return self.env.np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self.env.np_random = value

    @property
    def _np_random(self):
        """This code will never be run due to __getattr__ being called prior this.

        It seems that @property overwrites the variable (`_np_random`) meaning that __getattr__ gets called with the missing variable.
        """
        raise AttributeError(
            "Can't access `_np_random` of a wrapper, use `.unwrapped._np_random` or `.np_random`."
        )

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Uses the :meth:`step` of the :attr:`env` that can be overwritten to change the returned data."""
        return self.env.step(action)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Uses the :meth:`reset` of the :attr:`env` that can be overwritten to change the returned data."""
        return self.env.reset(seed=seed, options=options)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Uses the :meth:`render` of the :attr:`env` that can be overwritten to change the returned data."""
        return self.env.render()

    def close(self):
        """Closes the wrapper and :attr:`env`."""
        return self.env.close()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    @property
    def unwrapped(self) -> Env[ObsType, ActType]:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

