import abc

import equinox as eqx
from jaxtyping import Array, Float


class AbstractDQNPolicy(eqx.Module):
    @abc.abstractmethod
    def __call__(
        self, state: Float[Array, " n_obs_dim"]
    ) -> Float[Array, " n_actions"]: ...
