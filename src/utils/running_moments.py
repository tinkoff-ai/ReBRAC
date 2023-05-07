import chex
import jax
import jax.numpy as jnp

from typing import Dict


@chex.dataclass(frozen=True)
class RunningMeanStd:
    state: Dict[str, jax.Array]

    @staticmethod
    def create(eps: float = 1e-4) -> "RunningMeanStd":
        init_state = {
            "mean": jnp.array([0.0]),
            "var": jnp.array([0.0]),
            "count": jnp.array([eps])
        }
        return RunningMeanStd(state=init_state)

    def update(self, batch: jax.Array) -> "RunningMeanStd":
        batch_mean, batch_var, batch_count = batch.mean(), batch.var(), batch.shape[0]
        if batch_count == 1:
            batch_var = jnp.zeros_like(batch_mean)

        new_mean, new_var, new_count = self._update_mean_var_count_from_moments(
            self.state["mean"], self.state["var"], self.state["count"], batch_mean, batch_var, batch_count
        )
        return self.replace(state={"mean": new_mean, "var": new_var, "count": new_count})

    @staticmethod
    def _update_mean_var_count_from_moments(
        mean, var, count, batch_mean, batch_var, batch_count
    ):
        """Updates the mean, var and count using the previous mean, var, count and batch values."""
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count

    @property
    def std(self):
        return jnp.sqrt(self.state["var"])

    @property
    def mean(self):
        return self.state["mean"]
