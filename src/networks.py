import jax
import distrax
import jax.numpy as jnp
import flax.linen as nn

from src.nn import pytorch_init, uniform_init


class TanhNormal(distrax.Transformed):
    def __init__(self, loc, scale):
        normal_dist = distrax.Normal(loc, scale)
        tanh_bijector = distrax.Tanh()
        super().__init__(distribution=normal_dist, bijector=tanh_bijector)

    def mean(self):
        return self.bijector.forward(self.distribution.mean())


def identity(x):
    return x


class DetActor(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    layernorm: bool = True
    groupnorm: bool = False
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state):
        s_d, h_d = state.shape[-1], self.hidden_dim
        # Initialization as in the EDAC paper
        layers = [
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(s_d), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
            nn.GroupNorm() if self.groupnorm else identity,
            nn.LayerNorm(use_bias=False, use_scale=False) if self.featurenorm else identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=nn.initializers.constant(0.1)),
                nn.relu,
                nn.LayerNorm() if self.layernorm else identity,
                nn.GroupNorm() if self.groupnorm else identity,
                nn.LayerNorm(use_bias=False, use_scale=False) if self.featurenorm else identity,
            ]
        layers += [
            nn.Dense(self.action_dim, kernel_init=uniform_init(1e-3), bias_init=uniform_init(1e-3)),
            nn.tanh,
        ]
        net = nn.Sequential(layers)
        actions = net(state)
        return actions


class Critic(nn.Module):
    hidden_dim: int = 256
    layernorm: bool = True
    groupnorm: bool = False
    featurenorm: bool = False
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state, action):
        s_d, a_d, h_d = state.shape[-1], action.shape[-1], self.hidden_dim
        # Initialization as in the EDAC paper
        layers = [
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(s_d + a_d), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
            nn.GroupNorm() if self.groupnorm else identity,
            nn.LayerNorm(use_bias=False, use_scale=False) if self.featurenorm else identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=nn.initializers.constant(0.1)),
                nn.relu,
                nn.LayerNorm() if self.layernorm else identity,
                nn.GroupNorm() if self.groupnorm else identity,
                nn.LayerNorm(use_bias=False, use_scale=False) if self.featurenorm else identity,
            ]
        layers += [
            nn.Dense(1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3))
        ]
        network = nn.Sequential(layers)
        state_action = jnp.hstack([state, action])
        out = network(state_action).squeeze(-1)
        return out


class EnsembleCritic(nn.Module):
    hidden_dim: int = 256
    num_critics: int = 10
    layernorm: bool = True
    groupnorm: bool = False
    featurenorm: bool = False
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state, action):
        ensemble = nn.vmap(
            target=Critic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num_critics,
        )
        q_values = ensemble(self.hidden_dim, self.layernorm, self.groupnorm, self.n_hiddens)(state, action)
        return q_values
