import os

os.environ['MUJOCO_GL'] = 'egl'

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Dict, List, Optional, Callable
import copy
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
import uuid
import warnings

import d4rl
from dm_env import specs
import gym
import numpy as np
import pyrallis
from tqdm.auto import trange
import wandb

from src.utils.buffer import EfficientReplayBuffer, load_offline_dataset_into_buffer
from src.utils.vd4rl_utils import make

TensorBatch = List[torch.Tensor]

warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    task_name: str = "walker_walk"
    dataset_name: str = "expert"
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(1e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e5)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # TD3
    buffer_size: int = 1_000_000  # Replay buffer size
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    encoder_learning_rate: float = 3e-4
    hidden_dim: int = 256
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    expl_noise: float = 0.1  # Std of Gaussian exploration noise
    tau: float = 0.005  # Target network update rate
    policy_noise: float = 0.2  # Noise added to target actor during critic update
    noise_clip: float = 0.5  # Range to clip target actor noise
    policy_freq: int = 2  # Frequency of delayed actor updates
    # ReBRAC
    actor_bc_coef: float = 0.4
    critic_bc_coef: float = 0.0
    critic_ln: bool = True
    normalize: bool = False  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    # Wandb logging
    project: str = "ReBRAC-vis"
    group: str = "ReBRAC-VD4RL"
    name: str = "ReBRAC"

    def __post_init__(self):
        self.name = f"{self.name}-{self.task_name}-{self.dataset_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor_fn: Callable, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset().observation[None, ...], False
        episode_reward = 0.0
        while not done:
            action = actor_fn(state, device)
            env_out = env.step(action)
            state, reward, done = env_out.observation[None, ...], env_out.reward, env_out.last()
            episode_reward += reward
        episode_rewards.append(episode_reward)
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(
            self, state_dim: int, action_dim: int, max_action: float,
            hidden_dim: int = 256,
    ):
        super(Actor, self).__init__()

        self.trunk = nn.Sequential(nn.Linear(state_dim, 50),
                                   nn.LayerNorm(50), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(50, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_dim))

        self.apply(weight_init)

        # self.net = nn.Sequential(
        #     nn.Linear(state_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, action_dim),
        #     nn.Tanh(),
        # )
        # self.apply(weight_init)
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h = self.trunk(state)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        return self.max_action * mu
        # return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        # state = torch.tensor(state, device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class Critic(nn.Module):
    def __init__(
            self, state_dim: int, action_dim: int,
            hidden_dim: int = 256, critic_ln: bool = True,
    ):
        super(Critic, self).__init__()

        self.trunk = nn.Sequential(nn.Linear(state_dim, 50),
                                   nn.LayerNorm(50), nn.Tanh())

        self.net = nn.Sequential(
            nn.Linear(50 + action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim) if critic_ln else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim) if critic_ln else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim) if critic_ln else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(weight_init)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([self.trunk(state), action], 1)
        return self.net(sa)


class ReBRAC:  # noqa
    def __init__(
        self,
        max_action: float,
        aug: nn.Module,
        encoder: nn.Module,
        encoder_optimizer: torch.optim.Optimizer,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        actor_bc_coef: float = 0.4,
        critic_bc_coef: float = 0.0,
        device: str = "cpu",
    ):
        self.aug = aug
        self.encoder = encoder
        self.encoder_optimizer = encoder_optimizer
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.actor_optimizer = actor_optimizer
        self.critic_1 = critic_1
        self.critic_1_target = copy.deepcopy(critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2 = critic_2
        self.critic_2_target = copy.deepcopy(critic_2)
        self.critic_2_optimizer = critic_2_optimizer

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.actor_bc_coef = actor_bc_coef
        self.critic_bc_coef = critic_bc_coef

        self.total_it = 0
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        # state, action, reward, next_state, done = batch
        state, action, reward, discount, next_state, next_action_d = batch

        state = self.encoder(self.aug(state))
        with torch.no_grad():
            next_state = self.encoder(self.aug(next_state))

        with torch.no_grad():
            # Select action according to actor and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)

            bc_penalty = ((next_action - next_action_d) ** 2).sum(-1).unsqueeze(dim=-1)

            target_q = torch.min(target_q1, target_q2) - self.critic_bc_coef * bc_penalty
            target_q = reward + discount * target_q

        # Get current Q estimates
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        log_dict["critic_loss"] = critic_loss.item()
        # Optimize the critic
        self.encoder_optimizer.zero_grad()
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        self.encoder_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            pi = self.actor(state.detach())
            q = self.critic_1(state.detach(), pi)
            lmbda = 1 / (q.abs().mean().detach())

            actor_loss = -lmbda * q.mean() + self.actor_bc_coef * F.mse_loss(pi, action)
            log_dict["actor_loss"] = actor_loss.item()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "critic_1": self.critic_1.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.critic_1.load_state_dict(state_dict["critic_1"])
        self.critic_1_optimizer.load_state_dict(state_dict["critic_1_optimizer"])
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2.load_state_dict(state_dict["critic_2"])
        self.critic_2_optimizer.load_state_dict(state_dict["critic_2_optimizer"])
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)

        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    with open(os.path.join("./", "config.yaml"), "w") as f:
        pyrallis.dump(config, f)
    return

    env = make(config.task_name, 3, 2, config.eval_seed)

    data_specs = (env.observation_spec(),
                  env.action_spec(),
                  specs.Array((1,), np.float32, 'reward'),
                  specs.Array((1,), np.float32, 'discount'))
    buffer_size = config.buffer_size
    replay_buffer = EfficientReplayBuffer(
        buffer_size, config.batch_size, 3, config.discount, 3,
        data_specs=data_specs, sarsa=True
    )

    load_offline_dataset_into_buffer(
        Path(f"./vd4rl/main/{config.task_name}/{config.dataset_name}/84px"),
        replay_buffer,
        3,
        buffer_size,
    )

    max_action = 1.0

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed)

    action_dim = env.action_spec().shape[0]

    aug = RandomShiftsAug()

    encoder = Encoder((9, 84, 84)).to(config.device)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=config.encoder_learning_rate)

    actor = Actor(encoder.repr_dim, action_dim, max_action, config.hidden_dim).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)

    critic_1 = Critic(encoder.repr_dim, action_dim, config.hidden_dim, config.critic_ln).to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.critic_learning_rate)
    critic_2 = Critic(encoder.repr_dim, action_dim, config.hidden_dim, config.critic_ln).to(config.device)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.critic_learning_rate)

    kwargs = {
        "max_action": max_action,
        "aug": aug,
        "encoder": encoder,
        "encoder_optimizer": encoder_optimizer,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "critic_1": critic_1,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2": critic_2,
        "critic_2_optimizer": critic_2_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # TD3
        "policy_noise": config.policy_noise * max_action,
        "noise_clip": config.noise_clip * max_action,
        "policy_freq": config.policy_freq,
        # ReBRAC
        "actor_bc_coef": config.actor_bc_coef,
        "critic_bc_coef": config.critic_bc_coef,
    }

    print("---------------------------------------")
    print(f"Training TD3 + BC, Env: {config.task_name}, {config.dataset_name} Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ReBRAC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    def to_torch(xs, device):
        return tuple(torch.FloatTensor(x).to(device) for x in xs)

    evaluations = []
    for t in trange(config.max_timesteps, desc="ReBRAC steps"):
        batch = list(to_torch(next(replay_buffer), config.device))
        # batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")

            def act_fn(state, device):
                state = torch.tensor(state, device=device, dtype=torch.float32)
                return actor.act(encoder(state), device)

            eval_scores = eval_actor(
                env,
                act_fn,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_scores = eval_scores
            normalized_eval_scores = eval_scores / 10
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_scores.mean():.3f} , VD4RL score: {normalized_eval_scores.mean():.3f}"
            )
            print("---------------------------------------")
            wandb.log(
                {
                    "eval/normalized_score": normalized_eval_scores.mean(),
                    "eval/normalized_score_std": normalized_eval_scores.std(),
                },
                step=trainer.total_it,
            )


if __name__ == "__main__":
    train()
