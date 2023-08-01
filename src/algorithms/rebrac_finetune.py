import functools
import os
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import wandb
import uuid
import pyrallis
import random

import jax
import numpy as np
import optax
import tqdm

from functools import partial
from dataclasses import dataclass, asdict
from flax.core import FrozenDict
from typing import Dict, Tuple, Any, Callable
from tqdm.auto import trange

from flax.training.train_state import TrainState

from src.networks import EnsembleCritic, DetActor
from src.utils.buffer import ReplayBuffer, OnlineReplayBuffer, make_env_and_dataset, concat_batches
from src.utils.common import Metrics, make_env, evaluate, wrap_env, is_goal_reached


ENVS_WITH_GOAL = ("antmaze", "pen", "door", "hammer", "relocate")


@dataclass
class Config:
    # wandb params
    project: str = "ReBRAC"
    group: str = "rebrac-finetune"
    name: str = "rebrac-finetune"
    # model params
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    hidden_dim: int = 256
    actor_n_hiddens: int = 3
    critic_n_hiddens: int = 3
    replay_buffer_size: int = 2_000_000
    mixing_ratio: float = 0.5
    gamma: float = 0.99
    tau: float = 5e-3
    actor_bc_coef: float = 1.0
    critic_bc_coef: float = 1.0
    bc_coef_mul: float = 1.0  # Temporary
    actor_ln: bool = False
    critic_ln: bool = True
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    expl_noise: float = 0.03
    policy_freq: int = 2
    normalize_q: bool = True
    min_decay_coef: float = 0.0
    use_calibration: bool = False
    use_tanh: bool = False
    reset_opts: bool = False
    # training params
    dataset_name: str = "halfcheetah-medium-v2"
    batch_size: int = 256
    num_epochs: int = 1000
    num_updates_on_epoch: int = 1000
    num_offline_updates: int = 1_000_000
    num_online_updates: int = 1_000_000
    num_warmup_steps: int = 0
    normalize_reward: bool = False
    normalize_states: bool = False
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 5000
    # general params
    train_seed: int = 10
    eval_seed: int = 42

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_name}-{str(uuid.uuid4())[:8]}"


class CriticTrainState(TrainState):
    target_params: FrozenDict


class ActorTrainState(TrainState):
    target_params: FrozenDict


@jax.jit
def update_actor(
        key: jax.random.PRNGKey,
        actor: TrainState,
        critic: TrainState,
        batch: Dict[str, jax.Array],
        beta: float,
        tau: float,
        normalize_q: bool,
        metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, TrainState, jax.Array, Metrics]:
    key, random_action_key = jax.random.split(key, 2)

    def actor_loss_fn(params):
        actions = actor.apply_fn(params, batch["states"])

        bc_penalty = ((actions - batch["actions"]) ** 2).sum(-1)
        q_values = critic.apply_fn(critic.params, batch["states"], actions).min(0)
        # lmbda = 1
        # # if normalize_q:
        lmbda = jax.lax.stop_gradient(1 / jax.numpy.abs(q_values).mean())

        loss = (beta * bc_penalty - lmbda * q_values).mean()

        # logging stuff
        random_actions = jax.random.uniform(random_action_key, shape=batch["actions"].shape, minval=-1.0, maxval=1.0)
        new_metrics = metrics.update({
            "actor_loss": loss,
            "bc_mse_policy": bc_penalty.mean(),
            "bc_mse_random": ((random_actions - batch["actions"]) ** 2).sum(-1).mean(),
            "action_mse": ((actions - batch["actions"]) ** 2).mean()
        })
        return loss, new_metrics

    grads, new_metrics = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    new_actor = new_actor.replace(
        target_params=optax.incremental_update(actor.params, actor.target_params, tau)
    )
    new_critic = critic.replace(
        target_params=optax.incremental_update(critic.params, critic.target_params, tau)
    )

    return key, new_actor, new_critic, new_metrics


def update_critic(
        key: jax.random.PRNGKey,
        actor: TrainState,
        critic: CriticTrainState,
        batch: Dict[str, jax.Array],
        gamma: float,
        beta: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        use_calibration: bool,
        metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, TrainState, Metrics]:
    key, actions_key = jax.random.split(key)

    next_actions = actor.apply_fn(actor.target_params, batch["next_states"])
    noise = jax.numpy.clip(
        (jax.random.normal(actions_key, next_actions.shape) * policy_noise),
        -noise_clip,
        noise_clip,
    )
    next_actions = jax.numpy.clip(next_actions + noise, -1, 1)

    bc_penalty = ((next_actions - batch["next_actions"]) ** 2).sum(-1)
    next_q = critic.apply_fn(critic.target_params, batch["next_states"], next_actions).min(0)
    # lower_bounds = jax.numpy.repeat(batch['mc_returns'].reshape(-1, 1), next_q.shape[1], axis=1)
    next_q = next_q - beta * bc_penalty
    target_q = jax.lax.cond(
        use_calibration,
        lambda: jax.numpy.maximum(batch["rewards"] + (1 - batch["dones"]) * gamma * next_q, batch['mc_returns']),
        lambda: batch["rewards"] + (1 - batch["dones"]) * gamma * next_q
    )

    def critic_loss_fn(critic_params):
        # [N, batch_size] - [1, batch_size]
        q = critic.apply_fn(critic_params, batch["states"], batch["actions"])
        q_min = q.min(0).mean()
        loss = ((q - target_q[None, ...]) ** 2).mean(1).sum(0)
        return loss, q_min

    (loss, q_min), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)
    new_metrics = metrics.update({
        "critic_loss": loss,
        "q_min": q_min,
    })
    return key, new_critic, new_metrics


@jax.jit
def update_td3(
        key: jax.random.PRNGKey,
        actor: TrainState,
        critic: CriticTrainState,
        batch: Dict[str, Any],
        metrics: Metrics,
        gamma: float,
        actor_bc_coef: float,
        critic_bc_coef: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        normalize_q: bool,
        use_calibration: bool,
):
    key, new_critic, new_metrics = update_critic(
        key, actor, critic, batch, gamma, critic_bc_coef, tau, policy_noise, noise_clip, use_calibration, metrics
    )
    key, new_actor, new_critic, new_metrics = update_actor(key, actor,
                                                           new_critic, batch, actor_bc_coef, tau, normalize_q,
                                                           new_metrics)
    return key, new_actor, new_critic, new_metrics


@jax.jit
def update_td3_no_targets(
        key: jax.random.PRNGKey,
        actor: TrainState,
        critic: CriticTrainState,
        batch: Dict[str, Any],
        gamma: float,
        metrics: Metrics,
        actor_bc_coef: float,
        critic_bc_coef: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        use_calibration: bool,
):
    key, new_critic, new_metrics = update_critic(
        key, actor, critic, batch, gamma, critic_bc_coef, tau, policy_noise, noise_clip, use_calibration, metrics
    )
    return key, actor, new_critic, new_metrics


def action_fn(actor: TrainState) -> Callable:
    @jax.jit
    def _action_fn(obs: jax.Array) -> jax.Array:
        action = actor.apply_fn(actor.params, obs)
        return action

    return _action_fn


@pyrallis.wrap()
def main(config: Config):
    # config.actor_bc_coef = config.critic_bc_coef * config.bc_coef_mul
    dict_config = asdict(config)
    dict_config["mlc_job_name"] = os.environ.get("PLATFORM_JOB_NAME")
    is_env_with_goal = config.dataset_name.startswith(ENVS_WITH_GOAL)
    np.random.seed(config.train_seed)
    random.seed(config.train_seed)

    wandb.init(
        config=dict_config,
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
    )
    buffer = ReplayBuffer()
    buffer.create_from_d4rl(config.dataset_name, config.normalize_reward, config.normalize_states)

    key = jax.random.PRNGKey(seed=config.train_seed)
    key, actor_key, critic_key = jax.random.split(key, 3)

    init_state = buffer.data["states"][0][None, ...]
    init_action = buffer.data["actions"][0][None, ...]

    actor_module = DetActor(action_dim=init_action.shape[-1], hidden_dim=config.hidden_dim, layernorm=config.actor_ln)
    actor = ActorTrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.init(actor_key, init_state),
        target_params=actor_module.init(actor_key, init_state),
        tx=optax.adam(learning_rate=config.actor_learning_rate),
    )

    critic_module = EnsembleCritic(hidden_dim=config.hidden_dim, num_critics=2, layernorm=config.critic_ln, use_tanh=config.use_tanh)
    critic = CriticTrainState.create(
        apply_fn=critic_module.apply,
        params=critic_module.init(critic_key, init_state, init_action),
        target_params=critic_module.init(critic_key, init_state, init_action),
        tx=optax.adam(learning_rate=config.critic_learning_rate),
    )

    update_td3_partial = partial(
        update_td3, gamma=config.gamma,
        actor_bc_coef=config.actor_bc_coef, critic_bc_coef=config.critic_bc_coef, tau=config.tau,
        policy_noise=config.policy_noise,
        noise_clip=config.noise_clip,
        normalize_q=config.normalize_q,
        use_calibration=config.use_calibration,
    )

    update_td3_no_targets_partial = partial(
        update_td3_no_targets, gamma=config.gamma,
        actor_bc_coef=config.actor_bc_coef, critic_bc_coef=config.critic_bc_coef, tau=config.tau,
        policy_noise=config.policy_noise,
        noise_clip=config.noise_clip,
        use_calibration=config.use_calibration,
    )

    def td3_loop_update_step(i, carry):
        key, batch_key = jax.random.split(carry["key"])
        batch = carry["buffer"].sample_batch(batch_key, batch_size=config.batch_size)

        full_update = partial(update_td3_partial,
                              key=key,
                              actor=carry["actor"],
                              critic=carry["critic"],
                              batch=batch,
                              metrics=carry["metrics"])

        update = partial(update_td3_no_targets_partial,
                         key=key,
                         actor=carry["actor"],
                         critic=carry["critic"],
                         batch=batch,
                         metrics=carry["metrics"])

        key, new_actor, new_critic, new_metrics = jax.lax.cond(carry["delayed_updates"][i], full_update, update)

        # key, new_actor, new_critic, new_metrics = update_func(
        #     key=key,
        #     actor=carry["actor"],
        #     critic=carry["critic"],
        #     batch=batch,
        #     metrics=carry["metrics"]
        # )
        carry.update(
            key=key, actor=new_actor, critic=new_critic, metrics=new_metrics
        )
        return carry

    # metrics
    bc_metrics_to_log = [
        "critic_loss", "q_min", "actor_loss", "batch_entropy",
        "bc_mse_policy", "bc_mse_random", "action_mse"
    ]
    # shared carry for update loops
    carry = {
        "key": key,
        "actor": actor,
        "critic": critic,
        "buffer": buffer,
        "delayed_updates": jax.numpy.equal(
            jax.numpy.arange(config.num_offline_updates + config.num_online_updates) % config.policy_freq, 0
        ).astype(int)
    }

    # Online + offline tuning
    env, dataset = make_env_and_dataset(config.dataset_name, config.train_seed, False, discount=config.gamma)
    eval_env, _ = make_env_and_dataset(config.dataset_name, config.eval_seed, False, discount=config.gamma)

    max_steps = env._max_episode_steps

    action_dim = env.action_space.shape[0]
    replay_buffer = OnlineReplayBuffer(env.observation_space, action_dim,
                                       config.replay_buffer_size)
    replay_buffer.initialize_with_dataset(dataset, None)
    online_buffer = OnlineReplayBuffer(env.observation_space, action_dim, config.replay_buffer_size)

    online_batch_size = 0
    offline_batch_size = config.batch_size

    observation, done = env.reset(), False
    episode_step = 0
    goal_achieved = False

    @jax.jit
    def actor_action_fn(params, obs):
        return actor.apply_fn(params, obs)

    eval_successes = []
    train_successes = []
    print("Offline training")
    for i in tqdm.tqdm(range(config.num_online_updates + config.num_offline_updates), smoothing=0.1):
        carry["metrics"] = Metrics.create(bc_metrics_to_log)
        # actor_action_fn = action_fn(actor=update_carry["actor"])
        if i == config.num_offline_updates:
            print("Online training")

            online_batch_size = int(config.mixing_ratio * config.batch_size)
            offline_batch_size = config.batch_size - online_batch_size
            # replay_buffer = OnlineReplayBuffer(env.observation_space, action_dim,
            #                                    config.replay_buffer_size)
            # Reset optimizers similar to SPOT
            if config.reset_opts:
                actor = actor.replace(
                    opt_state=optax.adam(learning_rate=config.actor_learning_rate).init(actor.params)
                )
                critic = critic.replace(
                    opt_state=optax.adam(learning_rate=config.critic_learning_rate).init(critic.params)
                )

        update_td3_partial = partial(
                update_td3, gamma=config.gamma,
                tau=config.tau,
                policy_noise=config.policy_noise,
                noise_clip=config.noise_clip,
                normalize_q=config.normalize_q,
                use_calibration=config.use_calibration,
            )

        update_td3_no_targets_partial = partial(
                update_td3_no_targets, gamma=config.gamma,
                tau=config.tau,
                policy_noise=config.policy_noise,
                noise_clip=config.noise_clip,
                use_calibration=config.use_calibration,
            )
        online_log = {}

        if i >= config.num_offline_updates:
            episode_step += 1
            action = np.asarray(actor_action_fn(carry["actor"].params, observation))
            # print("A1", action)
            # print(action.shape)
            action = np.array(
                [
                    (
                            action
                            + np.random.normal(0, 1 * config.expl_noise, size=action_dim)
                    ).clip(-1, 1)
                ]
            )[0]

            # print(action.shape)
            next_observation, reward, done, info = env.step(action)
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, info)
            next_action = np.asarray(actor_action_fn(carry["actor"].params, next_observation))[0]
            next_action = np.array(
                [
                    (
                            next_action
                            + np.random.normal(0, 1 * config.expl_noise, size=action_dim)
                    ).clip(-1, 1)
                ]
            )[0]
            # print("A2", action)
            # print("NA2", next_action)
            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0
            real_done = False
            if done and episode_step < max_steps:
                real_done = True

            online_buffer.insert(observation, action, reward, mask,
                                 float(real_done), next_observation, next_action, 0)
            observation = next_observation
            if done:
                train_successes.append(goal_achieved)
                observation, done = env.reset(), False
                episode_step = 0
                goal_achieved = False

            #     for k, v in info['episode'].items():
            #         summary_writer.add_scalar(f'training/{k}', v,
            #                                   info['total']['timesteps'])
        if config.num_offline_updates <= i < config.num_offline_updates + config.num_warmup_steps:
            continue

        offline_batch = replay_buffer.sample(offline_batch_size)
        online_batch = online_buffer.sample(online_batch_size)
        batch = concat_batches(offline_batch, online_batch)

        # print(carry["actor"])
        if 'antmaze' in config.dataset_name and config.normalize_reward:
            batch["rewards"] *= 100

        # key, batch_key = jax.random.split(carry["key"])
        ### Update step
        actor_bc_coef = config.actor_bc_coef
        critic_bc_coef = config.critic_bc_coef
        if i >= config.num_offline_updates:
            decay_coef = max(config.min_decay_coef, (config.num_online_updates + config.num_offline_updates - i + config.num_warmup_steps) / config.num_online_updates)
            # decay_coef = (1 - (1 - decay_coef) ** 4)  # np.exp(np.log(0.5) * (1 -decay_coef))
            actor_bc_coef *= decay_coef
            critic_bc_coef *= 0 #decay_coef
        if i % config.policy_freq == 0:
            update_fn = partial(update_td3_partial,
                                actor_bc_coef=actor_bc_coef,
                                critic_bc_coef=critic_bc_coef,
                                key=key,
                                actor=carry["actor"],
                                critic=carry["critic"],
                                batch=batch,
                                metrics=carry["metrics"])
        else:
            update_fn = partial(update_td3_no_targets_partial,
                                actor_bc_coef=actor_bc_coef,
                                critic_bc_coef=critic_bc_coef,
                                key=key,
                                actor=carry["actor"],
                                critic=carry["critic"],
                                batch=batch,
                                metrics=carry["metrics"])
        key, new_actor, new_critic, new_metrics = update_fn()
        carry.update(
            key=key, actor=new_actor, critic=new_critic, metrics=new_metrics
        )

        if i % 1000 == 0:
            mean_metrics = carry["metrics"].compute()
            common = {f"TD3/{k}": v for k, v in mean_metrics.items()}
            common["actor_bc_coef"] = actor_bc_coef
            common["critic_bc_coef"] = critic_bc_coef
            if i < config.num_offline_updates:
                wandb.log({"offline_iter": i, **common})
            else:
                wandb.log({"online_iter": i - config.num_offline_updates, **common})
        if i % config.eval_every == 0 or i == config.num_offline_updates + config.num_online_updates - 1 or i == config.num_offline_updates - 1:
            # actor_action_fn = action_fn(actor=update_carry["actor"])

            eval_returns, success_rate = evaluate(eval_env, carry["actor"].params, actor_action_fn, config.eval_episodes,
                                    seed=config.eval_seed)
            normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
            eval_successes.append(success_rate)
            if is_env_with_goal:
                online_log["train/regret"] = np.mean(1 - np.array(train_successes))
            offline_log = {
                "eval/return_mean": np.mean(eval_returns),
                "eval/return_std": np.std(eval_returns),
                "eval/normalized_score_mean": np.mean(normalized_score),
                "eval/normalized_score_std": np.std(normalized_score),
                "eval/success_rate": success_rate
            }
            offline_log.update(online_log)
            wandb.log(offline_log)


if __name__ == "__main__":
    main()