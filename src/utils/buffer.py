import abc
from collections import deque
import gym
import d4rl
import chex
import h5py
import jax
import jax.numpy as jnp

import numpy as np
from typing import Dict, Tuple

from src.utils.vd4rl_utils import ExtendedTimeStep, step_type_lookup


def calc_return_to_go(is_sparse_reward, rewards, terminals, gamma):
    """
    A config dict for getting the default high/low rewrd values for each envs
    This is used in calc_return_to_go func in sampler.py and replay_buffer.py
    """
    if len(rewards) == 0:
        return []
    reward_neg = 0
    if is_sparse_reward and np.all(np.array(rewards) == reward_neg):
        """
        If the env has sparse reward and the trajectory is all negative rewards,
        we use r / (1-gamma) as return to go.
        For exapmle, if gamma = 0.99 and the rewards = [-1, -1, -1],
        then return_to_go = [-100, -100, -100]
        """
        # assuming failure reward is negative
        # use r / (1-gamma) for negative trajctory
        return_to_go = [float(reward_neg / (1 - gamma))] * len(rewards)
    else:
        return_to_go = [0] * len(rewards)
        prev_return = 0
        for i in range(len(rewards)):
            return_to_go[-i - 1] = rewards[-i - 1] + gamma * prev_return * (1 - terminals[-i - 1])
            prev_return = return_to_go[-i - 1]

    return return_to_go


# source: https://github.com/rail-berkeley/d4rl/blob/d842aa194b416e564e54b0730d9f934e3e32f854/d4rl/__init__.py#L63
# modified to also return next_action (needed for logging and in general useful to have)
def qlearning_dataset(env, dataset_name, normalize_reward=False, dataset=None, terminate_on_end=False, discount=0.99, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, next_actins, rewards,
     and a terminal flag.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            next_actions: An N x dim_action array of next actions.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)
    if normalize_reward:
        dataset['rewards'] = ReplayBuffer.normalize_reward(dataset_name, dataset['rewards'])
    N = dataset['rewards'].shape[0]
    is_sparse = "antmaze" in dataset_name
    obs_ = []
    next_obs_ = []
    action_ = []
    next_action_ = []
    reward_ = []
    done_ = []
    mc_returns_ = []
    print("SIZE", N)
    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    episode_rewards = []
    episode_terminals = []
    for i in range(N - 1):
        if episode_step == 0:
            episode_rewards = []
            episode_terminals = []

        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i + 1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        new_action = dataset['actions'][i + 1].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            mc_returns_ += calc_return_to_go(is_sparse, episode_rewards, episode_terminals, discount)
            # print(len(mc_returns_), len(episode_rewards), end=";")
            continue
        if done_bool or final_timestep:
            episode_step = 0
            # mc_returns_ += calc_return_to_go(is_sparse, episode_rewards, episode_terminals, discount)
            # print(i, len(mc_returns_), len(episode_rewards))

        episode_rewards.append(reward)
        episode_terminals.append(done_bool)

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        next_action_.append(new_action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1
    if episode_step != 0:
        mc_returns_ += calc_return_to_go(is_sparse, episode_rewards, episode_terminals, discount)
    print("SHAPE", np.array(mc_returns_).shape, np.array(reward_).shape, np.array(done_).shape)
    assert np.array(mc_returns_).shape == np.array(reward_).shape
    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'next_actions': np.array(next_action_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        'mc_returns': np.array(mc_returns_),
    }


def compute_mean_std(states: jax.Array, eps: float) -> Tuple[jax.Array, jax.Array]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: jax.Array, mean: jax.Array, std: jax.Array):
    return (states - mean) / std


@chex.dataclass
class ReplayBuffer:
    data: Dict[str, jax.Array] = None
    mean: float = 0
    std: float = 1

    def create_from_d4rl(self, dataset_name: str, normalize_reward: bool = False,
                         normalize: bool = False, discount=0.99):
        d4rl_data = qlearning_dataset(gym.make(dataset_name), dataset_name=dataset_name, normalize_reward=normalize_reward, discount=discount)
        buffer = {
            "states": jnp.asarray(d4rl_data["observations"], dtype=jnp.float32),
            "actions": jnp.asarray(d4rl_data["actions"], dtype=jnp.float32),
            "rewards": jnp.asarray(d4rl_data["rewards"], dtype=jnp.float32),
            "next_states": jnp.asarray(d4rl_data["next_observations"], dtype=jnp.float32),
            "next_actions": jnp.asarray(d4rl_data["next_actions"], dtype=jnp.float32),
            "dones": jnp.asarray(d4rl_data["terminals"], dtype=jnp.float32),
            "mc_returns": jnp.asarray(d4rl_data["mc_returns"], dtype=jnp.float32),
        }
        if normalize:
            self.mean, self.std = compute_mean_std(buffer["states"], eps=1e-3)
            buffer["states"] = normalize_states(
                buffer["states"], self.mean, self.std
            )
            buffer["next_states"] = normalize_states(
                buffer["next_states"], self.mean, self.std
            )
        # if normalize_reward:
        #     buffer["rewards"] = ReplayBuffer.normalize_reward(dataset_name, buffer["rewards"])
        self.data = buffer

    @property
    def size(self):
        # WARN: do not use __len__ here! It will use len of the dataclass, i.e. number of fields.
        return self.data["states"].shape[0]

    def sample_batch(self, key: jax.random.PRNGKey, batch_size: int) -> Dict[str, jax.Array]:
        indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=self.size)
        batch = jax.tree_map(lambda arr: arr[indices], self.data)
        return batch

    def get_moments(self, modality: str) -> Tuple[jax.Array, jax.Array]:
        mean = self.data[modality].mean(0)
        std = self.data[modality].std(0)
        return mean, std

    @staticmethod
    def normalize_reward(dataset_name: str, rewards: jax.Array) -> jax.Array:
        if "antmaze" in dataset_name:
            return rewards * 100.0  # like in LAPO
        else:
            raise NotImplementedError("Reward normalization is implemented only for AntMaze yet!")


def get_timestep_from_idx(offline_data: dict, idx: int):
    return ExtendedTimeStep(
        step_type=step_type_lookup[offline_data['step_type'][idx]],
        reward=offline_data['reward'][idx],
        observation=offline_data['observation'][idx],
        discount=offline_data['discount'][idx],
        action=offline_data['action'][idx]
    )

class AbstractReplayBuffer(abc.ABC):
    @abc.abstractmethod
    def add(self, time_step):
        pass

    @abc.abstractmethod
    def __next__(self, ):
        pass

    @abc.abstractmethod
    def __len__(self, ):
        pass

class EfficientReplayBuffer(AbstractReplayBuffer):
    '''Fast + efficient replay buffer implementation in numpy.'''

    def __init__(self, buffer_size, batch_size, nstep, discount, frame_stack,
                 data_specs=None, sarsa=False):
        self.buffer_size = buffer_size
        self.data_dict = {}
        self.index = -1
        self.traj_index = 0
        self.frame_stack = frame_stack
        self._recorded_frames = frame_stack + 1
        self.batch_size = batch_size
        self.nstep = nstep
        self.discount = discount
        self.full = False
        self.discount_vec = np.power(discount, np.arange(nstep))  # n_step - first dim should broadcast
        self.next_dis = discount ** nstep
        self.sarsa = sarsa

    def _initial_setup(self, time_step):
        self.index = 0
        self.obs_shape = list(time_step.observation.shape)
        self.ims_channels = self.obs_shape[0] // self.frame_stack
        self.act_shape = time_step.action.shape

        self.obs = np.zeros([self.buffer_size, self.ims_channels, *self.obs_shape[1:]], dtype=np.uint8)
        self.act = np.zeros([self.buffer_size, *self.act_shape], dtype=np.float32)
        self.rew = np.zeros([self.buffer_size], dtype=np.float32)
        self.dis = np.zeros([self.buffer_size], dtype=np.float32)
        self.valid = np.zeros([self.buffer_size], dtype=np.bool_)

    def add_data_point(self, time_step):
        first = time_step.first()
        latest_obs = time_step.observation[-self.ims_channels:]
        if first:
            end_index = self.index + self.frame_stack
            end_invalid = end_index + self.frame_stack + 1
            if end_invalid > self.buffer_size:
                if end_index > self.buffer_size:
                    end_index = end_index % self.buffer_size
                    self.obs[self.index:self.buffer_size] = latest_obs
                    self.obs[0:end_index] = latest_obs
                    self.full = True
                else:
                    self.obs[self.index:end_index] = latest_obs
                end_invalid = end_invalid % self.buffer_size
                self.valid[self.index:self.buffer_size] = False
                self.valid[0:end_invalid] = False
            else:
                self.obs[self.index:end_index] = latest_obs
                self.valid[self.index:end_invalid] = False
            self.index = end_index
            self.traj_index = 1
        else:
            np.copyto(self.obs[self.index], latest_obs)  # Check most recent image
            np.copyto(self.act[self.index], time_step.action)
            self.rew[self.index] = time_step.reward
            self.dis[self.index] = time_step.discount
            self.valid[(self.index + self.frame_stack) % self.buffer_size] = False
            if self.traj_index >= self.nstep:
                self.valid[(self.index - self.nstep + 1) % self.buffer_size] = True
            self.index += 1
            self.traj_index += 1
            if self.index == self.buffer_size:
                self.index = 0
                self.full = True

    def add(self, time_step):
        if self.index == -1:
            self._initial_setup(time_step)
        self.add_data_point(time_step)

    def __next__(self, ):
        indices = np.random.choice(self.valid.nonzero()[0], size=self.batch_size)
        return self.gather_nstep_indices(indices)

    def gather_nstep_indices(self, indices):
        n_samples = indices.shape[0]
        all_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack, indices[i] + self.nstep)
                                      for i in range(n_samples)], axis=0) % self.buffer_size
        gather_ranges = all_gather_ranges[:, self.frame_stack:]  # bs x nstep
        obs_gather_ranges = all_gather_ranges[:, :self.frame_stack]
        nobs_gather_ranges = all_gather_ranges[:, -self.frame_stack:]

        all_rewards = self.rew[gather_ranges]

        # Could implement below operation as a matmul in pytorch for marginal additional speed improvement
        rew = np.sum(all_rewards * self.discount_vec, axis=1, keepdims=True)

        obs = np.reshape(self.obs[obs_gather_ranges], [n_samples, *self.obs_shape])
        nobs = np.reshape(self.obs[nobs_gather_ranges], [n_samples, *self.obs_shape])

        act = self.act[indices]
        dis = np.expand_dims(self.next_dis * self.dis[nobs_gather_ranges[:, -1]], axis=-1)

        if self.sarsa:
            nact = self.act[indices + self.nstep]
            return (obs, act, rew, dis, nobs, nact)

        return (obs, act, rew, dis, nobs)

    def __len__(self):
        if self.full:
            return self.buffer_size
        else:
            return self.index

    def get_train_and_val_indices(self, validation_percentage):
        all_indices = self.valid.nonzero()[0]
        num_indices = all_indices.shape[0]
        num_val = int(num_indices * validation_percentage)
        np.random.shuffle(all_indices)
        val_indices, train_indices = np.split(all_indices,
                                              [num_val])
        return train_indices, val_indices

    def get_obs_act_batch(self, indices):
        n_samples = indices.shape[0]
        obs_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack, indices[i])
                                      for i in range(n_samples)], axis=0) % self.buffer_size
        obs = np.reshape(self.obs[obs_gather_ranges], [n_samples, *self.obs_shape])
        act = self.act[indices]
        return obs, act


def load_offline_dataset_into_buffer(offline_dir, replay_buffer, frame_stack, replay_buffer_size):
    filenames = sorted(offline_dir.glob('*.hdf5'))
    num_steps = 0
    for filename in filenames:
        try:
            episodes = h5py.File(filename, 'r')
            episodes = {k: episodes[k][:] for k in episodes.keys()}
            add_offline_data_to_buffer(episodes, replay_buffer, framestack=frame_stack)
            length = episodes['reward'].shape[0]
            num_steps += length
        except Exception as e:
            print(f'Could not load episode {str(filename)}: {e}')
            continue
        print("Loaded {} offline timesteps so far...".format(int(num_steps)))
        if num_steps >= replay_buffer_size:
            break
    print("Finished, loaded {} timesteps.".format(int(num_steps)))


def add_offline_data_to_buffer(offline_data: dict, replay_buffer: EfficientReplayBuffer, framestack: int = 3):
    offline_data_length = offline_data['reward'].shape[0]
    for v in offline_data.values():
        assert v.shape[0] == offline_data_length
    for idx in range(offline_data_length):
        time_step = get_timestep_from_idx(offline_data, idx)
        if not time_step.first():
            stacked_frames.append(time_step.observation)
            time_step_stack = time_step._replace(observation=np.concatenate(stacked_frames, axis=0))
            replay_buffer.add(time_step_stack)
        else:
            stacked_frames = deque(maxlen=framestack)
            while len(stacked_frames) < framestack:
                stacked_frames.append(time_step.observation)
            time_step_stack = time_step._replace(observation=np.concatenate(stacked_frames, axis=0))
            replay_buffer.add(time_step_stack)


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 next_actions: np.ndarray,
                 mc_returns: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.next_actions = next_actions
        self.mc_returns = mc_returns
        self.size = size

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        indx = np.random.randint(self.size, size=batch_size)
        return {
            "states": self.observations[indx],
            "actions": self.actions[indx],
            "rewards": self.rewards[indx],
            "dones": self.dones_float[indx],
            "next_states": self.next_observations[indx],
            "next_actions": self.next_actions[indx],
            "mc_returns": self.mc_returns[indx],
        }


class OnlineReplayBuffer(Dataset):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int):

        observations = np.empty((capacity, *observation_space.shape),
                                dtype=observation_space.dtype)
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity,), dtype=np.float32)
        mc_returns = np.empty((capacity,), dtype=np.float32)
        masks = np.empty((capacity,), dtype=np.float32)
        dones_float = np.empty((capacity,), dtype=np.float32)
        next_observations = np.empty((capacity, *observation_space.shape),
                                     dtype=observation_space.dtype)
        next_actions = np.empty((capacity, action_dim), dtype=np.float32)
        super().__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         next_actions=next_actions,
                         mc_returns=mc_returns,
                         size=0)

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset,
                                num_samples=None):
        assert self.insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert self.capacity >= num_samples, 'Dataset cannot be larger than the replay buffer capacity.'

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[
            indices]
        self.next_actions[:num_samples] = dataset.next_actions[
            indices]
        self.mc_returns[:num_samples] = dataset.mc_returns[indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray,
               next_action: np.ndarray, mc_return: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation
        self.next_actions[self.insert_index] = next_action
        self.mc_returns[self.insert_index] = mc_return

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)



class D4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 env_name: str,
                 normalize_reward: bool,
                 discount: float,
                 clip_to_eps: bool = False,
                 eps: float = 1e-5):

        d4rl_data = qlearning_dataset(env, env_name, normalize_reward=normalize_reward, discount=discount)
        dataset = {
            "states": jnp.asarray(d4rl_data["observations"], dtype=jnp.float32),
            "actions": jnp.asarray(d4rl_data["actions"], dtype=jnp.float32),
            "rewards": jnp.asarray(d4rl_data["rewards"], dtype=jnp.float32),
            "next_states": jnp.asarray(d4rl_data["next_observations"], dtype=jnp.float32),
            "next_actions": jnp.asarray(d4rl_data["next_actions"], dtype=jnp.float32),
            "dones": jnp.asarray(d4rl_data["terminals"], dtype=jnp.float32),
            "mc_returns": jnp.asarray(d4rl_data["mc_returns"], dtype=jnp.float32)
        }

        super().__init__(dataset['states'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['dones'].astype(np.float32),
                         dones_float=dataset['dones'].astype(np.float32),
                         next_observations=dataset['next_states'].astype(
                             np.float32),
                         next_actions=dataset["next_actions"],
                         mc_returns=dataset["mc_returns"],
                         size=len(dataset['states']))


def make_env_and_dataset(env_name: str,
                         seed: int,
                         normalize_reward: bool,
                         discount: float) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env, env_name, normalize_reward, discount=discount)

    # if 'antmaze' in env_name:
    #     # dataset.rewards -= 1.0
    #     pass  # normalized in the batch instead
    #     # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
    #     # but I found no difference between (x - 0.5) * 4 and x - 1.0
    # elif ('halfcheetah' in env_name or 'walker2d' in env_name
    #       or 'hopper' in env_name):
    #     normalize(dataset)

    return env, dataset


def concat_batches(b1, b2):
    new_batch = {}
    for k in b1:
        new_batch[k] = np.concatenate((b1[k], b2[k]), axis=0)
    return new_batch
