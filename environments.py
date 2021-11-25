from logging import ERROR

import d4rl_pybullet
import d4rl
import gym
import numpy as np
import pickle, os
import torch
from gym.spaces import Box

from training import TransitionDataset

gym.logger.set_level(ERROR)  # Ignore warnings from Gym logger

D4RL_ENV_NAMES = ['ant-bullet-medium-v0', 'halfcheetah-bullet-medium-v0', 'hopper-bullet-medium-v0', 'walker2d-bullet-medium-v0']


# Test environment for testing the code
class PendulumEnv():
  def __init__(self, env_name=''):
    self.env = gym.make('Pendulum-v0')
    self.env.action_space.high, self.env.action_space.low = torch.as_tensor(self.env.action_space.high), torch.as_tensor(self.env.action_space.low)  # Convert action space for action clipping

  def reset(self):
    state = self.env.reset()
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)  # Add batch dimension to state

  def step(self, action):
    action = action.clamp(min=self.env.action_space.low, max=self.env.action_space.high)  # Clip actions
    state, reward, terminal, _ = self.env.step(action[0].detach().numpy())  # Remove batch dimension from action
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0), reward, terminal  # Add batch dimension to state

  def seed(self, seed):
    return self.env.seed(seed)

  def render(self):
    return self.env.render()

  def close(self):
    self.env.close()

  @property
  def observation_space(self):
    return self.env.observation_space

  @property
  def action_space(self):
    return self.env.action_space

  def get_dataset(self, size=0, dtype=torch.float):
    return []



class D4RL_pybulletEnv():
  def __init__(self, env_name):
    assert env_name in D4RL_ENV_NAMES
    
    self.env = gym.make(env_name)
    if 'ant' in env_name:
      self.use_RED_DATA = True
    if self.use_RED_DATA:
      file_location = os.path.expanduser('~/external_repos/RED/data/Ant-v2.pkl')
      filehande = open(file_location, 'rb')
      self.dataset = pickle.load(filehande)
    else:
      self.dataset = self.env.get_dataset()  # Load dataset before (potentially) adjusting observation_space (fails assertion check otherwise)
    self.env.action_space.high, self.env.action_space.low = torch.as_tensor(self.env.action_space.high), torch.as_tensor(self.env.action_space.low)  # Convert action space for action clipping

  def reset(self):
    state = self.env.reset()
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)  # Add batch dimension to state

  def step(self, action):
    action = action.clamp(min=self.env.action_space.low, max=self.env.action_space.high)  # Clip actions
    state, reward, terminal, _ = self.env.step(action[0].detach().numpy())  # Remove batch dimension from action
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0), reward, terminal  # Add batch dimension to state

  def seed(self, seed):
    return self.env.seed(seed)

  def render(self):
    return self.env.render()

  def close(self):
    self.env.close()

  @property
  def observation_space(self):
    return self.env.observation_space

  @property
  def action_space(self):
    return self.env.action_space

  def get_dataset(self, size=0, subsample=20):
    if self.use_RED_DATA:
      states = torch.as_tensor(self.dataset['observations'][:-1], dtype=torch.float32)
      actions = torch.as_tensor(self.dataset['actions'][:-1], dtype=torch.float32)
      actions = actions.squeeze() # (4000, 1, 8) -> (4000, 8)
      next_states = torch.as_tensor(self.dataset['observations'][1:], dtype=torch.float32)
      terminals = torch.zeros(len(self.dataset['actions'][:-1]), dtype=torch.float32)
      weights = torch.ones_like(terminals)
      rewards = torch.zeros_like(terminals)
      transitions = dict(states=states, actions=actions, next_states=next_states, terminals=terminals, weights=weights, rewards=rewards)
      return TransitionDataset(transitions) 
    dataset = self.env.get_dataset()
    N = dataset['rewards'].shape[0]
    dataset_out = {'states': torch.as_tensor(dataset['observations'][:-1], dtype=torch.float32),
                   'actions': torch.as_tensor(dataset['actions'][:-1], dtype=torch.float32),
                   'rewards': torch.as_tensor(dataset['rewards'][:-1], dtype=torch.float32),
                   'next_states': torch.as_tensor(dataset['observations'][1:], dtype=torch.float32), 
                   'terminals': torch.as_tensor(dataset['terminals'][:-1], dtype=torch.float32)}
    # Postprocess
    if size > 0 and size < N:
      for key in dataset_out.keys():
        dataset_out[key] = dataset_out[key][0:size]
    if subsample > 0:
      for key in dataset_out.keys():
        dataset_out[key] = dataset_out[key][0::subsample]

    return TransitionDataset(dataset_out)


OLD_DATA_NAMES = {'ant-expert-v2': 'Ant-v2.pkl', 'halfcheetah-expert-v2': 'Halfcheetah-v2.pkl',
                  'hopper-expert-v2': 'Hopper-v2.pkl', 'walker2d-expert-v2': 'Walker2d-v2.pkl' }
class D4RLEnv():
  def __init__(self, env_name, absorbing=False, load_data=False, use_old_data=False):
    self.env = gym.make(env_name)
    if load_data:
      self.use_RED_DATA = use_old_data
      if self.use_RED_DATA:
        filename = OLD_DATA_NAMES[env_name]
        file_location = os.path.expanduser(f'~/external_repos/RED/data/{filename}')
        filehande = open(file_location, 'rb')
        print(f"Using RED OG data from: {file_location} , (note: no subsampling) ")
        self.dataset = pickle.load(filehande)
      else:
        self.dataset = self.env.get_dataset()  # Load dataset before (potentially) adjusting observation_space (fails assertion check otherwise)
    self.env.action_space.high, self.env.action_space.low = torch.as_tensor(self.env.action_space.high), torch.as_tensor(self.env.action_space.low)  # Convert action space for action clipping

    self.absorbing = absorbing
    if absorbing: self.env.observation_space = Box(low=np.concatenate([self.env.observation_space.low, np.zeros(1)]), high=np.concatenate([self.env.observation_space.high, np.ones(1)]))  # Append absorbing indicator bit to state dimension (assumes 1D state space)

  def reset(self):
    state = self.env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)  # Add batch dimension to state
    if self.absorbing: state = torch.cat([state, torch.zeros(state.size(0), 1)], dim=1)  # Add absorbing indicator (zero) to state
    return state 

  def step(self, action):
    action = action.clamp(min=self.env.action_space.low, max=self.env.action_space.high)  # Clip actions
    state, reward, terminal, _ = self.env.step(action[0].detach().numpy())  # Remove batch dimension from action
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)  # Add batch dimension to state
    if self.absorbing: state = torch.cat([state, torch.zeros(state.size(0), 1)], dim=1)  # Add absorbing indicator (zero) to state (absorbing state rewriting done in replay memory)
    return state, reward, terminal

  def seed(self, seed):
    return self.env.seed(seed)

  def render(self):
    return self.env.render()

  def close(self):
    self.env.close()

  @property
  def observation_space(self):
    return self.env.observation_space

  @property
  def action_space(self):
    return self.env.action_space

  @property
  def max_episode_steps(self):
    return self.env._max_episode_steps

  def get_dataset(self, trajectories=-1, subsample=20):
    # Extract data
    if self.use_RED_DATA:
      states = torch.as_tensor(self.dataset['observations'][:-1], dtype=torch.float32)
      actions = torch.as_tensor(self.dataset['actions'][:-1], dtype=torch.float32)
      actions = actions.squeeze() # (4000, 1, 8) -> (4000, 8)
      next_states = torch.as_tensor(self.dataset['observations'][1:], dtype=torch.float32)
      terminals = torch.zeros(len(self.dataset['actions'][:-1]), dtype=torch.float32)
      weights = torch.ones_like(terminals)
      rewards = torch.zeros_like(terminals)
      transitions = dict(states=states, actions=actions, next_states=next_states, terminals=terminals, weights=weights, rewards=rewards)
      return TransitionDataset(transitions) 
    states = torch.as_tensor(self.dataset['observations'], dtype=torch.float32)
    actions = torch.as_tensor(self.dataset['actions'], dtype=torch.float32)
    next_states = torch.as_tensor(self.dataset['next_observations'], dtype=torch.float32)
    terminals = torch.as_tensor(self.dataset['terminals'], dtype=torch.float32)
    timeouts = torch.as_tensor(self.dataset['timeouts'], dtype=torch.float32)
    state_size, action_size = states.size(1), actions.size(1)
    # Split into separate trajectories
    states_list, actions_list, next_states_list, terminals_list, weights_list, timeout_list = [], [], [], [], [], []
    terminal_idxs, timeout_idxs = terminals.nonzero().flatten(), timeouts.nonzero().flatten()
    ep_end_idxs = torch.sort(torch.cat([torch.tensor([-1]), terminal_idxs, timeout_idxs], dim=0))[0]
    for i in range(len(ep_end_idxs) - 1):
      states_list.append(states[ep_end_idxs[i] + 1:ep_end_idxs[i + 1] + 1])
      actions_list.append(actions[ep_end_idxs[i] + 1:ep_end_idxs[i + 1] + 1])
      next_states_list.append(next_states[ep_end_idxs[i] + 1:ep_end_idxs[i + 1] + 1])
      terminals_list.append(terminals[ep_end_idxs[i] + 1:ep_end_idxs[i + 1] + 1])  # Only store true terminations; timeouts should not be treated as such
      weights_list.append(torch.ones_like(terminals_list[-1]))  # Add an importance weight of 1 to every transition
      timeout_list.append(ep_end_idxs[i + 1] in timeout_idxs)  # Store if episode terminated due to timeout
    # Pick number of trajectories
    if trajectories > -1:
      states_list = states_list[:trajectories]
      actions_list = actions_list[:trajectories]
      next_states_list = next_states_list[:trajectories]
      terminals_list = terminals_list[:trajectories]
      weights_list = weights_list[:trajectories]
    # Wrap for absorbing states
    if self.absorbing:  
      absorbing_state, absorbing_action = torch.cat([torch.zeros(1, state_size), torch.ones(1, 1)], dim=1), torch.zeros(1, action_size)  # Create absorbing state and absorbing action
      for i in range(len(states_list)):
        # Append absorbing indicator (zero)
        states_list[i] = torch.cat([states_list[i], torch.zeros(states_list[i].size(0), 1)], dim=1)
        next_states_list[i] = torch.cat([next_states_list[i], torch.zeros(next_states_list[i].size(0), 1)], dim=1)
        if not timeout_list[i]:  # Apply for episodes that did not terminate due to time limits
          # Replace the final next state with the absorbing state and overwrite terminal status
          next_states_list[i][-1] = absorbing_state
          terminals_list[i][-1] = 0
          weights_list[i][-1] = 1 / subsample  # Importance weight absorbing state as kept during subsampling
          # Add absorbing state to absorbing state transition
          states_list[i] = torch.cat([states_list[i], absorbing_state], dim=0)
          actions_list[i] = torch.cat([actions_list[i], absorbing_action], dim=0)
          next_states_list[i] = torch.cat([next_states_list[i], absorbing_state], dim=0)
          terminals_list[i] = torch.cat([terminals_list[i], torch.zeros(1)], dim=0)
          weights_list[i] = torch.cat([weights_list[i], torch.full((1, ), 1 / subsample)], dim=0)  # Importance weight absorbing state as kept during subsampling
    # Subsample within trajectories
    if subsample > 0:
      for i in range(len(states_list)):
        rand_start_idx, T = np.random.choice(subsample), len(states_list[i])  # Subsample from random index in 0 to N-1 (procedure from original GAIL implementation)
        idxs = range(rand_start_idx, T, subsample)
        if self.absorbing: idxs = sorted(list(set(idxs) | set([T - 2, T - 1])))  # Subsample but keep absorbing state transitions
        states_list[i] = states_list[i][idxs]
        actions_list[i] = actions_list[i][idxs]
        next_states_list[i] = next_states_list[i][idxs]
        terminals_list[i] = terminals_list[i][idxs]
        weights_list[i] = weights_list[i][idxs]

    transitions = {'states': torch.cat(states_list, dim=0), 'actions': torch.cat(actions_list, dim=0), 'next_states': torch.cat(next_states_list, dim=0), 'terminals': torch.cat(terminals_list, dim=0), 'weights': torch.cat(weights_list, dim=0)}
    transitions['rewards'] = torch.zeros_like(transitions['terminals'])  # Pass 0 rewards to replay memory for interoperability
    return TransitionDataset(transitions)

ENVS = {'ant': D4RLEnv, 'halfcheetah': D4RLEnv, 'hopper': D4RLEnv, 'pendulum': PendulumEnv, 'walker2d': D4RLEnv}
