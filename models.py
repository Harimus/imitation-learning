import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal
from torch.nn import Parameter, functional as F

ACTIVATION_FUNCTIONS = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU}


# Concatenates the state and action (previously one-hot discrete version)
def _join_state_action(state, action, action_size):
    return torch.cat([state, action], dim=1)


# Computes the squared distance between two sets of vectors
def _squared_distance(x, y):
  n_1, n_2, d = x.size(0), y.size(0), x.size(1)
  tiled_x, tiled_y = x.view(n_1, 1, d).expand(n_1, n_2, d), y.view(1, n_2, d).expand(n_1, n_2, d)
  return (tiled_x - tiled_y).pow(2).mean(dim=2)


# Gaussian/radial basis function/exponentiated quadratic kernel
def _gaussian_kernel(x, y, gamma=1):
  return torch.exp(-gamma * _squared_distance(x, y))


# Creates a sequential fully-connected network
def _create_fcnn(input_size, hidden_size, output_size, activation_function, dropout=0, final_gain=1.0, depth=2):
  assert activation_function in ACTIVATION_FUNCTIONS.keys()
  
  network_dims, layers = (input_size, *[hidden_size] * depth), []
  for l in range(len(network_dims) - 1):
    layer = nn.Linear(network_dims[l], network_dims[l + 1])
    nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain(activation_function))
    nn.init.constant_(layer.bias, 0)
    layers.append(layer)
    if dropout > 0: layers.append(nn.Dropout(p=dropout))
    layers.append(ACTIVATION_FUNCTIONS[activation_function]())

  final_layer = nn.Linear(network_dims[-1], output_size)
  nn.init.orthogonal_(final_layer.weight, gain=final_gain)
  nn.init.constant_(final_layer.bias, 0)
  layers.append(final_layer)

  return nn.Sequential(*layers)


class Actor(nn.Module):
  def __init__(self, state_size, action_size, hidden_size, activation_function='tanh', log_std_dev_init=-0.5, dropout=0):
    super().__init__()
    self.actor = _create_fcnn(state_size, hidden_size, output_size=action_size, activation_function=activation_function, dropout=dropout, final_gain=0.01)
    self.log_std_dev = Parameter(torch.full((action_size, ), log_std_dev_init, dtype=torch.float32))

  def forward(self, state):
    mean = self.actor(state)
    policy = Independent(Normal(mean, self.log_std_dev.exp()), 1)
    return policy

  # Calculates the log probability of an action a with the policy π(·|s) given state s
  def log_prob(self, state, action):
    return self.forward(state).log_prob(action)

  def _get_action_uncertainty(self, state, action):
    ensemble_policies = []
    for _ in range(5):  # Perform Monte-Carlo dropout for an implicit ensemble
      ensemble_policies.append(self.log_prob(state, action).exp())
    return torch.stack(ensemble_policies).var(dim=0)

  # Set uncertainty threshold at the 98th quantile of uncertainty costs calculated over the expert data
  def set_uncertainty_threshold(self, expert_state, expert_action):
    self.q = torch.quantile(self._get_action_uncertainty(expert_state, expert_action), 0.98).item()

  def predict_reward(self, state, action):
    # Calculate (raw) uncertainty cost
    uncertainty_cost = self._get_action_uncertainty(state, action)
    # Calculate clipped uncertainty cost
    neg_idxs = uncertainty_cost.less_equal(self.q)
    uncertainty_cost[neg_idxs] = -1
    uncertainty_cost[~neg_idxs] = 1
    return -uncertainty_cost


class Critic(nn.Module):
  def __init__(self, state_size, hidden_size, activation_function='tanh'):
    super().__init__()
    self.critic = _create_fcnn(state_size, hidden_size, output_size=1, activation_function=activation_function)

  def forward(self, state):
    value = self.critic(state).squeeze(dim=1)
    return value


class ActorCritic(nn.Module):
  def __init__(self, state_size, action_size, hidden_size, activation_function='tanh', log_std_dev_init=-0.5, dropout=0):
    super().__init__()
    self.actor = Actor(state_size, action_size, hidden_size, activation_function=activation_function, log_std_dev_init=log_std_dev_init, dropout=dropout)
    self.critic = Critic(state_size, hidden_size, activation_function=activation_function)

  def forward(self, state):
    policy, value = self.actor(state), self.critic(state)
    return policy, value

  def get_greedy_action(self, state):
    return self.actor(state).mean

  # Calculates the log probability of an action a with the policy π(·|s) given state s
  def log_prob(self, state, action):
    return self.actor.log_prob(state, action)


class GAILDiscriminator(nn.Module):
  def __init__(self, state_size, action_size, hidden_size, state_only=False, forward_kl=False):
    super().__init__()
    self.action_size, self.state_only, self.forward_kl = action_size, state_only, forward_kl
    self.discriminator = _create_fcnn(state_size if state_only else state_size + action_size, hidden_size, 1, 'tanh')

  def forward(self, state, action):
    D = self.discriminator(state if self.state_only else _join_state_action(state, action, self.action_size)).squeeze(dim=1)
    return D
  
  def predict_reward(self, state, action):
    D = torch.sigmoid(self.forward(state, action))
    h = torch.log(D + 1e-6) - torch.log1p(-D + 1e-6) # Add epsilon to improve numerical stability given limited floating point precision
    return torch.exp(h) * -h if self.forward_kl else h


class GMMILDiscriminator(nn.Module):
  def __init__(self, state_size, action_size, self_similarity=True, state_only=True):
    super().__init__()
    self.action_size, self.state_only = action_size, state_only
    self.gamma_1, self.gamma_2, self.self_similarity = None, None, self_similarity

  def predict_reward(self, state, action, expert_state, expert_action):
    state_action = state if self.state_only else _join_state_action(state, action, self.action_size)
    expert_state_action = expert_state if self.state_only else _join_state_action(expert_state, expert_action, self.action_size)
    
    # Use median heuristics to set data-dependent bandwidths
    if self.gamma_1 is None:
      self.gamma_1 = 1 / _squared_distance(state_action, expert_state_action).median().item()
      self.gamma_2 = 1 / _squared_distance(expert_state_action.transpose(0, 1), expert_state_action.transpose(0, 1)).median().item()

    # Calculate negative of witness function (based on kernel mean embeddings)
    similarity = (_gaussian_kernel(expert_state_action, state_action, gamma=self.gamma_1).mean(dim=0) + _gaussian_kernel(expert_state_action, state_action, gamma=self.gamma_2).mean(dim=0))
    return similarity - (_gaussian_kernel(state_action, state_action, gamma=self.gamma_1).mean(dim=0) + _gaussian_kernel(state_action, state_action, gamma=self.gamma_2).mean(dim=0)) if self.self_similarity else similarity


class AIRLDiscriminator(nn.Module):
  def __init__(self, state_size, action_size, hidden_size, discount, state_only=False, ):
    super().__init__()
    self.action_size, self.state_only = action_size, state_only
    self.discount = discount
    self.g = nn.Linear(state_size if state_only else state_size + action_size, 1)  # Reward function r
    self.h = _create_fcnn(state_size, hidden_size, 1, 'tanh')  # Shaping function Φ

  def reward(self, state, action):
    if self.state_only:
      return self.g(state).squeeze(dim=1)
    else:
      return self.g(_join_state_action(state, action, self.action_size)).squeeze(dim=1)

  def value(self, state):
    return self.h(state).squeeze(dim=1)

  def forward(self, state, action, next_state, log_policy, terminal):
    f = self.reward(state, action) + (1 - terminal) * (self.discount * self.value(next_state) - self.value(state))
    return f - log_policy  # Note that this is equivalent to sigmoid^-1(e^f / (e^f + π))

  def predict_reward(self, state, action, next_state, log_policy, terminal):
    D = torch.sigmoid(self.forward(state, action, next_state, log_policy, terminal))
    return torch.log(D + 1e-6) - torch.log1p(-D + 1e-6) # Add epsilon to improve numerical stability given limited floating point precision


class EmbeddingNetwork(nn.Module):
  def __init__(self, input_size, hidden_size, depth=2):
    super().__init__()
    self.embedding = _create_fcnn(input_size, hidden_size, input_size, 'leaky_relu', depth=depth)

  def forward(self, input):
    return self.embedding(input)


class REDDiscriminator(nn.Module):
  def __init__(self, state_size, action_size, hidden_size, state_only=False):
    super().__init__()
    self.action_size, self.state_only = action_size, state_only
    self.sigma_1 = None
    hidden_size=128
    depth=1
    print(f"Predictor (traned) network with hidden: {hidden_size} and depth: {depth}")
    self.predictor = EmbeddingNetwork(state_size if state_only else state_size + action_size, hidden_size, depth=depth)
    hidden_size=128
    depth=4
    print(f"Target (Fixed) network with hidden: {hidden_size} and depth: {depth}")
    self.target = EmbeddingNetwork(state_size if state_only else state_size + action_size, hidden_size, depth=depth)
    for param in self.target.parameters():
      param.requires_grad = False

  def forward(self, state, action):
    state_action = state if self.state_only else _join_state_action(state, action, self.action_size)
    prediction, target = self.predictor(state_action), self.target(state_action)
    return prediction, target

  # Originally, sets σ based such that r(s, a) from expert demonstrations ≈ 1; instead this uses kernel median heuristic (same as GMMIL)
  def set_sigma(self, expert_state, expert_action):
    prediction, target = self.forward(expert_state, expert_action)
    self.sigma_1 = 1 / _squared_distance(prediction.transpose(0, 1), target.transpose(0, 1)).median().item()

  def predict_reward(self, state, action):
    prediction, target = self.forward(state, action)
    #return torch.exp(-self.sigma_1 * (target-prediction).pow(2).mean(axis=-1))
    return _gaussian_kernel(prediction, target, gamma=self.sigma_1).mean(dim=1)
