from collections import deque
import time

import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from torch import optim
from tqdm import tqdm

from environments import ENVS
from evaluation import evaluate_agent
from models import Actor, ActorCritic, AIRLDiscriminator, GAILDiscriminator, GMMILDiscriminator, REDDiscriminator
from training import TransitionDataset, adversarial_imitation_update, behavioural_cloning_update, ppo_update, target_estimation_update
from utils import flatten_list_dicts, lineplot


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
  # Configuration check
  assert cfg.env_type in ENVS.keys()
  assert cfg.algorithm in ['AIRL', 'DRIL', 'FAIRL', 'GAIL', 'GMMIL', 'PUGAIL', 'RED', 'BC', 'PPO']
  # General setup
  np.random.seed(cfg.seed)
  torch.manual_seed(cfg.seed)

  # Set up environment
  env = ENVS[cfg.env_type](cfg.env_name, load_data=True)
  env.seed(cfg.seed)
  expert_trajectories = env.get_dataset()  # Load expert trajectories dataset
  state_size, action_size = env.observation_space.shape[0], env.action_space.shape[0]
  
  # Set up agent
  agent = ActorCritic(state_size, action_size, cfg.model.hidden_size, log_std_dev_init=cfg.model.log_std_dev_init)
  agent_optimiser = optim.RMSprop(agent.parameters(), lr=cfg.reinforcement.learning_rate, alpha=0.9)
  # Set up imitation learning components
  if cfg.algorithm in ['AIRL', 'DRIL', 'FAIRL', 'GAIL', 'GMMIL', 'PUGAIL', 'RED']:
    if cfg.algorithm == 'AIRL':
      discriminator = AIRLDiscriminator(state_size, action_size, cfg.model.hidden_size, cfg.reinforcement.discount, state_only=cfg.imitation.state_only)
    elif cfg.algorithm == 'DRIL':
      discriminator = Actor(state_size, action_size, cfg.model.hidden_size, dropout=0.1)
    elif cfg.algorithm in ['FAIRL', 'GAIL', 'PUGAIL']:
      discriminator = GAILDiscriminator(state_size, action_size, cfg.model.hidden_size, state_only=cfg.imitation.state_only, forward_kl=cfg.algorithm == 'FAIRL')
    elif cfg.algorithm == 'GMMIL':
      discriminator = GMMILDiscriminator(state_size, action_size, self_similarity=cfg.imitation.self_similarity, state_only=cfg.imitation.state_only)
    elif cfg.algorithm == 'RED':
      discriminator = REDDiscriminator(state_size, action_size, cfg.model.hidden_size, state_only=cfg.imitation.state_only)
    if cfg.algorithm in ['AIRL', 'DRIL', 'FAIRL', 'GAIL', 'PUGAIL', 'RED']:
      if cfg.discriminator_adam:
        discriminator_optimiser = optim.Adam(discriminator.parameters(), lr=cfg.imitation.learning_rate)
      else:
        discriminator_optimiser = optim.RMSprop(discriminator.parameters(), lr=cfg.imitation.learning_rate)

  # Metrics
  metrics = dict(train_steps=[], train_returns=[], test_steps=[], test_returns=[], 
  update_steps=[], predicted_returns=[], predicted_expert_returns=[], alphas=[], entropy=[], Q_values=[])
  recent_returns = deque(maxlen=cfg.evaluation.average_window)  # Stores most recent evaluation returns
  recent_returns = deque(maxlen=cfg.evaluation.average_window)  # Stores most recent evaluation returns
  # Main training loop
  state, terminal, train_return, trajectories = env.reset(), False, 0, []
  if cfg.algorithm in ['AIRL', 'FAIRL', 'GAIL', 'PUGAIL']: policy_trajectory_replay_buffer = deque(maxlen=cfg.imitation.replay_size)
  pbar = tqdm(range(1, cfg.steps + 1), unit_scale=1, smoothing=0)
  if cfg.check_time_usage: start_time = time.time()  # Performance tracking
  for step in pbar:
    # Perform initial training (if needed)
    if cfg.algorithm in ['BC', 'DRIL', 'RED']:
      if step == 1:
        for _ in tqdm(range(cfg.imitation.epochs), leave=False):
          if cfg.algorithm == 'BC':
            # Perform behavioural cloning updates offline
            behavioural_cloning_update(agent, expert_trajectories, agent_optimiser, cfg.training.batch_size)
          elif cfg.algorithm == 'DRIL':
            # Perform behavioural cloning updates offline on policy ensemble (dropout version)
            behavioural_cloning_update(discriminator, expert_trajectories, discriminator_optimiser, cfg.training.batch_size)
            with torch.no_grad():  # TODO: Check why inference mode fails?
              discriminator.set_uncertainty_threshold(expert_trajectories['states'], expert_trajectories['actions'])
          elif cfg.algorithm == 'RED':
            # Train predictor network to match random target network
            target_estimation_update(discriminator, expert_trajectories, discriminator_optimiser, cfg.training.batch_size)
            with torch.inference_mode():
              discriminator.set_sigma(expert_trajectories['states'], expert_trajectories['actions'])

        if cfg.check_time_usage:
          metrics['pre_training_time'] = time.time() - start_time
          start_time = time.time()

    if cfg.algorithm != 'BC':
      # Collect set of trajectories by running policy π in the environment
      with torch.inference_mode():
        policy, value = agent(state)
        action = policy.sample()
        log_prob_action = policy.log_prob(action)
        next_state, reward, terminal = env.step(action)
        train_return += reward
        trajectories.append(dict(states=state, actions=action, rewards=torch.tensor([reward], dtype=torch.float32), terminals=torch.tensor([terminal], dtype=torch.float32), log_prob_actions=log_prob_action, old_log_prob_actions=log_prob_action.detach(), values=value))
        state = next_state

      if terminal:
        # Store metrics and reset environment
        metrics['train_steps'].append(step)
        metrics['train_returns'].append([train_return])
        pbar.set_description(f'Step: {step} | Return: {train_return}')
        state, train_return = env.reset(), 0

      # Update models
      if len(trajectories) >= cfg.training.batch_size:
        policy_trajectories = flatten_list_dicts(trajectories)  # Flatten policy trajectories (into a single batch for efficiency; valid for feedforward networks)

        if cfg.algorithm in ['AIRL', 'DRIL', 'FAIRL', 'GAIL', 'GMMIL', 'PUGAIL', 'RED']:
          # Train discriminator
          if cfg.algorithm in ['AIRL', 'FAIRL', 'GAIL', 'PUGAIL']:
            # Use a replay buffer of previous trajectories to prevent overfitting to current policy
            policy_trajectory_replay_buffer.append(policy_trajectories)
            policy_trajectory_replays = flatten_list_dicts(policy_trajectory_replay_buffer)
            for _ in tqdm(range(cfg.imitation.epochs), leave=False):
              adversarial_imitation_update(cfg.algorithm, agent, discriminator, expert_trajectories, TransitionDataset(policy_trajectory_replays), discriminator_optimiser, cfg.training.batch_size, cfg.imitation.r1_reg_coeff, cfg.get('pos_class_prior', 0.5), cfg.get('nonnegative_margin', 0))

          # Predict rewards
          states, actions, next_states, terminals = policy_trajectories['states'], policy_trajectories['actions'], torch.cat([policy_trajectories['states'][1:], next_state]), policy_trajectories['terminals']
          expert_states, expert_actions, expert_next_states =  expert_trajectories['states'][1:], expert_trajectories['actions'][1:], expert_trajectories['states'][:-1]
          subsample_idx = torch.ones(expert_states.shape[0]).multinomial(256) #temp hardcode
          expert_states, expert_actions, expert_next_states = expert_states[subsample_idx], expert_actions[subsample_idx], expert_next_states[subsample_idx]
          expert_rewards = None
          with torch.inference_mode():
            if cfg.algorithm == 'AIRL':
              policy_trajectories['rewards'] = discriminator.predict_reward(states, actions, next_states, policy_trajectories['log_prob_actions'], terminals)
            elif cfg.algorithm == 'DRIL':
              policy_trajectories['rewards'] = discriminator.predict_reward(states, actions)
              if cfg.save_aux_metrics:
                expert_rewards = discriminator.predict_reward(expert_states, expert_actions)
            elif cfg.algorithm in ['FAIRL', 'GAIL', 'PUGAIL']:
              policy_trajectories['rewards'] = discriminator.predict_reward(states, actions)
            elif cfg.algorithm == 'GMMIL':
              expert_states, expert_actions = expert_trajectories['states'], expert_trajectories['actions']
              policy_trajectories['rewards'] = discriminator.predict_reward(states, actions, expert_states, expert_actions)
            elif cfg.algorithm == 'RED':
              policy_trajectories['rewards'] = discriminator.predict_reward(states, actions)
              if cfg.save_aux_metrics:
                expert_rewards = discriminator.predict_reward(expert_states, expert_actions)

        # Perform PPO updates (includes GAE re-estimation with updated value function)
        for _ in tqdm(range(cfg.reinforcement.ppo_epochs), leave=False):
          ppo_update(agent, policy_trajectories, next_state, agent_optimiser, cfg.reinforcement.discount, cfg.reinforcement.trace_decay, cfg.reinforcement.ppo_clip, cfg.reinforcement.value_loss_coeff, cfg.reinforcement.entropy_loss_coeff, cfg.reinforcement.max_grad_norm)
        if cfg.save_aux_metrics:
          metrics['update_steps'].append(step), metrics['predicted_returns'].append(policy_trajectories['rewards'].numpy()), 
          metrics['predicted_expert_returns'].append(expert_rewards.numpy())
          with torch.inference_mode():
              log_prob = agent.actor.log_prob(states, actions)
              entropy = -log_prob.exp() * log_prob
              metrics['entropy'].append(entropy.detach().numpy())
              Q_value = agent.critic(states)
              metrics['Q_values'].append(Q_value.numpy())
        trajectories, policy_trajectories = [], None
    
    
    # Evaluate agent and plot metrics
    if step % cfg.evaluation.interval == 0 and not cfg.check_time_usage:
      test_returns = evaluate_agent(agent, cfg.evaluation.episodes, ENVS[cfg.env_type], cfg.env_name, cfg.seed)
      recent_returns.append(sum(test_returns) / cfg.evaluation.episodes)
      metrics['test_steps'].append(step)
      metrics['test_returns'].append(test_returns)
      lineplot(metrics['test_steps'], metrics['test_returns'], title='test_returns')
      if len(metrics['train_returns']) > 0:  # Plot train returns if any
        lineplot(metrics['train_steps'], metrics['train_returns'], title='train_returns')
        if cfg.save_aux_metrics:
          lineplot(metrics['update_steps'], metrics['predicted_returns'], metrics['predicted_expert_returns'], filename='predicted_returns', title=f'{cfg.env_name} : {cfg.algorithm}')
          lineplot(metrics['update_steps'], metrics['entropy'], filename='ppo_entropy', title=f'{cfg.env_name} : {cfg.algorithm}')
          lineplot(metrics['update_steps'], metrics['Q_values'], filename='Q_values', title=f'{cfg.env_name} : {cfg.algorithm}')

  if cfg.check_time_usage:
    metrics['training_time'] = time.time() - start_time

  if cfg.save_trajectories:
    # Store trajectories from agent after training
    _, trajectories = evaluate_agent(agent, cfg.evaluation.episodes, ENVS[cfg.env_type], cfg.env_name, cfg.seed, return_trajectories=True, render=cfg.render)
    torch.save(trajectories, 'trajectories.pth')
  # Save agent and metrics
  torch.save(agent.state_dict(), 'agent.pth')
  if cfg.algorithm in ['AIRL', 'DRIL', 'FAIRL', 'GAIL', 'PUGAIL', 'RED']: torch.save(discriminator.state_dict(), 'discriminator.pth')
  torch.save(metrics, 'metrics.pth')

  env.close()
  return sum(recent_returns) / float(1 if cfg.algorithm == 'BC' else cfg.evaluation.average_window)


if __name__ == '__main__':
  main()
