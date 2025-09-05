# traders/ppo_core.py
"""
Feedforward (MLP) PPO implementation for trading agents.
Based on CleanRL's PPO implementation but adapted for single-agent trading.
"""

import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.distributions.categorical import Categorical


# --- Utilities ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize weights with Orthogonal and biases."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RunningMeanStd:
    """Running statistics for reward normalization."""
    def __init__(self, epsilon=1e-8, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
        self.epsilon = epsilon
        
    def update(self, x):
        """Update running statistics with batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if isinstance(x, np.ndarray) else 1
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        # Update mean
        self.mean = self.mean + delta * batch_count / total_count
        
        # Update variance (Welford's online algorithm)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        
        self.count = total_count
        
    def normalize(self, x):
        """Normalize input using running statistics."""
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)


class MLPAgent(nn.Module):
    """Simple feedforward actor-critic agent."""
    def __init__(self, state_dim, action_dim, hidden_layers=None):
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [256, 128]  # Default architecture
        
        # Build critic (value) network
        critic_layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            critic_layers.append(layer_init(nn.Linear(input_dim, hidden_dim)))
            critic_layers.append(nn.Tanh())
            input_dim = hidden_dim
        critic_layers.append(layer_init(nn.Linear(input_dim, 1), std=1.0))
        self.critic = nn.Sequential(*critic_layers)
        
        # Build actor (policy) network
        actor_layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            actor_layers.append(layer_init(nn.Linear(input_dim, hidden_dim)))
            actor_layers.append(nn.Tanh())
            input_dim = hidden_dim
        actor_layers.append(layer_init(nn.Linear(input_dim, action_dim), std=0.01))
        self.actor = nn.Sequential(*actor_layers)
    
    def get_value(self, x):
        """Get value estimate for states."""
        return self.critic(x).squeeze(-1)
    
    def get_action_and_value(self, x, action=None):
        """Get action, log prob, entropy, and value for states."""
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), self.critic(x).squeeze(-1)


class PPOLogic:
    """Feedforward PPO logic for trading agents."""
    
    def __init__(self, state_dim, action_dim, config, seed):
        self.logger = logging.getLogger('rl_agent_logic.ppo')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        params = config.get('rl_params', {})
        
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"PPO using device: {self.device}")
        
        # Hyperparameters
        self.gamma = float(params.get('gamma', 0.99))
        self.gae_lambda = float(params.get('gae_lambda', 0.95))
        self.clip_coef = float(params.get('clip_epsilon', 0.2))
        self.ent_coef_initial = float(params.get('entropy_coef', 0.01))
        self.ent_coef = self.ent_coef_initial
        self.vf_coef = float(params.get('value_loss_coef', 0.5))
        self.learning_rate_initial = float(params.get('learning_rate', params.get('lr', 3e-4)))
        self.learning_rate = self.learning_rate_initial
        self.update_epochs = int(params.get('n_epochs', params.get('update_epochs', 4)))
        self.norm_adv = params.get('norm_adv', True)
        self.clip_vloss = params.get('clip_vloss', True)
        self.max_grad_norm = float(params.get('max_grad_norm', 0.5))
        self.batch_size = int(params.get('batch_size', 64))
        
        # Optimization features
        self.target_kl = float(params.get('target_kl', 0.015)) if params.get('target_kl') else None
        self.use_lr_annealing = params.get('use_lr_annealing', True)
        self.use_entropy_annealing = params.get('use_entropy_annealing', True)
        self.use_reward_scaling = params.get('use_reward_scaling', True)
        self.total_training_rounds = config.get('num_training_rounds', config.get('num_rounds', 1000))
        self.current_round = 0
        
        # Exploration parameters
        self.epsilon_greedy = float(params.get('epsilon_greedy', 0.0))
        
        # Network parameters
        hidden_layers = params.get('nn_hidden_layers', [256, 128])
        
        # Initialize agent
        self.agent = MLPAgent(state_dim, action_dim, hidden_layers).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)
        
        # Initialize reward normalization
        self.reward_rms = RunningMeanStd() if self.use_reward_scaling else None
        
        # Storage for trajectory
        self.trajectory_buffer = []
        self.is_training = True
        self.last_train_stats = {}
    
    def set_mode(self, training=True):
        """Set training/evaluation mode."""
        self.is_training = training
        if training:
            self.agent.train()
        else:
            self.agent.eval()
        self.logger.debug(f"PPO mode set to: {'Training' if training else 'Evaluation'}")
    
    def get_action(self, state):
        """Get action for a single state during rollout."""
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Epsilon-greedy exploration during training
            if self.is_training and self.epsilon_greedy > 0:
                if random.random() < self.epsilon_greedy:
                    # Random action
                    action = torch.randint(0, self.action_dim, (1,)).to(self.device)
                    # Still need log_prob and value for the actual action
                    _, log_prob, _, value = self.agent.get_action_and_value(state_tensor, action)
                    return action.item(), log_prob.item(), value.item()
            
            action, log_prob, _, value = self.agent.get_action_and_value(state_tensor)
            return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, done, log_prob, value):
        """Store transition in buffer."""
        if not self.is_training:
            return
        
        state_np = np.array(state, dtype=np.float32) if state is not None else np.zeros(self.state_dim, dtype=np.float32)
        
        # Normalize reward if enabled
        if self.use_reward_scaling and self.reward_rms is not None:
            self.reward_rms.update(np.array([reward]))
            reward = self.reward_rms.normalize(np.array([reward]))[0]
        
        # Convert tensors to python types for storage
        action_item = action.item() if isinstance(action, torch.Tensor) else action
        log_prob_item = log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob
        value_item = value.item() if isinstance(value, torch.Tensor) else value
        
        self.trajectory_buffer.append((state_np, action_item, reward, done, log_prob_item, value_item))
    
    def clear_buffer(self):
        """Clear trajectory buffer."""
        self.trajectory_buffer = []
        self.logger.debug("Trajectory buffer cleared.")
    
    def update_schedule(self, current_round):
        """Update learning rate and entropy coefficient based on training progress."""
        self.current_round = current_round
        
        if self.total_training_rounds > 0:
            progress = min(current_round / self.total_training_rounds, 1.0)
            
            # Linear annealing for learning rate
            if self.use_lr_annealing:
                self.learning_rate = self.learning_rate_initial * (1.0 - progress)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
                self.logger.debug(f"Learning rate updated to {self.learning_rate:.6f}")
            
            # Linear annealing for entropy coefficient
            if self.use_entropy_annealing:
                self.ent_coef = self.ent_coef_initial * (1.0 - progress)
                self.logger.debug(f"Entropy coefficient updated to {self.ent_coef:.6f}")
    
    def learn(self, next_state):
        """Perform PPO update based on stored trajectory."""
        if not self.is_training or not self.trajectory_buffer:
            self.trajectory_buffer = []
            self.last_train_stats = {}
            return
        
        traj_len = len(self.trajectory_buffer)
        if traj_len < 2:
            self.logger.debug(f"Trajectory too short ({traj_len} steps). Skipping update.")
            self.trajectory_buffer = []
            self.last_train_stats = {}
            return
        
        self.logger.debug(f"Performing PPO update (trajectory size: {traj_len})")
        
        # Prepare data from buffer
        states, actions, rewards, dones, old_log_probs, values = map(np.array, zip(*self.trajectory_buffer))
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        
        # Calculate advantages using GAE
        with torch.no_grad():
            # Bootstrap value for last state
            next_state_tensor = torch.tensor(np.array(next_state), dtype=torch.float32).unsqueeze(0).to(self.device)
            next_value = self.agent.get_value(next_state_tensor).item()
            
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            
            for t in reversed(range(traj_len)):
                if t == traj_len - 1:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + values
        
        # PPO update loop
        clipfracs = []
        all_pg_loss, all_v_loss, all_entropy_loss, all_approx_kl = [], [], [], []
        
        self.agent.train()
        
        for epoch in range(self.update_epochs):
            # Process full batch (no minibatching for single-agent)
            _, new_log_probs, entropy, new_values = self.agent.get_action_and_value(states, actions)
            
            logratio = new_log_probs - old_log_probs
            ratio = logratio.exp()
            
            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs.append(((ratio - 1.0).abs() > self.clip_coef).float().mean().item())
            
            # Normalize advantages
            if self.norm_adv:
                adv_mean = advantages.mean()
                adv_std = advantages.std()
                advantages_norm = (advantages - adv_mean) / (adv_std + 1e-8) if adv_std > 1e-8 else advantages
            else:
                advantages_norm = advantages
            
            # Policy loss
            pg_loss1 = -advantages_norm * ratio
            pg_loss2 = -advantages_norm * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            
            # Value loss
            if self.clip_vloss:
                v_loss_unclipped = (new_values - returns) ** 2
                v_clipped = values + torch.clamp(new_values - values, -self.clip_coef, self.clip_coef)
                v_loss_clipped = (v_clipped - returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_values - returns) ** 2).mean()
            
            entropy_loss = entropy.mean()
            loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            all_pg_loss.append(pg_loss.item())
            all_v_loss.append(v_loss.item())
            all_entropy_loss.append(entropy_loss.item())
            all_approx_kl.append(approx_kl.item())
            
            # Target KL early stopping
            if self.target_kl is not None and approx_kl > self.target_kl:
                self.logger.debug(f"Early stopping at epoch {epoch} due to KL divergence {approx_kl:.4f} > {self.target_kl}")
                break
        
        self.trajectory_buffer = []  # Clear buffer
        
        # Store stats
        self.last_train_stats = {
            'avg_policy_loss': np.mean(all_pg_loss) if all_pg_loss else np.nan,
            'avg_value_loss': np.mean(all_v_loss) if all_v_loss else np.nan,
            'avg_entropy': np.mean(all_entropy_loss) if all_entropy_loss else np.nan,
            'avg_approx_kl': np.mean(all_approx_kl) if all_approx_kl else np.nan,
            'avg_clip_frac': np.mean(clipfracs) if clipfracs else np.nan
        }
        
        self.logger.debug(f"PPO Update Stats: PL={self.last_train_stats['avg_policy_loss']:.4f}, "
                         f"VL={self.last_train_stats['avg_value_loss']:.4f}")
    
    def save_model(self, path_prefix):
        """Save model weights."""
        agent_path = f"{path_prefix}_agent.pth"
        self.logger.info(f"Saving PPO model to {agent_path}")
        try:
            torch.save(self.agent.state_dict(), agent_path)
        except Exception as e:
            self.logger.error(f"Error saving PPO model: {e}", exc_info=True)
    
    def load_model(self, path_prefix):
        """Load model weights."""
        agent_path = f"{path_prefix}_agent.pth"
        self.logger.info(f"Loading PPO model from {agent_path}")
        try:
            if os.path.exists(agent_path):
                self.agent.load_state_dict(torch.load(agent_path, map_location=self.device))
                self.agent.eval()
                self.logger.info(f"Successfully loaded PPO model from {agent_path}")
            else:
                self.logger.error(f"Model file not found at {agent_path}")
        except Exception as e:
            self.logger.error(f"Error loading PPO model: {e}", exc_info=True)