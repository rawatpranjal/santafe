# traders/ppo_lstm_core.py
import logging
import os # Added for load/save model checks
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

# --- Combined Agent Class (MLP + LSTM) ---
class LSTMAgent(nn.Module):
    def __init__(self, state_dim, action_dim, nn_hidden_layers, lstm_hidden_size, lstm_num_layers):
        super().__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        # --- MLP Backbone ---
        mlp_layers = []
        input_dim = state_dim
        mlp_output_size = input_dim # Default if no MLP layers
        if nn_hidden_layers:
             for hidden_dim in nn_hidden_layers:
                 mlp_layers.append(layer_init(nn.Linear(input_dim, hidden_dim)))
                 mlp_layers.append(nn.Tanh())
                 input_dim = hidden_dim
             mlp_output_size = hidden_dim # Output size of MLP backbone
        self.network = nn.Sequential(*mlp_layers)
        # --- End MLP Backbone ---

        # --- LSTM Layer ---
        self.lstm = nn.LSTM(mlp_output_size, lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        # --- End LSTM Layer ---

        # --- Actor Head ---
        self.actor = nn.Sequential(
            # Removed optional extra layer for simplicity, can add back if needed
            # layer_init(nn.Linear(lstm_hidden_size, lstm_hidden_size)),
            # nn.Tanh(),
            layer_init(nn.Linear(lstm_hidden_size, action_dim), std=0.01)
        )
        # --- End Actor Head ---

        # --- Critic Head ---
        self.critic = nn.Sequential(
            # Removed optional extra layer
            # layer_init(nn.Linear(lstm_hidden_size, lstm_hidden_size)),
            # nn.Tanh(),
            layer_init(nn.Linear(lstm_hidden_size, 1), std=1.0)
        )
        # --- End Critic Head ---

    def get_initial_state(self, batch_size, device):
        """Returns initial dummy hidden state for LSTM."""
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        return (h0, c0)

    def forward(self, x, lstm_state):
        """
        Main forward pass. Handles sequences.
        x shape: (batch_size, seq_len, feature_dim)
        lstm_state: tuple (h0, c0), each shape (num_layers, batch_size, hidden_size)
        Returns: logits (B, T, A), value (B, T), new_lstm_state
        """
        batch_size, seq_len, feature_dim = x.shape
        x_flat = x.view(batch_size * seq_len, feature_dim)
        mlp_out_flat = self.network(x_flat)
        mlp_out = mlp_out_flat.view(batch_size, seq_len, -1)

        lstm_out, new_lstm_state = self.lstm(mlp_out, lstm_state)

        lstm_out_flat = lstm_out.reshape(batch_size * seq_len, -1)
        logits_flat = self.actor(lstm_out_flat)
        values_flat = self.critic(lstm_out_flat)

        logits = logits_flat.view(batch_size, seq_len, -1)
        values = values_flat.view(batch_size, seq_len)

        return logits, values, new_lstm_state

    def get_action_value_and_state(self, x_step, lstm_state):
        """
        Used for single-step action selection during environment interaction.
        x_step shape: (1, 1, feature_dim) - Batch=1, Seq_len=1
        lstm_state: tuple (h0, c0), each shape (num_layers, 1, hidden_size)
        Returns: action, log_prob, entropy, value (all scalar tensors), new_lstm_state
        """
        logits_seq, value_seq, new_lstm_state = self.forward(x_step, lstm_state)

        logits = logits_seq[:, -1, :] # Shape (1, action_dim)
        value = value_seq[:, -1]      # Shape (1,)

        probs = Categorical(logits=logits)
        action = probs.sample() # Shape (1,)

        log_prob = probs.log_prob(action) # Shape (1,)
        entropy = probs.entropy() # Shape (1,)

        # Squeeze unnecessary batch dimension for return values (but not state)
        return action.squeeze(0), log_prob.squeeze(0), entropy.squeeze(0), value.squeeze(0), new_lstm_state

    def get_value(self, x_step, lstm_state):
         """ Get value estimate for a single step. x_step shape (1, 1, feature_dim) """
         _, value_seq, _ = self.forward(x_step, lstm_state)
         return value_seq[:, -1].squeeze(0) # Return value for the single step

# --- PPO+LSTM Update Logic ---
class PPOLogicLSTM:
    def __init__(self, state_dim, action_dim, config, seed):
        self.logger = logging.getLogger('rl_agent_logic.ppo_lstm')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        params = config.get('rl_params', {})

        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"PPO-LSTM using device: {self.device}")

        # Hyperparameters
        self.gamma = float(params.get('gamma', 0.99))
        self.gae_lambda = float(params.get('gae_lambda', 0.95))
        self.clip_coef = float(params.get('clip_epsilon', 0.2))
        self.ent_coef_initial = float(params.get('entropy_coef', 0.01))
        self.ent_coef = self.ent_coef_initial  # Current entropy coefficient
        self.vf_coef = float(params.get('vf_coef', 0.5))
        self.learning_rate_initial = float(params.get('learning_rate', params.get('lr', 3e-4)))
        self.learning_rate = self.learning_rate_initial  # Current learning rate
        self.update_epochs = int(params.get('update_epochs', params.get('n_epochs', 4)))
        self.norm_adv = params.get('norm_adv', True)
        self.clip_vloss = params.get('clip_vloss', True)
        self.max_grad_norm = float(params.get('max_grad_norm', 0.5))
        self.minibatch_size = int(params.get('minibatch_size', params.get('batch_size', 0))) # 0 means use full trajectory
        
        # New optimization parameters
        self.target_kl = float(params.get('target_kl', 0.015))  # Target KL for early stopping
        self.use_lr_annealing = params.get('use_lr_annealing', True)
        self.use_entropy_annealing = params.get('use_entropy_annealing', True)
        self.use_reward_scaling = params.get('use_reward_scaling', True)
        self.total_training_rounds = config.get('num_training_rounds', config.get('num_rounds', 1000))
        self.current_round = 0

        # Network Params
        nn_hidden_layers = params.get('nn_hidden_layers', [64]) # Default to one layer if unspecified
        lstm_hidden_size = params.get('lstm_hidden_size', 128)
        lstm_num_layers = params.get('lstm_num_layers', 1)

        self.agent = LSTMAgent(
            state_dim, action_dim, nn_hidden_layers, lstm_hidden_size, lstm_num_layers
        ).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)
        
        # Initialize reward normalization
        self.reward_rms = RunningMeanStd() if self.use_reward_scaling else None

        self.trajectory_buffer = []
        self.is_training = True
        self.last_train_stats = {}
        # current_lstm_state is managed externally by PPOTrader

    def set_mode(self, training=True):
        self.is_training = training
        if training: self.agent.train()
        else: self.agent.eval()
        self.logger.debug(f"PPO-LSTM Logic mode set to: {'Training' if training else 'Evaluation'}")

    def store_transition(self, state, action, reward, done, log_prob, value):
        """ Store flat transition data. """
        if not self.is_training: return
        state_np = np.array(state, dtype=np.float32) if state is not None else np.zeros(self.state_dim, dtype=np.float32)
        if state is None: self.logger.warning("Storing transition with None state!")
        
        # Normalize reward if enabled
        if self.use_reward_scaling and self.reward_rms is not None:
            # Update running statistics with raw reward
            self.reward_rms.update(np.array([reward]))
            # Normalize the reward
            reward = self.reward_rms.normalize(np.array([reward]))[0]
        
        # Convert tensors to numpy/python types for storage if needed
        action_item = action.item() if isinstance(action, torch.Tensor) else action
        log_prob_item = log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob
        value_item = value.item() if isinstance(value, torch.Tensor) else value
        self.trajectory_buffer.append((state_np, action_item, reward, done, log_prob_item, value_item))


    def clear_buffer(self):
        """Explicitly clears the trajectory buffer to free memory."""
        self.trajectory_buffer = []
        self.logger.debug("Trajectory buffer explicitly cleared.")
    
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
                self.logger.debug(f"Learning rate updated to {self.learning_rate:.6f} (round {current_round}/{self.total_training_rounds})")
            
            # Linear annealing for entropy coefficient
            if self.use_entropy_annealing:
                self.ent_coef = self.ent_coef_initial * (1.0 - progress)
                self.logger.debug(f"Entropy coefficient updated to {self.ent_coef:.6f}")
    
    def learn(self, initial_lstm_state, last_step_next_state):
        """ Perform PPO+LSTM update based on the stored trajectory buffer. """
        if not self.is_training or not self.trajectory_buffer:
            self.trajectory_buffer = []
            self.last_train_stats = {}
            return

        traj_len = len(self.trajectory_buffer)
        if traj_len < 2:
             self.logger.debug(f"Trajectory too short ({traj_len} steps). Skipping update.")
             self.trajectory_buffer = []; self.last_train_stats = {}; return

        self.logger.debug(f"Performing PPO-LSTM learning update (Trajectory size: {traj_len})")

        # 1. Prepare Data from Buffer
        states, actions, rewards, dones, old_log_probs, values = map(np.array, zip(*self.trajectory_buffer))
        states = torch.tensor(states, dtype=torch.float32).unsqueeze(0).to(self.device) # (1, T, state_dim)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(0).to(self.device)     # (1, T)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(0).to(self.device) # (1, T)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(0).to(self.device)     # (1, T)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).unsqueeze(0).to(self.device) # (1, T)
        values = torch.tensor(values, dtype=torch.float32).unsqueeze(0).to(self.device)   # (1, T)
        # Ensure initial_lstm_state is correctly formatted for batch_size=1
        initial_h, initial_c = initial_lstm_state
        if initial_h.shape[1] != 1:
            initial_lstm_state = (initial_h[:,:1,:].contiguous(), initial_c[:,:1,:].contiguous())


        # 2. Calculate Advantages (GAE)
        with torch.no_grad():
            # Need value of the state *after* the last step for GAE bootstrap
            last_state_tensor = torch.tensor(np.array(last_step_next_state), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device) # (1, 1, state_dim)
            # Need the LSTM state *after* processing the last step (should be stored externally)
            # Assuming external logic provides the correct final LSTM state
            # For now, let's recompute it - less efficient but works for single trajectory
            _, _, final_lstm_state_recomputed = self.agent.forward(states, initial_lstm_state)
            next_value = self.agent.get_value(last_state_tensor, final_lstm_state_recomputed)

            advantages = torch.zeros_like(rewards).to(self.device) # (1, T)
            lastgaelam = 0
            for t in reversed(range(traj_len)):
                if t == traj_len - 1:
                    # Check the 'done' flag of the *last transition* stored
                    final_done_flag = dones[:, t]
                    nextnonterminal = 1.0 - final_done_flag
                    nextvalues = next_value # Value of the state *after* the last action
                else:
                    nextnonterminal = 1.0 - dones[:, t + 1] # Use done flag of state s_{t+1}
                    nextvalues = values[:, t + 1] # Use stored value of s_{t+1}
                delta = rewards[:, t] + self.gamma * nextvalues * nextnonterminal - values[:, t]
                advantages[:, t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values # (1, T)

        # 3. Perform PPO Updates
        batch_size = 1 # We process one full trajectory
        sequence_len = traj_len
        # If minibatch_size > 0, calculate steps_per_minibatch and adapt loop
        steps_per_minibatch = self.minibatch_size if self.minibatch_size > 0 else sequence_len
        num_minibatches = sequence_len // steps_per_minibatch

        inds = np.arange(sequence_len)
        clipfracs = []; all_pg_loss, all_v_loss, all_entropy_loss, all_approx_kl = [], [], [], []

        self.agent.train()
        for epoch in range(self.update_epochs):
             # No shuffling needed if processing full sequence as one minibatch
             # np.random.shuffle(inds) # Shuffling steps breaks sequence for LSTM
             
             # Process the whole sequence (batch_size=1, seq_len=T)
             mb_obs_seq = states
             mb_actions_seq = actions.view(-1) # Flatten actions (T,)
             mb_logprobs_old = old_log_probs.view(-1) # (T,)
             mb_advantages = advantages.view(-1) # (T,)
             mb_returns = returns.view(-1) # (T,)
             mb_values_old = values.view(-1) # (T,)

             # --- Forward pass for the whole sequence ---
             logits_seq, values_seq, _ = self.agent(mb_obs_seq, initial_lstm_state)
             # logits (1, T, A), values (1, T)

             new_values_flat = values_seq.view(-1) # (T,)
             probs = Categorical(logits=logits_seq.view(-1, self.action_dim)) # (T, A)

             new_logprobs_flat = probs.log_prob(mb_actions_seq) # (T,)
             entropy_flat = probs.entropy() # (T,) - Entropy at each step

             logratio = new_logprobs_flat - mb_logprobs_old
             ratio = logratio.exp()

             with torch.no_grad():
                 old_approx_kl = (-logratio).mean()
                 approx_kl = ((ratio - 1) - logratio).mean()
                 clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

             # Normalize advantages (full sequence advantages)
             if self.norm_adv:
                 adv_mean = mb_advantages.mean(); adv_std = mb_advantages.std()
                 mb_advantages = (mb_advantages - adv_mean) / (adv_std + 1e-8) if adv_std > 1e-8 else torch.zeros_like(mb_advantages)

             # Policy loss
             pg_loss1 = -mb_advantages * ratio
             pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
             pg_loss = torch.max(pg_loss1, pg_loss2).mean()

             # Value loss
             newvalue_mb = new_values_flat
             if self.clip_vloss:
                 v_loss_unclipped = (newvalue_mb - mb_returns) ** 2
                 v_clipped = mb_values_old + torch.clamp(
                     newvalue_mb - mb_values_old, -self.clip_coef, self.clip_coef,
                 )
                 v_loss_clipped = (v_clipped - mb_returns) ** 2
                 v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                 v_loss = 0.5 * v_loss_max.mean()
             else:
                 v_loss = 0.5 * ((newvalue_mb - mb_returns) ** 2).mean()

             entropy_loss = entropy_flat.mean()
             loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

             self.optimizer.zero_grad()
             loss.backward()
             nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
             self.optimizer.step()

             all_pg_loss.append(pg_loss.item()); all_v_loss.append(v_loss.item())
             all_entropy_loss.append(entropy_loss.item()); all_approx_kl.append(approx_kl.item())

             # Target KL early stopping
             if self.target_kl is not None and approx_kl > self.target_kl:
                 self.logger.debug(f"Early stopping at epoch {epoch} due to KL divergence {approx_kl:.4f} > {self.target_kl}")
                 break

        self.trajectory_buffer = [] # Clear buffer

        # Store stats
        self.last_train_stats = {
            'avg_policy_loss': np.mean(all_pg_loss) if all_pg_loss else np.nan,
            'avg_value_loss': np.mean(all_v_loss) if all_v_loss else np.nan,
            'avg_entropy': np.mean(all_entropy_loss) if all_entropy_loss else np.nan,
            'avg_approx_kl': np.mean(all_approx_kl) if all_approx_kl else np.nan,
            'avg_clip_frac': np.mean(clipfracs) if clipfracs else np.nan
        }
        self.logger.debug(f"PPO-LSTM Update Stats: PL={self.last_train_stats['avg_policy_loss']:.4f}, VL={self.last_train_stats['avg_value_loss']:.4f}")

    def save_model(self, path_prefix):
        agent_path = f"{path_prefix}_agent.pth"
        self.logger.info(f"Saving PPO-LSTM Agent model to {agent_path}")
        try: torch.save(self.agent.state_dict(), agent_path)
        except Exception as e: self.logger.error(f"Error saving PPO-LSTM Agent model: {e}", exc_info=True)

    def load_model(self, path_prefix):
        agent_path = f"{path_prefix}_agent.pth"
        self.logger.info(f"Loading PPO-LSTM Agent model from {agent_path}")
        try:
            if os.path.exists(agent_path):
                self.agent.load_state_dict(torch.load(agent_path, map_location=self.device))
                self.agent.eval()
                self.logger.info(f"Successfully loaded PPO-LSTM Agent model from {agent_path}")
            else: self.logger.error(f"Agent model file not found at {agent_path}")
        except Exception as e: self.logger.error(f"Error loading PPO-LSTM Agent model: {e}", exc_info=True)