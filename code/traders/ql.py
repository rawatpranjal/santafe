# traders/ql.py
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import os

from .base import BaseTrader

# --- Experience and QNetwork remain the same ---
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class QNetwork(nn.Module):
    """Simple Feedforward Neural Network for Q-value approximation."""
    def __init__(self, state_dim, action_dim, hidden_layers):
        super().__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights) # Initialize weights

    def _init_weights(self, module):
        """Initialize weights using Xavier uniform for linear layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01) # Small bias initialization

    def forward(self, x):
        if not isinstance(x, torch.Tensor): x = torch.FloatTensor(x)
        elif x.dtype != torch.float32: x = x.float()
        return self.network(x)

# --- DQNLogic remains the same ---
class DQNLogic:
    """Encapsulates the core DQN algorithm components and logic."""
    def __init__(self, state_dim, action_dim, config, seed):
        self.logger = logging.getLogger('rl_agent_logic.dqn')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        params = config.get('rl_params', {})

        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"DQN using device: {self.device}")

        # Hyperparameters
        self.buffer_size = int(params.get('buffer_size', 10000))
        self.batch_size = int(params.get('batch_size', 64))
        self.gamma = float(params.get('gamma', 0.99))
        self.lr = float(params.get('learning_rate', 0.0005))
        self.target_update_freq = int(params.get('target_update_freq', 100))
        self.epsilon_start = float(params.get('epsilon_start', 1.0))
        self.epsilon_end = float(params.get('epsilon_end', 0.05))
        self.epsilon_decay_steps = int(params.get('epsilon_decay_steps', 10000))
        self.optimizer_name = params.get('optimizer', 'Adam')
        self.grad_clip = params.get('gradient_clip_value')

        if self.epsilon_decay_steps <= 0: self.epsilon_decay_rate = 0
        else: self.epsilon_decay_rate = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
        self.current_epsilon = self.epsilon_start
        self.total_steps_taken = 0

        hidden_layers = params.get('nn_hidden_layers', [128, 64])
        self.q_network = QNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        if self.optimizer_name.lower() == 'adam': self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        elif self.optimizer_name.lower() == 'rmsprop': self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=self.lr)
        else: self.logger.warning(f"Unknown optimizer '{self.optimizer_name}', defaulting to Adam."); self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        self.memory = deque(maxlen=self.buffer_size)
        self.is_training = True

    def set_mode(self, training=True):
        self.is_training = training
        if training: self.q_network.train()
        else: self.q_network.eval(); self.current_epsilon = 0.0
        self.logger.info(f"DQN Logic mode set to: {'Training' if training else 'Evaluation'}")

    def _decay_epsilon(self):
        if self.is_training: self.current_epsilon = max(self.epsilon_end, self.epsilon_start - self.epsilon_decay_rate * self.total_steps_taken)

    def choose_action(self, state):
        self.total_steps_taken += 1
        if self.is_training: self._decay_epsilon()

        if self.is_training and random.random() < self.current_epsilon:
            action_idx = random.randrange(self.action_dim)
            self.logger.debug(f"Exploring: Random action {action_idx} (Eps={self.current_epsilon:.4f})")
            return action_idx
        else:
            with torch.no_grad():
                if not isinstance(state, np.ndarray): state = np.array(state)
                if state.dtype != np.float32: state = state.astype(np.float32)
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
                self.q_network.eval()
                q_values = self.q_network(state_tensor)
                if self.is_training: self.q_network.train()
                action_idx = q_values.argmax().item()
                if self.logger.isEnabledFor(logging.DEBUG):
                     q_vals_list = q_values.cpu().numpy().flatten().tolist()
                     q_vals_str = ", ".join([f"{q:.2f}" for q in q_vals_list])
                     self.logger.debug(f"Exploiting: Chose action {action_idx} (Q-vals: [{q_vals_str}])")
            return action_idx

    def store_transition(self, state, action, reward, next_state, done):
        if not self.is_training: return
        state_np = np.array(state).astype(np.float32) if state is not None else None
        next_state_np = np.array(next_state).astype(np.float32) if next_state is not None else None
        if state_np is None: self.logger.warning("Attempted to store transition with None state. Skipping."); return
        experience = Experience(state_np, action, reward, next_state_np, done)
        self.memory.append(experience)
        self.logger.debug(f"Stored transition: A={action}, R={reward:.2f}, Done={done}, BufferSize={len(self.memory)}")

    def learn(self):
        if not self.is_training or len(self.memory) < self.batch_size: return
        self.logger.debug(f"Performing learning update (Buffer size: {len(self.memory)}, BatchSize: {self.batch_size})")
        experiences = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)
        non_final_mask = torch.tensor(tuple(e.next_state is not None for e in experiences), device=self.device, dtype=torch.bool)
        non_final_next_states = None
        if torch.sum(non_final_mask) > 0: non_final_next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e.next_state is not None])).float().to(self.device)
        self.q_network.train()
        current_q_values = self.q_network(states).gather(1, actions)
        next_q_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_next_states is not None:
             with torch.no_grad(): next_q_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values.unsqueeze(1) * (1 - dones))
        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad(); loss.backward()
        if self.grad_clip: torch.nn.utils.clip_grad_value_(self.q_network.parameters(), self.grad_clip); self.logger.debug(f"Gradient clip applied: {self.grad_clip}")
        self.optimizer.step()
        if self.total_steps_taken % self.target_update_freq == 0: self.logger.info(f"Updating target network at step {self.total_steps_taken}"); self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, path):
        self.logger.info(f"Saving Q-network model to {path}");
        try: torch.save(self.q_network.state_dict(), path)
        except Exception as e: self.logger.error(f"Error saving model to {path}: {e}", exc_info=True)

    def load_model(self, path):
         self.logger.info(f"Loading Q-network model from {path}")
         try:
             if os.path.exists(path): self.q_network.load_state_dict(torch.load(path, map_location=self.device)); self.target_network.load_state_dict(self.q_network.state_dict()); self.q_network.eval(); self.target_network.eval(); self.logger.info(f"Successfully loaded model from {path}")
             else: self.logger.error(f"Model file not found at {path}")
         except Exception as e: self.logger.error(f"Error loading model from {path}: {e}", exc_info=True)


# --- QLTrader Class ---
class QLTrader(BaseTrader):
    """
    Trader using DQN to choose bid/ask prices from a predefined grid.
    Automatically accepts profitable trades.
    """
    def __init__(self, name, is_buyer, private_values, rl_config, **kwargs):
        strategy_name = "dqn_pricegrid" # New strategy name
        super().__init__(name, is_buyer, private_values, strategy=strategy_name)
        self.logger = logging.getLogger(f'trader.{self.name}')
        rl_log_level_str = rl_config.get('log_level_rl', 'WARNING').upper()
        rl_log_level = getattr(logging, rl_log_level_str, logging.WARNING)
        self.logger.setLevel(rl_log_level)

        self.rl_config = rl_config
        self.episode_rewards = []

        self.state_dim = self._define_state_dim()
        # Define action space based on price grid
        # Example: 11 actions corresponding to 11 price levels
        # Action 0: price = val/cost - 10% range
        # Action 5: price = val/cost
        # Action 10: price = val/cost + 10% range
        # Add Action 11: Pass (do not submit quote)
        self.num_price_actions = kwargs.get('num_price_actions', 11) # e.g., 11 price points
        self.action_dim = self.num_price_actions + 1 # Add 1 for the "pass" action
        self.price_range_pct = kwargs.get('price_range_pct', 0.10) # e.g., +/- 10% around val/cost

        base_seed = rl_config.get("rng_seed_rl", random.randint(0, 1e6))
        agent_seed = base_seed + self.id_numeric
        self.agent = DQNLogic(self.state_dim, self.action_dim, rl_config, agent_seed)

        load_path = rl_config.get("load_rl_model_path")
        if load_path: self.agent.load_model(load_path)

        self._last_state = None
        self._last_action_idx = None

    def _define_state_dim(self):
        # Same state representation as before
        return 10

    def _map_action_to_price(self, action_idx):
        """Maps the chosen action index to a specific bid/ask price."""
        if action_idx == self.action_dim - 1: # Last action is "pass"
            return None

        val_cost = self.get_next_value_cost()
        if val_cost is None: return None # Cannot determine price without value/cost

        # Calculate price based on action index relative to value/cost
        # Action 0 = aggressive, Action num_price_actions-1 = passive
        # Example: 11 price actions (indices 0 to 10)
        # Midpoint action index: (num_price_actions - 1) / 2 (e.g., 5 for 11 actions)
        mid_action_idx = (self.num_price_actions - 1) / 2.0
        # Calculate relative position from midpoint [-0.5, +0.5]
        relative_pos = (action_idx - mid_action_idx) / max(1, mid_action_idx * 2) # Normalize to roughly -0.5 to +0.5

        # Calculate price range based on percentage
        price_span = (self.max_price - self.min_price) * self.price_range_pct
        # Price delta relative to val_cost. More negative relative_pos means more aggressive bid/ask.
        # Buyer wants lower price -> aggressive means closer to val_cost (less negative delta)
        # Seller wants higher price -> aggressive means closer to val_cost (less positive delta)
        if self.is_buyer:
            # Aggressive (low action_idx, negative relative_pos) -> small negative delta (price near value)
            # Passive (high action_idx, positive relative_pos) -> large negative delta (price much lower than value)
            price_delta = -price_span * (0.5 + relative_pos) # delta is negative, maps [-1, 0]
            target_price = val_cost + price_delta # Add negative delta
        else: # Seller
            # Aggressive (low action_idx, negative relative_pos) -> small positive delta (price near cost)
            # Passive (high action_idx, positive relative_pos) -> large positive delta (price much higher than cost)
            price_delta = price_span * (0.5 + relative_pos) # delta is positive, maps [0, +1]
            target_price = val_cost + price_delta # Add positive delta

        final_price = max(self.min_price, min(self.max_price, int(round(target_price))))
        self.logger.debug(f"Action {action_idx} -> Price Grid -> TargetPrice={target_price:.2f} -> FinalPrice={final_price}")
        return final_price


    def _get_state(self, current_bid_info, current_ask_info, phibid, phiask):
        # Identical state calculation as before
        state = np.zeros(self.state_dim, dtype=np.float32)
        price_range = max(self.max_price - self.min_price, 1)
        state[0] = (self.total_steps_in_period - self.current_step) / self.total_steps_in_period if self.total_steps_in_period > 0 else 0
        state[1] = self.tokens_left / self.max_tokens if self.max_tokens > 0 else 0
        num_periods = self.rl_config.get('num_periods', 1)
        state[2] = self.current_period / max(1, num_periods -1) if num_periods > 1 else 0
        val_cost = self.get_next_value_cost(); state[3] = ((val_cost - self.min_price) / price_range) if val_cost is not None else -1
        state[4] = ((current_bid_info['price'] - self.min_price) / price_range) if current_bid_info else -1
        state[5] = ((current_ask_info['price'] - self.min_price) / price_range) if current_ask_info else -1
        state[6] = (phibid - self.min_price) / price_range
        state[7] = (phiask - self.min_price) / price_range
        state[8] = 1.0 if current_bid_info and current_bid_info['agent'] == self else 0.0
        state[9] = 1.0 if current_ask_info and current_ask_info['agent'] == self else 0.0
        state = np.clip(state, -1.0, 1.0); return state

    def set_mode(self, training=True): self.agent.set_mode(training)

    def observe_reward(self, last_state, action_idx, reward, next_state, done):
         reward = float(reward)
         if last_state is not None and action_idx is not None:
             actual_next_state = None if done else next_state
             if not done and actual_next_state is None: self.logger.warning(f"R{self.current_round}P{self.current_period}S{self.current_step}: Received non-done transition with None next_state. Storing None.")
             self.agent.store_transition(last_state, action_idx, reward, actual_next_state, done)
             self.episode_rewards.append(reward)
             if self.agent.is_training: self.agent.learn()
         elif done:
             final_avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
             self.logger.info(f"End of Round/Episode {self.current_round}. Total reward: {sum(self.episode_rewards):.2f}, Avg reward: {final_avg_reward:.3f}, Final Epsilon: {self.agent.current_epsilon:.4f}")
             self._last_state = None; self._last_action_idx = None

    def get_last_episode_stats(self):
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        total_reward = sum(self.episode_rewards)
        epsilon = self.agent.current_epsilon
        stats = {'avg_reward': avg_reward, 'total_reward': total_reward, 'epsilon': epsilon}
        self.episode_rewards = []
        return stats

    # --- Implement BaseTrader Strategy Methods ---
    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Use DQN to choose a price action, then map it to a price."""
        if not self.can_trade():
            self._last_state = None; self._last_action_idx = None
            return None

        state = self._get_state(current_bid_info, current_ask_info, phibid, phiask)
        action_idx = self.agent.choose_action(state)

        # Store state/action for reward assignment
        self._last_state = state
        self._last_action_idx = action_idx

        # Map the chosen action index to a price (or None if "pass")
        final_price = self._map_action_to_price(action_idx)

        if final_price is not None:
            self.logger.debug(f"DQN chose Action {action_idx} -> Proposing {'Bid' if self.is_buyer else 'Ask'} {final_price}")
        else:
            self.logger.debug(f"DQN chose Action {action_idx} ('pass') -> Not submitting Bid/Ask")

        return final_price # Return price or None

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Automatically accept if profitable. DQN is not used for accept/reject."""
        if not self.can_trade() or current_offer_price is None: return False
        if not self.is_buyer: return False # Only buyers accept asks

        value = self.get_next_value_cost()
        is_profitable = value is not None and current_offer_price <= value

        if is_profitable:
            self.logger.debug(f"Auto-accepting BUY at {current_offer_price} (Value={value})")
            # We need to store a state/action pair even for auto-accept for learning
            # Use the state *before* this decision, and a placeholder action (e.g., -1?)
            # Or, more simply, don't learn from auto-accepts. Let's do that for simplicity.
            self._last_state = None
            self._last_action_idx = None
        else:
             self.logger.debug(f"Auto-rejecting BUY at {current_offer_price} (Value={value})")

        return is_profitable

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Automatically accept if profitable. DQN is not used for accept/reject."""
        if not self.can_trade() or current_bid_price is None: return False
        if self.is_buyer: return False # Only sellers accept bids

        cost = self.get_next_value_cost()
        is_profitable = cost is not None and current_bid_price >= cost

        if is_profitable:
            self.logger.debug(f"Auto-accepting SELL at {current_bid_price} (Cost={cost})")
            self._last_state = None
            self._last_action_idx = None
        else:
             self.logger.debug(f"Auto-rejecting SELL at {current_bid_price} (Cost={cost})")

        return is_profitable