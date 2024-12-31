# traders/ppo.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .base import BaseTrader

###############################
# 1) Simple Actor-Critic Model
###############################
class PPOModel(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        # Actor: outputs a single logit => fraction = 1.5 * sigmoid(logit)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        # Critic: outputs scalar value
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Returns (logit, value).
        x: shape [batch_size, state_dim].
        """
        logit = self.actor(x)
        value = self.critic(x).squeeze(-1)  # shape [batch_size]
        return logit, value

###############################
# 2) PPO Agent
###############################
class PPOAgent:
    """
    "More Proper" PPO:
      - GAE advantage across entire round's transitions
      - Multiple epochs of mini-batch updates
      - Sigmoid bounding for action => fraction in [0..1.5]
      - Periodic print statements
    """
    def __init__(self,
                 state_dim=4,
                 lr=1e-4,
                 gamma=0.99,
                 lam=0.95,
                 clip_coef=0.2,
                 vf_coef=0.5,
                 ent_coef=0.0,
                 max_grad_norm=0.5,
                 num_update_epochs=5,
                 minibatch_size=32):
        self.model = PPOModel(state_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.num_update_epochs = num_update_epochs
        self.minibatch_size = minibatch_size

        # We'll gather transitions each round
        self.storage = []

        # Stats tracking
        self.round_count = 0

    def store(self, state, action, reward, value, logit, done):
        """
        Store one transition for GAE. We'll need next_value eventually, so we also store
        the model outputs (logit, value).
        """
        self.storage.append({
            "state": state,       # np array
            "action": action,     # float
            "reward": reward,     # float
            "value": value,       # float
            "logit": logit,       # float
            "done": done          # bool
        })

    def select_action(self, state_np):
        """
        state_np: shape (state_dim,). We'll do a single forward pass.
        Returns fraction, logit, value
        """
        s_t = torch.FloatTensor(state_np).unsqueeze(0)  # [1, state_dim]
        with torch.no_grad():
            logit, value = self.model(s_t)
        # fraction in [0..1.5]
        frac = 1.2 * torch.sigmoid(logit)
        return frac.item(), logit.item(), value.item()

    def finish_round(self):
        """
        Called at the end of each round to:
          - compute advantages using GAE
          - run multiple epochs of minibatch PPO updates
          - clear the storage
        """
        if not self.storage:
            return

        self.round_count += 1

        # 1) Convert to arrays/tensors
        states = []
        actions = []
        rewards = []
        values = []
        logits = []
        dones = []
        for step in self.storage:
            states.append(step["state"])
            actions.append(step["action"])
            rewards.append(step["reward"])
            values.append(step["value"])
            logits.append(step["logit"])
            dones.append(step["done"])

        # We'll need next_values to do GAE, so let's create an array shifted by 1
        next_values = values[1:] + [values[-1]]  # "bootstrap" from last
        # Also the "done" info
        # Convert to Tensors
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.FloatTensor(np.array(actions))
        values_t = torch.FloatTensor(np.array(values))
        logits_t = torch.FloatTensor(np.array(logits))
        rewards_t = torch.FloatTensor(np.array(rewards))
        dones_t = torch.FloatTensor(np.array(dones))
        next_values_t = torch.FloatTensor(np.array(next_values))

        # 2) GAE advantage
        advantages = []
        gae = 0.0
        for i in reversed(range(len(rewards))):
            mask = 1.0 - dones_t[i]
            delta = rewards_t[i] + self.gamma * next_values_t[i] * mask - values_t[i]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages.insert(0, gae)
        advantages_t = torch.FloatTensor(advantages)

        # "target" => value + advantage
        returns_t = values_t + advantages_t

        # 3) Flatten data for mini-batch PPO
        full_size = len(states)
        indices = np.arange(full_size)

        # We'll do multiple update epochs
        epoch_losses = []
        for _epoch in range(self.num_update_epochs):
            np.random.shuffle(indices)
            # mini-batch loop
            for start in range(0, full_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]

                mb_states = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_values = values_t[mb_idx]
                mb_logits_old = logits_t[mb_idx]
                mb_adv = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                # normalize adv (often helps)
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # forward pass
                new_logit, new_value = self.model(mb_states)
                # policy => fraction = 1.5 * sigmoid(new_logit)
                # approximate old "log probability" => ratio = exp(new - old)
                # This is not a perfect measure, but a stand-in for demonstration
                ratio = torch.exp(new_logit - mb_logits_old)

                # clipped surrogate
                obj1 = -mb_adv * ratio
                obj2 = -mb_adv * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                policy_loss = torch.mean(torch.max(obj1, obj2))

                # value loss
                if True:  # emulate "clip_vloss" if desired
                    v_clipped = mb_values + torch.clamp(
                        new_value - mb_values, -self.clip_coef, self.clip_coef
                    )
                    v_loss_unclipped = (new_value - mb_returns) ** 2
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.mean(torch.max(v_loss_unclipped, v_loss_clipped))
                else:
                    v_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()

                # optional entropy term if you want (simple measure)
                # not fully correct since we only have logits => we skip
                # ent = ???

                loss = policy_loss + self.vf_coef * v_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_adv = advantages_t.mean().item()
        avg_ret = returns_t.mean().item()

        # Print training info every N rounds
        if self.round_count % 20 == 0:
            print(f"[PPOAgent] Round {self.round_count}: loss={avg_loss:.4f} adv={avg_adv:.4f} ret={avg_ret:.4f}")

        # 4) Clear storage
        self.storage = []

###############################
# 3) PPO Buyer
###############################
class PPOBuyer(BaseTrader):
    """
    - state => (best_bid, best_ask, last_trade_price, my_valuation)
    - action => fraction = 1.5 * sigmoid(logit), final_bid= fraction * my_valuation
    - We store transitions each step. 
    - At round-end, do a PPO update (finish_round).
    """
    def __init__(self, name, is_buyer, private_values, agent=None):
        super().__init__(name, is_buyer, private_values, strategy="ppobuyer")
        # We'll define a 4-dim state for demonstration
        if agent is not None:
            self.agent = agent
        else:
            # create a separate agent if none is shared
            self.agent = PPOAgent(state_dim=4, lr=1e-4)

        self.last_trade_price = 0.0
        self.prev_state = None
        self.prev_logit = None
        self.prev_value = None

    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        if not self.can_trade():
            return None
        val = self.next_token_value()
        if val is None:
            return None

        # Build state
        best_bid = 0.0 if c_bid is None else c_bid
        best_ask = 1.0 if c_ask is None else c_ask
        s = np.array([best_bid, best_ask, self.last_trade_price, val], dtype=float)

        # select action => fraction, then final_bid
        frac, logit, v_est = self.agent.select_action(s)
        final_bid = frac * val
        final_bid = float(np.clip(final_bid, 0.0, 1.0))

        # Store "prev" so that we can record next step's reward
        self.prev_state = s
        self.prev_logit = logit
        self.prev_value = v_est

        return (final_bid, self)

    def decide_to_buy(self, best_ask):
        # If best_ask <= my valuation => accept trade
        val = self.next_token_value()
        return (val is not None) and (best_ask is not None) and (val >= best_ask)

    def decide_to_sell(self, best_bid):
        return False  # buyer only

    def update_after_trade(self, reward):
        """
        Called upon a trade. We'll store (state, action, reward, value, logit).
        We assume 'action' = fraction, but we only stored the logit => we can do fraction easily:
        fraction = 1.5 * sigmoid(logit).
        """
        if self.prev_state is None:
            return
        fraction = 1.5 * (1.0 / (1.0 + np.exp(-self.prev_logit)))  # same transform as forward
        done = False  # in a single round, we can define done if no tokens left

        self.agent.store(
            state=self.prev_state,
            action=fraction,
            reward=reward,
            value=self.prev_value,
            logit=self.prev_logit,
            done=done
        )

        # reset
        self.prev_state = None
        self.prev_logit = None
        self.prev_value = None

    def update_trade_stats(self, trade_price):
        self.last_trade_price = trade_price

    def reset_for_new_period(self, total_steps_in_period, round_idx, period_idx):
        """
        Overridden to also finalize PPO for the previous round
        (assuming 1 round => 1 call). 
        """
        # If you want the agent to update after each round, do it here:
        self.agent.finish_round()

        # Then do the parent's reset
        super().reset_for_new_period(total_steps_in_period, round_idx, period_idx)
        self.last_trade_price = 0.0
        self.prev_state = None
        self.prev_logit = None
        self.prev_value = None
