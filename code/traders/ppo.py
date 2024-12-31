import numpy as np
from .base import BaseTrader

class PPOAgent:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.memory = []
        self.policy_params = np.random.rand(2)
    def select_action(self, state):
        return self.policy_params[0]*state + self.policy_params[1]
    def store_transition(self, state, action, reward):
        self.memory.append((state, action, reward))
    def update_policy(self):
        if not self.memory:
            return
        avg_r = np.mean([r for (_,_,r) in self.memory])
        self.policy_params += self.learning_rate*avg_r*np.random.randn(2)
        self.memory = []

class PPOBuyer(BaseTrader):
    def __init__(self, name, is_buyer, private_values, agent=None):
        super().__init__(name, is_buyer, private_values, strategy="ppo-buyer")
        self.agent = agent if agent else PPOAgent()
    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        if not self.can_trade(): return None
        val = self.next_token_value()
        if val is None: return None
        best_bid = c_bid if c_bid else 0.0
        best_ask = c_ask if c_ask else 1.0
        state = np.array([best_bid, best_ask, val])
        raw_action = self.agent.select_action(state.mean())
        action_price = np.clip(raw_action, 0.0, val)
        return (action_price, self)
    def decide_to_buy(self, best_ask):
        val = self.next_token_value()
        return (val is not None) and (best_ask is not None) and (val >= best_ask)
    def decide_to_sell(self, best_bid):
        return False
    def update_after_trade(self, reward):
        self.agent.store_transition(0.0, 0.0, reward)

class PPOSeller(BaseTrader):
    def __init__(self, name, is_buyer, private_values, agent=None):
        super().__init__(name, is_buyer, private_values, strategy="ppo-seller")
        self.agent = agent if agent else PPOAgent()
    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        if not self.can_trade(): return None
        cost = self.next_token_value()
        if cost is None: return None
        best_bid = c_bid if c_bid else 0.0
        best_ask = c_ask if c_ask else 1.0
        state = np.array([best_bid, best_ask, cost])
        raw_action = self.agent.select_action(state.mean())
        action_price = np.clip(raw_action, cost, 1.0)
        return (action_price, self)
    def decide_to_buy(self, best_ask):
        return False
    def decide_to_sell(self, best_bid):
        cost = self.next_token_value()
        return (cost is not None) and (best_bid is not None) and (best_bid >= cost)
    def update_after_trade(self, reward):
        self.agent.store_transition(0.0, 0.0, reward)
