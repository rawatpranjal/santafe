"""
Agent Factory.
"""

from typing import List, Any
from traders.base import Agent
from traders.legacy.zi import ZI
from traders.legacy.zic import ZIC
from traders.legacy.kaplan import Kaplan
from traders.legacy.zip import ZIP
from traders.legacy.gd import GD
from traders.legacy.skeleton import Skeleton
from traders.legacy.zi2 import ZI2
from traders.legacy.lin import Lin
from traders.legacy.jacobson import Jacobson
from traders.legacy.perry import Perry
from traders.legacy.el import EL
from traders.legacy.markup import Markup
from traders.legacy.reservation_price import ReservationPrice
from traders.legacy.histogram_learner import HistogramLearner
from traders.legacy.rule_trader import RuleTrader
from traders.legacy.ringuette import Ringuette
from traders.legacy.truth_teller import TruthTeller
from traders.rl.ppo_agent import PPOAgent
from traders.llm.placeholder_agent import PlaceholderLLM
from traders.llm.gpt_agent import GPTAgent

def create_agent(
    agent_type: str,
    player_id: int,
    is_buyer: bool,
    num_tokens: int,
    valuations: list[int],
    seed: int | None = None,
    num_times: int = 100,
    num_buyers: int = 1,
    num_sellers: int = 1,
    price_min: int = 0,
    price_max: int = 1000,
    **kwargs: Any
) -> Agent:
    """
        Agent instance
    """
    # Filter out LLM-specific kwargs for legacy agents
    llm_kwargs = ["output_dir", "llm_output_dir"]
    legacy_kwargs = {k: v for k, v in kwargs.items() if k not in llm_kwargs}

    if agent_type == "ZI":
        return ZI(player_id, is_buyer, num_tokens, valuations, price_min=price_min, price_max=price_max, seed=seed)
    elif agent_type == "ZIC":
        return ZIC(player_id, is_buyer, num_tokens, valuations, price_min=price_min, price_max=price_max, **legacy_kwargs)
    elif agent_type == "Kaplan":
        return Kaplan(player_id, is_buyer, num_tokens, valuations, price_min=price_min, price_max=price_max, num_times=num_times, seed=seed, **legacy_kwargs)
    elif agent_type == "ZIP":
        return ZIP(player_id, is_buyer, num_tokens, valuations, price_min=price_min, price_max=price_max, **legacy_kwargs)
    elif agent_type == "GD":
        return GD(player_id, is_buyer, num_tokens, valuations, price_min=price_min, price_max=price_max, **legacy_kwargs)
    elif agent_type == "Skeleton":
        return Skeleton(player_id, is_buyer, num_tokens, valuations, price_min=price_min, price_max=price_max, **legacy_kwargs)
    elif agent_type == "ZI2":
        return ZI2(player_id, is_buyer, num_tokens, valuations, price_min=price_min, price_max=price_max, seed=seed, **legacy_kwargs)
    elif agent_type == "Lin":
        return Lin(player_id, is_buyer, num_tokens, valuations,
                  price_min=price_min, price_max=price_max,
                  num_buyers=num_buyers, num_sellers=num_sellers, num_times=num_times,
                  seed=seed, **legacy_kwargs)
    elif agent_type == "Jacobson":
        return Jacobson(player_id, is_buyer, num_tokens, valuations,
                       price_min=price_min, price_max=price_max,
                       num_times=num_times, seed=seed, **legacy_kwargs)
    elif agent_type == "Perry":
        return Perry(player_id, is_buyer, num_tokens, valuations,
                    price_min=price_min, price_max=price_max,
                    num_buyers=num_buyers, num_sellers=num_sellers, num_times=num_times,
                    seed=seed, **legacy_kwargs)
    elif agent_type == "EL":
        return EL(player_id, is_buyer, num_tokens, valuations,
                 price_min=price_min, price_max=price_max,
                 num_times=num_times, seed=seed, **legacy_kwargs)
    elif agent_type == "Markup":
        return Markup(player_id, is_buyer, num_tokens, valuations,
                     price_min=price_min, price_max=price_max,
                     seed=seed, **legacy_kwargs)
    elif agent_type == "ReservationPrice":
        return ReservationPrice(player_id, is_buyer, num_tokens, valuations,
                               price_min=price_min, price_max=price_max,
                               num_times=num_times, seed=seed, **legacy_kwargs)
    elif agent_type == "HistogramLearner":
        return HistogramLearner(player_id, is_buyer, num_tokens, valuations,
                               price_min=price_min, price_max=price_max,
                               seed=seed, **legacy_kwargs)
    elif agent_type == "RuleTrader":
        return RuleTrader(player_id, is_buyer, num_tokens, valuations,
                         price_min=price_min, price_max=price_max,
                         seed=seed, **legacy_kwargs)
    elif agent_type == "Ringuette":
        return Ringuette(player_id, is_buyer, num_tokens, valuations,
                        price_min=price_min, price_max=price_max,
                        seed=seed, **legacy_kwargs)
    elif agent_type == "TruthTeller":
        return TruthTeller(player_id, is_buyer, num_tokens, valuations,
                          price_min=price_min, price_max=price_max,
                          seed=seed, **legacy_kwargs)
    elif agent_type == "PPO":
        model_path = kwargs.get("model_path")
        if not model_path:
            raise ValueError("PPO agent requires 'model_path' in kwargs")
        return PPOAgent(player_id, is_buyer, num_tokens, valuations,
                       model_path=model_path,
                       max_price=price_max,
                       max_steps=num_times)
    elif agent_type == "PlaceholderLLM":
        return PlaceholderLLM(player_id, is_buyer, num_tokens, valuations,
                             price_min=price_min, price_max=price_max,
                             seed=seed, **kwargs)
    elif agent_type in ["GPT4", "GPT3.5", "GPT4-mini", "Groq-Llama", "Groq-Llama-70b"]:
        # Map agent type to model name
        model_map = {
            "GPT4": "gpt-4o",
            "GPT4-mini": "gpt-4o-mini",
            "GPT3.5": "gpt-3.5-turbo",
            "Groq-Llama": "groq/llama-3.3-70b-versatile",
            "Groq-Llama-70b": "groq/llama-3.3-70b-versatile"
        }
        model = model_map.get(agent_type, "gpt-4o-mini")
        return GPTAgent(player_id, is_buyer, num_tokens, valuations,
                       price_min=price_min, price_max=price_max,
                       num_times=num_times, model=model, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
