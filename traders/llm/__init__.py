"""
LLM-based trading agents for the Santa Fe Double Auction.

This package implements Large Language Model agents that participate
in the AURORA double auction protocol using natural language reasoning.

Agents:
    PlaceholderLLM: Simple rule-based agent for testing infrastructure
    GPTAgent: OpenAI GPT-4/3.5 implementation (Phase B)
"""

from traders.llm.placeholder_agent import PlaceholderLLM
from traders.llm.gpt_agent import GPTAgent

__all__ = ["PlaceholderLLM", "GPTAgent"]
