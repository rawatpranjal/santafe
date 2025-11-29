#!/usr/bin/env python3
"""Debug script to test LLM API integration with full logging."""

import logging
import sys

# Setup DEBUG logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.INFO)

from traders.llm.gpt_agent import GPTAgent
from traders.llm.prompt_builder import PromptBuilder

def main():
    print("=" * 60)
    print("LLM API DEBUG TEST")
    print("=" * 60)

    # Create test agent
    agent = GPTAgent(
        player_id=1,
        is_buyer=True,
        num_tokens=3,
        valuations=[80, 70, 60],
        price_min=0,
        price_max=100,
        model='gpt-3.5-turbo',
        use_cache=False,  # Disable cache for this test
        prompt_style='original'
    )

    # Build test prompt
    pb = PromptBuilder()
    prompt = pb.build_bid_ask_prompt(
        is_buyer=True,
        valuation=80,
        tokens_remaining=3,
        tokens_total=3,
        time_step=10,
        max_time=100,
        best_bid=50,
        best_ask=90,
        spread=40,
        price_min=0,
        price_max=100
    )

    print("\n=== PROMPT ===")
    print(prompt)
    print()

    # Make LLM call
    print("=== CALLING LLM ===")
    action = agent._generate_bid_ask_action(prompt, valuation=80, best_bid=50, best_ask=90)

    print("\n=== RESULT ===")
    print(f"Action: {action.action}")
    print(f"Price: {action.price}")
    print(f"Reasoning: {action.reasoning}")

    # Verify action is valid
    if action.action == "bid":
        if action.price and 51 <= action.price <= 79:
            print("\n✓ VALID: Bid is in range [51, 79]")
        else:
            print(f"\n✗ INVALID: Bid {action.price} not in valid range [51, 79]")
    elif action.action == "pass":
        print("\n- PASS: Agent chose to pass")
    else:
        print(f"\n✗ WRONG ACTION TYPE: Expected 'bid' or 'pass', got '{action.action}'")

if __name__ == "__main__":
    main()
