"""
GPT-based trading agent using LiteLLM.

Uses OpenAI GPT models (GPT-4, GPT-3.5) to make trading decisions
via natural language reasoning.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from traders.llm.action_parser import BidAskAction, BuySellAction
from traders.llm.base_llm_agent import BaseLLMAgent
from traders.llm.cache_manager import CacheManager

# Import LiteLLM
try:
    import litellm
    from litellm import completion

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logging.warning("LiteLLM not available. Install with: pip install litellm")


logger = logging.getLogger(__name__)


class GPTAgent(BaseLLMAgent):
    """
    GPT-powered trading agent.

    Uses OpenAI GPT models via LiteLLM to make decisions based on
    natural language prompts describing market state.

    Models supported:
    - gpt-4o: Latest GPT-4 Omni model (fast, high quality)
    - gpt-4o-mini: Smaller GPT-4 model (faster, cheaper)
    - gpt-3.5-turbo: GPT-3.5 (cheapest, baseline)
    - groq/llama-3.1-8b-instant: Groq free tier (fast, free!)
    - groq/llama-3.1-70b-versatile: Groq larger model (free!)
    """

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
        price_min: int = 0,
        price_max: int = 1000,
        max_retries: int = 3,
        num_times: int = 100,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        use_cache: bool = True,
        cache_dir: str = ".cache/llm_responses",
        output_dir: str | None = None,
        api_key: str | None = None,
        use_cot: bool = True,
        prompt_style: str = "minimal",
        **kwargs: Any,
    ) -> None:
        """
        Initialize GPT agent.

        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations
            price_min: Minimum market price
            price_max: Maximum market price
            max_retries: Max retry attempts for invalid actions
            num_times: Maximum time steps per period (default 100)
            model: Model name (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
            temperature: Sampling temperature (0-2, default 0.7)
            use_cache: Enable response caching
            cache_dir: Cache directory path
            output_dir: Directory to save prompts/responses (None = no logging)
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            use_cot: Enable chain-of-thought reasoning (default True)
            prompt_style: "minimal" for pure facts, "original" for verbose prompts
            **kwargs: Additional parameters
        """
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "LiteLLM is required for GPT agents. " "Install with: pip install litellm"
            )

        super().__init__(
            player_id,
            is_buyer,
            num_tokens,
            valuations,
            price_min,
            price_max,
            max_retries,
            num_times,
            **kwargs,
        )

        self.model = model
        self.temperature = temperature

        # Setup API key
        if api_key:
            # Determine which env var to set based on model
            if "groq/" in model:
                os.environ["GROQ_API_KEY"] = api_key
            else:
                os.environ["OPENAI_API_KEY"] = api_key
        else:
            # Check for appropriate API key
            if "groq/" in model and "GROQ_API_KEY" not in os.environ:
                logger.warning("No Groq API key found. Get free key at: https://console.groq.com")
            elif "groq/" not in model and "OPENAI_API_KEY" not in os.environ:
                logger.warning("No OpenAI API key found. Set OPENAI_API_KEY environment variable.")

        # Setup caching
        self.use_cache = use_cache
        self.cache = CacheManager(cache_dir) if use_cache else None

        # Setup logging directories
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.prompts_dir = self.output_dir / "prompts"
            self.responses_dir = self.output_dir / "responses"
            self.prompts_dir.mkdir(parents=True, exist_ok=True)
            self.responses_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Logging prompts/responses to: {self.output_dir}")

        # Chain-of-Thought reasoning
        self.use_cot = use_cot

        # Prompt style: "minimal" or "original"
        self.prompt_style = prompt_style

        # System prompts - separate for each stage to avoid action confusion
        if prompt_style == "deep":
            self.system_prompt_bid_ask = self.prompt_builder.build_deep_context_system_prompt(
                is_buyer,
                stage="bid_ask",
                num_buyers=self.num_buyers,
                num_sellers=self.num_sellers,
                num_tokens=num_tokens,
                max_time=num_times,
            )
            self.system_prompt_buy_sell = self.prompt_builder.build_deep_context_system_prompt(
                is_buyer,
                stage="buy_sell",
                num_buyers=self.num_buyers,
                num_sellers=self.num_sellers,
                num_tokens=num_tokens,
                max_time=num_times,
            )
        elif prompt_style == "minimal":
            self.system_prompt_bid_ask = self.prompt_builder.build_minimal_system_prompt(
                is_buyer, stage="bid_ask"
            )
            self.system_prompt_buy_sell = self.prompt_builder.build_minimal_system_prompt(
                is_buyer, stage="buy_sell"
            )
        else:
            self.system_prompt_bid_ask = self.prompt_builder.build_system_prompt(
                is_buyer, use_cot=use_cot, stage="bid_ask"
            )
            self.system_prompt_buy_sell = self.prompt_builder.build_system_prompt(
                is_buyer, use_cot=use_cot, stage="buy_sell"
            )
        # Default for logging compatibility
        self.system_prompt = self.system_prompt_bid_ask

        # Decision counter for logging
        self._decision_count = 0

        logger.info(f"Initialized GPTAgent (id={player_id}, model={model}, style={prompt_style})")

    def _call_llm(self, prompt: str, stage: str = "bid_ask") -> str:
        """
        Call LLM API with caching, rate limit handling, and error handling.

        Args:
            prompt: The prompt to send
            stage: "bid_ask" or "buy_sell" - determines which system prompt to use

        Returns:
            LLM response text

        Raises:
            Exception if API call fails after retries
        """
        # Select appropriate system prompt for stage
        system_prompt = (
            self.system_prompt_buy_sell if stage == "buy_sell" else self.system_prompt_bid_ask
        )
        self.system_prompt = system_prompt  # Update for logging

        # Check cache first (include stage in key to avoid cross-stage collisions)
        if self.use_cache and self.cache:
            cached = self.cache.get_cached_response(prompt, self.model, stage)
            if cached:
                # Log cache hit if output directory is set
                if self.output_dir:
                    self._log_decision(prompt, cached["response"], from_cache=True)
                logger.debug(f"Cache hit for stage={stage}")
                return cached["response"]

        # Call LLM API with rate limit retry
        max_retries = 5
        base_delay = 4.0  # seconds

        for attempt in range(max_retries):
            try:
                response = completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},  # JSON mode
                    max_tokens=1500,  # Large buffer for CoT reasoning
                    timeout=30.0,  # 30 second timeout
                )

                # Check for truncation
                finish_reason = response.choices[0].finish_reason
                if finish_reason == "length":
                    logger.warning(
                        f"Response TRUNCATED (finish_reason=length). "
                        f"Increase max_tokens. Model: {self.model}"
                    )

                # Extract response
                response_text = response.choices[0].message.content

                # Validate response is not empty
                if not response_text:
                    logger.error("Empty content in LLM response (None)")
                    raise ValueError("Empty content in LLM response")

                if not response_text.strip():
                    logger.error(
                        f"Empty content in LLM response (whitespace only): {repr(response_text)}"
                    )
                    raise ValueError("Empty content in LLM response (whitespace only)")

                # Debug log raw response
                logger.debug(f"RAW LLM RESPONSE: {repr(response_text[:200])}...")

                # Log decision to files
                if self.output_dir:
                    self._log_decision(prompt, response_text, from_cache=False)

                # Store in cache (include stage in key)
                if self.use_cache and self.cache:
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                    self.cache.store_response(
                        prompt, self.model, response_text, input_tokens, output_tokens, stage
                    )

                return response_text

            except litellm.RateLimitError:
                # Exponential backoff for rate limits
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
                if attempt == max_retries - 1:
                    logger.error(f"Rate limit exceeded after {max_retries} retries")
                    raise

            except Exception as e:
                logger.error(f"LLM API call failed: {e}")
                raise

    def _log_decision(self, prompt: str, response: str, from_cache: bool = False):
        """
        Log prompt and response to files for debugging.

        Args:
            prompt: The prompt sent to LLM
            response: The response received from LLM
            from_cache: Whether response came from cache
        """
        if not self.output_dir:
            return

        # Increment decision counter
        self._decision_count += 1

        # Generate filename
        agent_type = "buyer" if self.is_buyer else "seller"
        filename_base = f"agent_{self.player_id}_{agent_type}_decision_{self._decision_count:04d}"

        # Save prompt (includes full context)
        prompt_file = self.prompts_dir / f"{filename_base}.txt"
        full_prompt = (
            f"=== SYSTEM PROMPT ===\n{self.system_prompt}\n\n=== USER PROMPT ===\n{prompt}\n"
        )
        prompt_file.write_text(full_prompt)

        # Parse response to extract reasoning
        parsed_response = (
            json.loads(response) if response.strip().startswith("{") else {"raw": response}
        )

        # Save response with metadata
        response_file = self.responses_dir / f"{filename_base}.json"
        response_data = {
            "decision_number": self._decision_count,
            "agent_id": self.player_id,
            "agent_type": agent_type,
            "model": self.model,
            "use_cot": self.use_cot,
            "from_cache": from_cache,
            "reasoning": parsed_response.get("reasoning", None),  # Extract reasoning
            "action": parsed_response.get("action", None),
            "price": parsed_response.get("price", None),
            "full_response": parsed_response,
        }
        response_file.write_text(json.dumps(response_data, indent=2))

    def _generate_bid_ask_action(
        self, prompt: str, valuation: int, best_bid: int, best_ask: int
    ) -> BidAskAction:
        """
        Generate bid/ask action using GPT.

        Args:
            prompt: Natural language prompt
            valuation: Current token valuation
            best_bid: Current best bid
            best_ask: Current best ask

        Returns:
            BidAskAction parsed from LLM response
        """
        response_text = None
        try:
            # Call LLM with bid_ask stage
            response_text = self._call_llm(prompt, stage="bid_ask")

            # Clean whitespace
            response_text = response_text.strip()

            # Parse JSON response
            response_data = json.loads(response_text)

            # Log parsed action
            logger.debug(
                f"Parsed bid_ask action: {response_data.get('action')}, price={response_data.get('price')}"
            )

            # Validate and create action
            action = BidAskAction(**response_data)
            return action

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in bid_ask: {e}")
            logger.error(f"Raw response was: {repr(response_text)}")
            return BidAskAction(action="pass", reasoning="JSON parse error")

        except Exception as e:
            logger.error(f"Failed to parse bid_ask response: {e}")
            logger.error(f"Raw response was: {repr(response_text)}")
            return BidAskAction(action="pass", reasoning=f"Parse error: {e}")

    def _generate_buy_sell_action(
        self, prompt: str, valuation: int, trade_price: int
    ) -> BuySellAction:
        """
        Generate buy/sell decision using GPT.

        Args:
            prompt: Natural language prompt
            valuation: Current token valuation
            trade_price: Price if trade is accepted

        Returns:
            BuySellAction parsed from LLM response
        """
        response_text = None
        try:
            # Call LLM with buy_sell stage
            response_text = self._call_llm(prompt, stage="buy_sell")

            # Clean whitespace
            response_text = response_text.strip()

            # Parse JSON response
            response_data = json.loads(response_text)

            # Log parsed action
            logger.debug(f"Parsed buy_sell action: {response_data.get('action')}")

            # Validate and create action
            action = BuySellAction(**response_data)
            return action

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in buy_sell: {e}")
            logger.error(f"Raw response was: {repr(response_text)}")
            return BuySellAction(action="pass", reasoning="JSON parse error")

        except Exception as e:
            logger.error(f"Failed to parse buy_sell response: {e}")
            logger.error(f"Raw response was: {repr(response_text)}")
            return BuySellAction(action="pass", reasoning=f"Parse error: {e}")

    def end_period(self):
        """Print cache statistics at end of period."""
        super().end_period()

        # Print cache stats every 10 periods
        if self.use_cache and self.cache:
            stats = self.cache.get_statistics()
            if stats["total_calls"] > 0 and stats["total_calls"] % 50 == 0:
                self.cache.print_statistics()

    def __repr__(self) -> str:
        """String representation."""
        agent_type = "Buyer" if self.is_buyer else "Seller"
        invalid_rate = self.get_invalid_action_rate()
        return (
            f"GPTAgent(id={self.player_id}, model={self.model}, "
            f"type={agent_type}, trades={self.num_trades}/{self.num_tokens}, "
            f"invalid_rate={invalid_rate:.1f}%)"
        )
