"""
Cache manager for LLM API calls.

Implements semantic caching and cost tracking to minimize API expenses
during LLM agent experiments.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional
from datetime import datetime


logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages caching of LLM responses and tracks API costs.

    Features:
    - Semantic caching: Hash (prompt, model) → response
    - Cost tracking: Total tokens and estimated cost
    - Persistent disk storage
    """

    # Pricing per 1M tokens (as of 2024)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.150, "output": 0.600},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
        "groq/llama-3.1-8b-instant": {"input": 0.0, "output": 0.0},  # FREE!
        "groq/llama-3.1-70b-versatile": {"input": 0.0, "output": 0.0},  # FREE!
    }

    def __init__(self, cache_dir: str = ".cache/llm_responses"):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.total_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

        # Load cache
        self._cache = self._load_cache()

        logger.info(f"CacheManager initialized: {len(self._cache)} cached responses")

    def _load_cache(self) -> dict:
        """Load cache from disk."""
        cache_file = self.cache_dir / "cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return {}
        return {}

    def _save_cache(self):
        """Save cache to disk."""
        cache_file = self.cache_dir / "cache.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _compute_cache_key(self, prompt: str, model: str, stage: str = "") -> str:
        """
        Compute cache key from prompt, model, and stage.

        Args:
            prompt: The prompt string
            model: Model name
            stage: Stage identifier (bid_ask or buy_sell)

        Returns:
            SHA256 hash as hex string
        """
        content = f"{model}::{stage}::{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get_cached_response(self, prompt: str, model: str, stage: str = "") -> Optional[dict]:
        """
        Try to get cached response.

        Args:
            prompt: The prompt string
            model: Model name
            stage: Stage identifier (bid_ask or buy_sell)

        Returns:
            Cached response dict or None if not found
        """
        self.total_calls += 1
        cache_key = self._compute_cache_key(prompt, model, stage)

        if cache_key in self._cache:
            self.cache_hits += 1
            logger.debug(f"Cache HIT for key: {cache_key[:16]}... (stage={stage})")
            return self._cache[cache_key]

        self.cache_misses += 1
        logger.debug(f"Cache MISS for key: {cache_key[:16]}... (stage={stage})")
        return None

    def store_response(
        self,
        prompt: str,
        model: str,
        response: str,
        input_tokens: int,
        output_tokens: int,
        stage: str = ""
    ):
        """
        Store LLM response in cache.

        Args:
            prompt: The prompt string
            model: Model name
            response: LLM response text
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            stage: Stage identifier (bid_ask or buy_sell)
        """
        cache_key = self._compute_cache_key(prompt, model, stage)

        # Store in cache
        self._cache[cache_key] = {
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model,
            "timestamp": datetime.now().isoformat()
        }

        # Update statistics
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        # Calculate cost
        if model in self.PRICING:
            pricing = self.PRICING[model]
            cost = (
                (input_tokens / 1_000_000) * pricing["input"] +
                (output_tokens / 1_000_000) * pricing["output"]
            )
            self.total_cost += cost

        # Periodically save cache (every 10 misses)
        if self.cache_misses % 10 == 0:
            self._save_cache()

    def get_statistics(self) -> dict:
        """
        Get cache and cost statistics.

        Returns:
            Dictionary with statistics
        """
        hit_rate = (self.cache_hits / self.total_calls * 100) if self.total_calls > 0 else 0

        return {
            "total_calls": self.total_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "cached_responses": len(self._cache)
        }

    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()

        logger.info("\n" + "=" * 60)
        logger.info("LLM CACHE STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total API calls:     {stats['total_calls']}")
        logger.info(f"Cache hits:          {stats['cache_hits']} ({stats['hit_rate']:.1f}%)")
        logger.info(f"Cache misses:        {stats['cache_misses']}")
        logger.info(f"Cached responses:    {stats['cached_responses']}")
        logger.info(f"\nToken Usage:")
        logger.info(f"Input tokens:        {stats['total_input_tokens']:,}")
        logger.info(f"Output tokens:       {stats['total_output_tokens']:,}")
        logger.info(f"Total tokens:        {stats['total_tokens']:,}")
        logger.info(f"\nEstimated Cost:      ${stats['total_cost']:.4f}")
        logger.info("=" * 60)

    def clear_cache(self):
        """Clear all cached responses."""
        self._cache = {}
        self._save_cache()
        logger.info("Cache cleared")

    def __del__(self):
        """Save cache on cleanup."""
        self._save_cache()
