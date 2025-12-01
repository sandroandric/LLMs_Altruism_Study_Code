"""OpenRouter API client for querying LLMs."""

import time
import httpx
from typing import Optional

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    EXPERIMENT_CONFIG,
)


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""

    def __init__(self):
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY not set. "
                "Copy .env.example to .env and add your key."
            )
        self.base_url = OPENROUTER_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/sandroandric/altruism",
            "X-Title": "LLM Implicit Altruism Test",
        }
        self.timeout = EXPERIMENT_CONFIG["request_timeout_seconds"]
        self.max_retries = EXPERIMENT_CONFIG["max_retries"]
        self.retry_delay = EXPERIMENT_CONFIG["retry_delay_seconds"]

    def call_model(
        self,
        model_name: str,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        Send a prompt to a model and return the response text.

        Args:
            model_name: OpenRouter model identifier (e.g., "openai/gpt-4o")
            prompt: The prompt to send
            temperature: Sampling temperature (default from config)
            top_p: Top-p sampling parameter (default from config)

        Returns:
            The model's response text

        Raises:
            Exception: If all retries fail
        """
        if temperature is None:
            temperature = EXPERIMENT_CONFIG["iat_temperature"]
        if top_p is None:
            top_p = EXPERIMENT_CONFIG["iat_top_p"]

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()

                    if "choices" in data and len(data["choices"]) > 0:
                        return data["choices"][0]["message"]["content"]
                    else:
                        raise ValueError(f"Unexpected response format: {data}")

            except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                continue

        raise Exception(
            f"Failed to get response from {model_name} after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )


# Global client instance
_client: Optional[OpenRouterClient] = None


def get_client() -> OpenRouterClient:
    """Get or create the global OpenRouter client."""
    global _client
    if _client is None:
        _client = OpenRouterClient()
    return _client


def call_model(model_name: str, prompt: str) -> str:
    """
    Convenience function to call a model.

    This is the main interface used by experiment runners.
    """
    return get_client().call_model(model_name, prompt)
