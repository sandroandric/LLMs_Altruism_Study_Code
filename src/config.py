"""Configuration for the LLM Implicit Altruism Test study."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models to test (10 diverse LLMs from different providers)
MODELS = [
    # OpenAI
    "openai/gpt-5.1-chat",
    "openai/chatgpt-4o-latest",
    "openai/gpt-4.1",
    # Anthropic
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-opus-4.5",
    # Google
    "google/gemini-2.5-flash-preview-09-2025",
    # xAI
    "x-ai/grok-4.1-fast:free",
    # DeepSeek
    "deepseek/deepseek-v3.2-exp",
    # Qwen
    "qwen/qwen3-max",
]

# Experiment parameters
EXPERIMENT_CONFIG = {
    # IAT task settings
    "iat_trials_per_model": 30,
    "iat_temperature": 0.1,
    "iat_top_p": 0.3,

    # Decision task settings
    "decision_repeats_per_scenario": 3,
    "decision_temperature": 0.1,
    "decision_top_p": 0.3,

    # General settings
    "random_seed": 42,
    "max_retries": 3,
    "retry_delay_seconds": 2,
    "request_timeout_seconds": 60,
}

# Output paths
RESULTS_DIR = "results"
