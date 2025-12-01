"""Configuration for full study with 24 models."""

# Full study model list - 24 unique models
# Priority 1: 9 pilot study models (validated)
# Priority 2: 15 additional models from user selection

FULL_STUDY_MODELS = [
    # === PRIORITY 1: Core Models (9) ===
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-haiku",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash-lite",
    "meta-llama/llama-3.1-70b-instruct",
    "mistralai/mistral-large",
    "x-ai/grok-4-fast",

    # === PRIORITY 2: Additional Models ===
    # OpenAI additions
    "openai/gpt-4.1-mini",
    "openai/gpt-5.1",
    "openai/gpt-5-mini",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",

    # Anthropic additions
    "anthropic/claude-opus-4.5",

    # Google additions
    "google/gemini-3-pro-preview",
    "google/gemma-3-12b-it",

    # Meta/Llama additions
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-4-maverick",

    # Mistral additions
    "mistralai/mistral-nemo",

    # X-AI additions
    "x-ai/grok-4",

    # Other providers
    "z-ai/glm-4.6",

    # === BUFFER: Extra models for robustness (2) ===
    "microsoft/phi-4",
    "ibm-granite/granite-4.0-h-micro",
]

# Verify count
assert len(FULL_STUDY_MODELS) == 24, f"Expected 24, got {len(FULL_STUDY_MODELS)}"

# Provider breakdown
PROVIDERS = {
    "openai": 7,
    "anthropic": 3,
    "google": 4,
    "meta-llama": 3,
    "mistralai": 2,
    "x-ai": 2,
    "minimax": 1,
    "moonshotai": 1,
    "z-ai": 1,
    "microsoft": 1,
    "ibm-granite": 1,
}

# Study parameters
IAT_TRIALS = 30          # Per model
FC_REPEATS = 3           # Per scenario (17 scenarios x 3 = 51 trials)
SA_REPEATS = 3           # Per assessment (15 items x 3 = 45 trials)
TEMPERATURE = 0.1        # Low temperature for reproducibility
