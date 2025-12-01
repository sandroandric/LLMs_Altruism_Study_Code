# Model List for Full Study

**Target N = 24+ models** (for statistical power)

## Tier 1: Frontier Models (Must Include)

| # | Provider | Model ID (OpenRouter) | Notes |
|---|----------|----------------------|-------|
| 1 | OpenAI | `openai/gpt-4o` | Flagship multimodal |
| 2 | OpenAI | `openai/gpt-4o-mini` | Smaller, faster |
| 3 | OpenAI | `openai/gpt-4-turbo` | Previous flagship |
| 4 | OpenAI | `openai/o1-preview` | Reasoning model |
| 5 | OpenAI | `openai/o1-mini` | Smaller reasoning |
| 6 | Anthropic | `anthropic/claude-3.5-sonnet` | Current flagship |
| 7 | Anthropic | `anthropic/claude-3-opus` | Largest Claude |
| 8 | Anthropic | `anthropic/claude-3-haiku` | Smallest Claude |
| 9 | Google | `google/gemini-pro-1.5` | Flagship Gemini |
| 10 | Google | `google/gemini-flash-1.5` | Fast Gemini |

## Tier 2: Strong Open/Commercial Models

| # | Provider | Model ID (OpenRouter) | Notes |
|---|----------|----------------------|-------|
| 11 | Meta | `meta-llama/llama-3.1-405b-instruct` | Largest open |
| 12 | Meta | `meta-llama/llama-3.1-70b-instruct` | Mid-size open |
| 13 | Meta | `meta-llama/llama-3.1-8b-instruct` | Smallest Llama 3.1 |
| 14 | Mistral | `mistralai/mistral-large` | Mistral flagship |
| 15 | Mistral | `mistralai/mixtral-8x22b-instruct` | MoE model |
| 16 | Mistral | `mistralai/mistral-7b-instruct` | Smallest Mistral |
| 17 | Cohere | `cohere/command-r-plus` | Enterprise model |
| 18 | Cohere | `cohere/command-r` | Smaller Cohere |

## Tier 3: Emerging/Regional Models

| # | Provider | Model ID (OpenRouter) | Notes |
|---|----------|----------------------|-------|
| 19 | X-AI | `x-ai/grok-beta` | X/Twitter model |
| 20 | DeepSeek | `deepseek/deepseek-chat` | Chinese lab |
| 21 | Qwen | `qwen/qwen-2.5-72b-instruct` | Alibaba |
| 22 | Qwen | `qwen/qwen-2.5-7b-instruct` | Smaller Qwen |
| 23 | 01.AI | `01-ai/yi-large` | Yi model |
| 24 | Databricks | `databricks/dbrx-instruct` | Enterprise |

## Tier 4: Backup/Additional Models

| # | Provider | Model ID (OpenRouter) | Notes |
|---|----------|----------------------|-------|
| 25 | Perplexity | `perplexity/llama-3.1-sonar-large-128k-chat` | Search-augmented |
| 26 | Nous | `nousresearch/hermes-3-llama-3.1-405b` | Fine-tuned |
| 27 | Microsoft | `microsoft/phi-3-medium-128k-instruct` | Small but capable |
| 28 | AI21 | `ai21/jamba-1-5-large` | Hybrid architecture |

---

## Model Selection Criteria

1. **Availability**: Must be accessible via OpenRouter API
2. **Instruction-tuned**: Must follow instructions (not base models)
3. **Diverse providers**: Represent different training approaches
4. **Size variation**: Include small, medium, large within providers
5. **Recency**: Prefer recent models (2024+)

---

## Cost Estimates

### Per-Model Costs (Approximate)

| Task | Trials | Tokens/Trial | Total Tokens |
|------|--------|--------------|--------------|
| IAT | 30 | ~200 | 6,000 |
| Forced Choice | 51 | ~300 | 15,300 |
| Self-Assessment | 3 | ~2,000 | 6,000 |
| **Total** | | | **~27,300** |

### Estimated Cost by Model Tier

| Model Tier | $/1M tokens | Per Model | N Models | Subtotal |
|------------|-------------|-----------|----------|----------|
| Frontier (GPT-4o, Claude) | $15 | $0.41 | 10 | $4.10 |
| Mid-tier (Llama 70B) | $1 | $0.03 | 8 | $0.24 |
| Small/Free | $0.10 | $0.003 | 6 | $0.02 |
| **Total Estimate** | | | **24** | **~$5-10** |

*Note: Costs are approximate and vary by provider. Budget $20-30 for safety margin.*

---

## Verification Checklist

Before running full study:

- [ ] Verify each model is available on OpenRouter
- [ ] Test each model with 1 trial of each task
- [ ] Confirm parse rates > 80%
- [ ] Check for rate limits
- [ ] Estimate actual costs from test runs

---

## Configuration File

Copy to `src/config.py` when ready:

```python
FULL_STUDY_MODELS = [
    # Tier 1: Frontier
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4-turbo",
    "openai/o1-preview",
    "openai/o1-mini",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-haiku",
    "google/gemini-pro-1.5",
    "google/gemini-flash-1.5",
    # Tier 2: Strong
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-large",
    "mistralai/mixtral-8x22b-instruct",
    "mistralai/mistral-7b-instruct",
    "cohere/command-r-plus",
    "cohere/command-r",
    # Tier 3: Emerging
    "x-ai/grok-beta",
    "deepseek/deepseek-chat",
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "01-ai/yi-large",
    "databricks/dbrx-instruct",
]
```
