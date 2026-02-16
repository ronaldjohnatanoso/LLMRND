# Groq Setup Guide - FREE Llama 3.1 8B Inference

**Groq** offers the fastest LLM inference with Llama 3.1 models - **completely FREE**.

## Quick Setup

### 1. Get Your API Key

1. Go to https://console.groq.com/
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy your key

### 2. Configure Environment

Create a `.env` file in your project root:

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```env
GROQ_API_KEY=gsk_your_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
LLM_PROVIDER=groq
```

### 3. Use in Code

```python
from cog_memory import CognitiveMemory

# Option 1: Automatic (reads from env)
memory = CognitiveMemory()

# Option 2: Explicit
memory = CognitiveMemory(
    provider="groq",
    model="llama-3.1-8b-instant",
)

# Ingest text
text = "We need to launch by Q3. Budget is $50k."
nodes = memory.ingest_paragraph(text)
```

## Available Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `llama-3.1-8b-instant` | Fast, efficient | General use, testing |
| `llama-3.1-70b-versatile` | Higher quality | Complex reasoning |
| `llama-3.3-70b-speculate` | Speculative | Fast generation |

## Performance

- **Speed**: 500+ tokens/second (fastest available)
- **Latency**: <100ms for most requests
- **Cost**: $0 (free tier)
- **Limits**: Generous rate limits for development

## Comparison

| Provider | Free Tier | Speed | Model |
|----------|-----------|-------|-------|
| **Groq** | ✅ Unlimited | 🚀 Fastest | Llama 3.1 8B/70B |
| Hugging Face | 60 req/min | Fast | Llama 3.1 8B |
| OpenAI | No | Fast | GPT-4o (paid) |
| Fireworks | $1 credit | Fast | Llama 3.1 8B |

## Troubleshooting

### API Key Not Found

```bash
export GROQ_API_KEY="your_key_here"
```

Or create `.env` file with the key.

### Rate Limits

Groq has generous limits. If you hit limits:
- Use batch requests where possible
- Implement exponential backoff
- Consider caching embeddings

### Model Not Available

Check the model name:
```python
# Correct
model="llama-3.1-8b-instant"

# Incorrect
model="llama-3.1-8b"  # Missing "-instant"
model="llama-3-8b"    # Wrong version
```

## Switching Providers

```python
# Groq (default, free)
memory = CognitiveMemory(provider="groq")

# OpenAI (paid)
memory = CognitiveMemory(provider="openai")

# Hugging Face (free tier)
memory = CognitiveMemory(provider="huggingface")

# Dummy extractor (no API)
memory = CognitiveMemory(use_dummy_extractor=True)
```

## Links

- **Console**: https://console.groq.com/
- **Docs**: https://console.groq.com/docs
- **Models**: https://console.groq.com/docs/models
- **Python SDK**: https://github.com/groq/groq-python
