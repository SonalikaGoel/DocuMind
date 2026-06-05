# DocuMind with Ollama Setup Guide

Welcome to the free, quota-unlimited version of DocuMind! This guide will help you set up Ollama locally.

## Why Ollama?

- ✅ **Completely free** - No API costs, no quotas, no rate limits
- ✅ **Runs locally** - All processing happens on your machine
- ✅ **Unlimited usage** - Ask as many questions as you want
- ✅ **No internet required** (after initial setup) - Works offline

## Prerequisites

- Windows/Mac/Linux
- 8GB+ RAM (minimum)
- 10GB+ free disk space (for models)
- Python 3.8+

## Installation Steps

### 1. Install Ollama

**Windows/Mac:**
1. Download from: https://ollama.ai/download
2. Run the installer and follow the prompts
3. Verify installation in terminal:
   ```bash
   ollama --version
   ```

**Linux (Ubuntu/Debian):**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Pull a Language Model

Open a terminal and run:
```bash
ollama pull mistral
```

This downloads the Mistral 7B model (~4GB). First run takes 2-5 minutes.

**Alternative models (smaller/larger):**
- `ollama pull neural-chat` - Smaller, 4B parameters
- `ollama pull llama2` - Larger, 7B parameters  
- `ollama pull dolphin-mixtral` - Larger, 8x7B parameters

### 3. Start Ollama Server

```bash
ollama serve
```

This starts a local server on `http://localhost:11434`

**Note:** Keep this terminal open while using DocuMind. You can minimize it.

### 4. Install Python Dependencies

In another terminal, activate your venv and install:
```bash
pip install -r requirements.txt
```

This installs:
- `langchain-community` - Ollama integration
- `sentence-transformers` - Free embeddings  
- Other dependencies

### 5. Run DocuMind

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Troubleshooting

### "Connection refused" error
- Make sure Ollama is running (`ollama serve` in another terminal)
- Check that it's listening on `http://localhost:11434`

### Slow responses
- First query is slow (model loading)
- Subsequent queries are faster
- Try smaller model: `ollama pull neural-chat`

### Out of memory
- Reduce model size: switch to `neural-chat` or `orca-mini`
- Close other applications

### Model not found
```bash
ollama list  # See installed models
ollama pull mistral  # Download mistral
```

## Changing Models

Edit [rag_pipeline.py](rag_pipeline.py) and change:
```python
LLM_MODEL = "mistral"  # Change this to another model name
```

Then restart the app.

## Performance Notes

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| neural-chat | 4GB | Fast | Good |
| mistral | 4GB | Medium | Excellent |
| llama2 | 4GB | Medium | Good |
| dolphin-mixtral | 26GB | Slow | Best |

**Recommendation:** Start with `mistral` for balanced quality and speed.

## Next Steps

1. Start Ollama: `ollama serve`
2. Run DocuMind: `streamlit run app.py`
3. Upload a PDF and start asking questions!

## For Production

Want to use this with a cloud API? You can easily swap Ollama for:
- **OpenAI** (ChatOpenAI)
- **LLaMA Cloud** (Together AI)
- **HuggingFace Inference**

Just update [rag_pipeline.py](rag_pipeline.py) and install the appropriate package.
