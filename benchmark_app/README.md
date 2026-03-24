# 📄 Document Processing Benchmark

Compare **Azure Content Understanding**, **Document Intelligence + GPT-4.1**, and **Mistral Document AI** side-by-side on your documents.

## Quick Start

```bash
cd benchmark_app
pip install -r requirements.txt
az login                           # DefaultAzureCredential is used for Azure services
streamlit run app.py
```

## Configuration

Copy `.env.example` to `.env` and fill in your endpoints:

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_CU_ENDPOINT` | For CU | Content Understanding endpoint |
| `DOC_INTELLIGENCE_ENDPOINT` | For DI | Document Intelligence endpoint |
| `DOC_INTELLIGENCE_KEY` | For DI | Document Intelligence API key |
| `GPT_ENDPOINT` | Yes | GPT-4.1 chat completions URL (full URL with deployment + api-version) |
| `GPT_KEY` | Optional | GPT API key (falls back to `DefaultAzureCredential`) |
| `MISTRAL_DOC_AI_ENDPOINT` | For Mistral | Mistral OCR endpoint (Azure APIM) |
| `MISTRAL_DOC_AI_KEY` | For Mistral | Mistral API key |
| `MISTRAL_DOC_AI_MODEL` | For Mistral | Model name (default: `mistral-document-ai-2505`) |

## How to Use

1. **Select models** — pick one or more prebuilt models (`invoice`, `layout`, `read`) in the sidebar
2. **Enable pipelines** — check which services to compare (CU, Doc Intelligence, Mistral)
3. **Upload documents** — drag & drop files or paste URLs
4. **Run benchmark** — click 🚀 and wait for parallel execution
5. **Compare results** — view metrics, field comparisons, and rendered markdown previews

When multiple models are selected, CU and Doc Intelligence run with **each model** so you can compare read vs layout side-by-side. Mistral uses its own OCR model regardless.

## Architecture

```
Upload → ┌── Azure Content Understanding (SDK, prebuilt analyzer)
         ├── Azure Document Intelligence + GPT-4.1 (text extraction → LLM)
         └── Mistral Document AI (REST OCR via Azure APIM)
              ↓
         Side-by-side comparison, metrics & rendered markdown preview
```

## Project Structure

```
benchmark_app/
├── app.py                          # Streamlit UI & benchmark runner
├── config.py                       # Env vars & model mappings
├── prompts.py                      # Default LLM prompts
├── requirements.txt
├── services/
│   ├── content_understanding.py    # Azure CU SDK + GPT-4.1 text summary
│   ├── doc_intel_gpt.py            # Doc Intelligence SDK + GPT-4.1 text summary
│   └── mistral_vision.py           # Mistral OCR via REST
└── utils/
    └── comparison.py               # Comparison tables & summary stats
```
