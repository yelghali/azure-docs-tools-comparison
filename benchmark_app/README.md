# 📄 Document Processing Benchmark

Compare **Azure Content Understanding**, **Document Intelligence + GPT**, and **Mistral Document AI** side-by-side on your documents. Supports GPT-4.1 and GPT-5.1 as the LLM summary model.

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
| `AZURE_OPENAI_BASE` | Yes | Azure OpenAI base URL (e.g. `https://my-resource.openai.azure.com`) |
| `AZURE_OPENAI_KEY` | Optional | API key (falls back to `DefaultAzureCredential`) |
| `MISTRAL_DOC_AI_ENDPOINT` | For Mistral | Mistral OCR endpoint (Azure APIM) |
| `MISTRAL_DOC_AI_KEY` | For Mistral | Mistral API key |
| `MISTRAL_DOC_AI_MODEL` | For Mistral | Model name (default: `mistral-document-ai-2505`) |

## How to Use

1. **Select models** — pick `layout`, `read`, or both in the sidebar
2. **Choose LLM** — select GPT-4.1 or GPT-5.1 for the text summary step
3. **Enable pipelines** — check which services to compare (CU, Doc Intelligence, Mistral)
4. **Upload documents** — drag & drop files or paste URLs
5. **Run benchmark** — click 🚀 and wait for parallel execution
6. **Compare results** — view metrics, field comparisons, rendered markdown, and raw API results (paragraphs, polygons, tables)

When multiple prebuilt models are selected, CU and Doc Intelligence run with **each model** so you can compare read vs layout side-by-side. Mistral uses its own OCR model regardless.

## Architecture

```
Upload → ┌── Azure Content Understanding (SDK, prebuilt analyzer) + GPT summary
         ├── Azure Document Intelligence (SDK) + GPT summary
         └── Mistral Document AI (REST OCR via Azure APIM)
              ↓
         Side-by-side comparison, metrics, markdown preview & raw API results
```

## Project Structure

```
benchmark_app/
├── app.py                          # Streamlit UI & benchmark runner
├── config.py                       # Env vars, model mappings & GPT endpoint builder
├── prompts.py                      # Default LLM prompts
├── requirements.txt
├── services/
│   ├── content_understanding.py    # Azure CU SDK + GPT text summary
│   ├── doc_intel_gpt.py            # Doc Intelligence SDK + GPT text summary
│   └── mistral_vision.py           # Mistral OCR via REST
└── utils/
    └── comparison.py               # Comparison tables & summary stats
```
