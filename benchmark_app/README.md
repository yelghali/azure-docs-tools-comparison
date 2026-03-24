# 📄 Document Processing Benchmark

Compare **Azure Content Understanding**, **Document Intelligence + GPT**, and **Mistral Document AI** side-by-side on your documents. Supports GPT-4.1 and GPT-5.1 as the LLM summary model.

## Quick Start

```bash
cd benchmark_app
pip install -r requirements.txt
az login                           # DefaultAzureCredential is used for Azure services
streamlit run app.py               # opens at http://localhost:8501
```

## Configuration

Create a `.env` file in the project root with your endpoints:

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_CU_ENDPOINT` | For CU | Content Understanding endpoint |
| `DOC_INTELLIGENCE_ENDPOINT` | For DI | Document Intelligence endpoint |
| `DOC_INTELLIGENCE_KEY` | For DI | Document Intelligence API key |
| `AZURE_OPENAI_BASE` | Yes | Azure OpenAI base URL (e.g. `https://my-resource.openai.azure.com`) |
| `AZURE_OPENAI_KEY` | Optional | API key (falls back to `DefaultAzureCredential`) |
| `AZURE_OPENAI_API_VERSION` | Optional | API version (default: `2024-10-21`) |
| `MISTRAL_DOC_AI_ENDPOINT` | For Mistral | Mistral OCR endpoint (Azure APIM) |
| `MISTRAL_DOC_AI_KEY` | For Mistral | Mistral API key |
| `MISTRAL_DOC_AI_MODEL` | For Mistral | Model name (default: `mistral-document-ai-2505`) |

## How to Use

1. **Sidebar → Prebuilt Model(s)** — pick `Layout`, `Read`, or both. CU and Doc Intelligence run each selected model separately so you can compare them.
2. **Sidebar → LLM Model** — choose GPT-4.1 or GPT-5.1 for the structured text summary step.
3. **Sidebar → Pipelines** — check which services to benchmark (🔵 CU, 🟢 Doc Intelligence + GPT, 🟠 Mistral). All checked pipelines run **in parallel**.
4. **Upload or paste URLs** — use the "Upload Files" tab (drag & drop, multi-file) or the "Enter URLs" tab.
5. **Customize prompts** *(optional)* — expand "📝 Customize Analysis Prompts" in the main area to edit the system and user prompts sent to GPT.
6. **Click 🚀 Run Benchmark** — progress bar shows per-document status.
7. **Review results** — for each document and pipeline you get:
   - ⏱ Timing & confidence metrics
   - 📝 LLM-generated description
   - 📄 Extracted markdown (raw + rendered tabs)
   - 📋 Extracted fields (expandable JSON)
   - 🔬 Raw API result (full response with paragraphs, polygons, tables)
8. **Download** — click "📥 Download All Results (JSON)" at the bottom for a single JSON export.

> **Tip:** Mistral uses its own OCR model regardless of prebuilt model selection.

## Architecture

```
Upload / URL → ┌── Azure Content Understanding (SDK) ──→ GPT summary
               ├── Azure Document Intelligence (SDK) ──→ GPT summary
               └── Mistral Document AI (REST OCR via Azure APIM)
                    ↓
               Side-by-side comparison, metrics, markdown preview & raw API output
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
