"""
Central configuration for the Benchmark App.
All Azure endpoints, keys, and model settings live here
and are loaded from environment variables / .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Azure Content Understanding ───────────────────────────────────────
CU_ENDPOINT = os.getenv("AZURE_CU_ENDPOINT", "")
CU_API_VERSION = os.getenv("AZURE_CU_API_VERSION", "2025-11-01")

# ─── Azure Blob Storage (used by Content Understanding for URL-based input) ──
STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT", "")
STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "cu-temp")
STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY", "")  # optional, falls back to DefaultAzureCredential

# ─── Azure Document Intelligence ───────────────────────────────────────
DOC_INTEL_ENDPOINT = os.getenv("DOC_INTELLIGENCE_ENDPOINT", "")
DOC_INTEL_KEY = os.getenv("DOC_INTELLIGENCE_KEY", "")

# ─── Azure OpenAI ──────────────────────────────────────────────────────
AZURE_OPENAI_BASE = os.getenv("AZURE_OPENAI_BASE", "")   # e.g. https://my-foundry-yaya.openai.azure.com
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")     # optional, falls back to DefaultAzureCredential
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

# Legacy full-URL vars (still supported as fallback)
GPT_ENDPOINT = os.getenv("GPT_ENDPOINT", "")   # full chat/completions URL
GPT_KEY = os.getenv("GPT_KEY", "")
GPT4_ENDPOINT = os.getenv("GPT4_ENDPOINT", "")  # full chat/completions URL
GPT4_KEY = os.getenv("GPT4_KEY", "")

# ─── LLM model choices ─────────────────────────────────────────────────
GPT_MODELS = {
    "gpt-4.1": "GPT-4.1",
    "gpt-5.1": "GPT-5.1",
}

def get_gpt_endpoint(deployment: str) -> str:
    """Build the chat/completions URL for a given deployment name."""
    if AZURE_OPENAI_BASE:
        base = AZURE_OPENAI_BASE.rstrip("/")
        return f"{base}/openai/deployments/{deployment}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    # Fallback to legacy env vars
    return GPT_ENDPOINT or GPT4_ENDPOINT

# ─── Mistral Doc AI (Azure-hosted OCR) ─────────────────────────────────
MISTRAL_DOC_AI_ENDPOINT = os.getenv("MISTRAL_DOC_AI_ENDPOINT", "")
MISTRAL_DOC_AI_KEY = os.getenv("MISTRAL_DOC_AI_KEY", "")
MISTRAL_DOC_AI_MODEL = os.getenv("MISTRAL_DOC_AI_MODEL", "mistral-document-ai-2505")

# ─── Prebuilt Analyzers available in Content Understanding ─────────────
PREBUILT_ANALYZERS = {
    "prebuilt-layout": "📐 Layout — Extracts tables, figures, and document structure",
    "prebuilt-read": "📖 Read — OCR for printed and handwritten text",
}

# ─── Document Intelligence prebuilt models ─────────────────────────────
DOC_INTEL_MODELS = {
    "prebuilt-layout": "prebuilt-layout",
    "prebuilt-read": "prebuilt-read",
}

# ─── Supported file types ──────────────────────────────────────────────
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".pdf"]
