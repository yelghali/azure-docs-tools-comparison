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

# ─── GPT (extraction LLM — full URL includes deployment + api-version) ──
GPT_ENDPOINT = os.getenv("GPT_ENDPOINT", "")   # full chat/completions URL
GPT_KEY = os.getenv("GPT_KEY", "")

# ─── GPT-4 (LLM summary for Content Understanding pipeline) ────────────
GPT4_ENDPOINT = os.getenv("GPT4_ENDPOINT", "")  # full chat/completions URL
GPT4_KEY = os.getenv("GPT4_KEY", "")

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
