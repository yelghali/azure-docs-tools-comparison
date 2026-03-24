"""
Azure Document Intelligence + GPT-4.1 (Argus) pipeline.
Step 1: Extract document with Doc Intelligence SDK.
Step 2: Send extracted text to GPT-4.1 for a rich LLM summary.
Auth: Uses DefaultAzureCredential or key-based auth via env vars.
"""

import io
import time
import requests
from azure.identity import DefaultAzureCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from config import (
    DOC_INTEL_ENDPOINT,
    DOC_INTEL_KEY,
    GPT_ENDPOINT,
    GPT_KEY,
)


class DocIntelGPTService:
    """Document Intelligence extraction + GPT (Argus) Vision description."""

    def __init__(self):
        # DefaultAzureCredential handles az login, managed identity, etc.
        self.credential = DefaultAzureCredential()
        self._token = None

        # Use API key if provided, otherwise use credential
        if DOC_INTEL_KEY:
            self.di_client = DocumentIntelligenceClient(
                endpoint=DOC_INTEL_ENDPOINT,
                credential=AzureKeyCredential(DOC_INTEL_KEY),
            )
        else:
            self.di_client = DocumentIntelligenceClient(
                endpoint=DOC_INTEL_ENDPOINT,
                credential=self.credential,
            )

    def _get_bearer_token(self) -> str:
        if self._token is None or time.time() > self._token.expires_on - 120:
            self._token = self.credential.get_token(
                "https://cognitiveservices.azure.com/.default"
            )
        return self._token.token

    # ── GPT-4.1 text-based call ──────────────────────────────────────
    def _gpt_describe(self, filename: str, extracted_text: str,
                      system_prompt: str = None, user_prompt: str = None) -> str:
        if not GPT_ENDPOINT:
            return "(GPT-4.1 description skipped — GPT_ENDPOINT not configured)"

        sys_content = system_prompt or (
            "You are an expert document analysis assistant. "
            "You analyse documents (invoices, quotes, purchase orders, etc.) "
            "and provide a concise structured description."
        )
        user_text = (user_prompt or (
            'Analyse this document "{filename}". '
            "Provide: document type, issuer, recipient, total amount, "
            "date, and any key information. Be concise (3-5 sentences)."
        )).format(filename=filename)

        user_content = user_text + f"\n\nExtracted document content:\n{extracted_text[:8000]}"

        body = {
            "messages": [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 2000,
            "temperature": 0.3,
        }
        headers = {"Content-Type": "application/json"}
        if GPT_KEY:
            headers["api-key"] = GPT_KEY
        else:
            headers["Authorization"] = f"Bearer {self._get_bearer_token()}"
        r = requests.post(GPT_ENDPOINT, headers=headers, json=body, timeout=180)
        if not r.ok:
            raise RuntimeError(f"GPT-4.1 HTTP {r.status_code}: {r.text[:500]}")
        return r.json()["choices"][0]["message"]["content"].strip()

    # ── Public API ──────────────────────────────────────────────────────
    def analyze(
        self,
        file_bytes: bytes,
        filename: str,
        model_id: str = "prebuilt-invoice",
        mime: str = "image/jpeg",
        prompts: dict = None,
    ) -> dict:
        """
        Run Doc Intelligence + GPT Vision on a document.
        
        Args:
            prompts: Optional dict with 'gpt_system' and 'gpt_user' keys for custom prompts
        
        Returns a result dict similar to Content Understanding's output.
        """
        t0 = time.time()
        errors = []
        prompts = prompts or {}

        # ── Step 1: Document Intelligence ───────────────────────────────
        di_result = {}
        di_fields = {}
        di_markdown = ""
        di_tables = 0
        di_confidence = None
        try:
            poller = self.di_client.begin_analyze_document(
                model_id,
                body=io.BytesIO(file_bytes),
                content_type="application/octet-stream",
            )
            result = poller.result()

            # Extract markdown / content
            di_markdown = result.content or ""

            # Extract fields
            if result.documents:
                for doc in result.documents:
                    if doc.fields:
                        for k, v in doc.fields.items():
                            if v and v.content:
                                di_fields[k] = v.content
                            elif v and v.value:
                                di_fields[k] = v.value

            # Tables
            di_tables = len(result.tables) if result.tables else 0

            # Average confidence
            confs = []
            if result.documents:
                for doc in result.documents:
                    if doc.confidence is not None:
                        confs.append(doc.confidence)
                    if doc.fields:
                        for v in doc.fields.values():
                            if v and v.confidence is not None:
                                confs.append(v.confidence)
            di_confidence = round(sum(confs) / len(confs), 4) if confs else None

            di_result = {
                "content": di_markdown[:2000],
                "fields": di_fields,
                "tables_count": di_tables,
                "avg_confidence": di_confidence,
            }
        except Exception as e:
            errors.append(f"DocIntel: {e}")

        # ── Step 2: GPT-4.1 Summary ─────────────────────────────────────
        gpt_description = ""
        try:
            gpt_description = self._gpt_describe(
                filename, di_markdown,
                system_prompt=prompts.get("gpt_system"),
                user_prompt=prompts.get("gpt_user"),
            )
        except Exception as e:
            errors.append(f"GPT-4.1: {e}")

        dt = round(time.time() - t0, 2)
        return {
            "status": "success" if not errors else "partial",
            "time_seconds": dt,
            "markdown": di_markdown,
            "fields": di_fields,
            "field_count": len(di_fields),
            "fields_with_values": len(di_fields),
            "tables_count": di_tables,
            "avg_confidence": di_confidence,
            "gpt_description": gpt_description,
            "errors": errors if errors else None,
            "di_detail": di_result,
        }
