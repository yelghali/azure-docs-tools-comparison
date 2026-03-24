"""
Azure Content Understanding service — prebuilt analyzers.
Uses the official azure-ai-contentunderstanding SDK with DefaultAzureCredential.
After extraction, sends extracted text to GPT-4.1 for a structured LLM summary.
"""

import time
import requests
from azure.identity import DefaultAzureCredential
from azure.ai.contentunderstanding import ContentUnderstandingClient
from config import (
    CU_ENDPOINT,
    GPT4_ENDPOINT, GPT4_KEY,
    AZURE_OPENAI_KEY,
)


class ContentUnderstandingService:
    """Wraps Azure Content Understanding SDK for document analysis."""

    def __init__(self):
        self.credential = DefaultAzureCredential()
        self.client = ContentUnderstandingClient(
            endpoint=CU_ENDPOINT,
            credential=self.credential,
        )
        self._token = None

    # ── Auth header for GPT-4 calls ───────────────────────────────────
    def _get_bearer_token(self):
        if self._token is None or time.time() > self._token.expires_on - 120:
            self._token = self.credential.get_token(
                "https://cognitiveservices.azure.com/.default"
            )
        return self._token.token

    # ── GPT-4.1 LLM summary (text-based) ─────────────────────────────
    def _gpt4_describe(self, filename: str, extracted_text: str,
                       system_prompt: str = None, user_prompt: str = None,
                       gpt_endpoint: str = None) -> str:
        """Send extracted document text to GPT for a structured summary."""
        endpoint = gpt_endpoint or GPT4_ENDPOINT
        if not endpoint:
            return "(GPT summary skipped — no endpoint configured)"

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
        key = GPT4_KEY or AZURE_OPENAI_KEY
        if key:
            headers["api-key"] = key
        else:
            headers["Authorization"] = f"Bearer {self._get_bearer_token()}"
        r = requests.post(endpoint, headers=headers, json=body, timeout=120)
        if not r.ok:
            raise RuntimeError(f"GPT HTTP {r.status_code}: {r.text[:500]}")
        return r.json()["choices"][0]["message"]["content"].strip()

    # ── Public API ──────────────────────────────────────────────────────
    def analyze(self, file_bytes: bytes, filename: str, analyzer_id: str,
                mime: str = "image/jpeg",
                system_prompt: str = None, user_prompt: str = None,
                gpt_endpoint: str = None) -> dict:
        """
        Full pipeline: SDK begin_analyze_binary → poll → parse → GPT-4 summary.
        """
        t0 = time.time()
        try:
            # Use SDK binary upload — no blob storage needed
            poller = self.client.begin_analyze_binary(
                analyzer_id=analyzer_id,
                binary_input=file_bytes,
                content_type=mime,
            )
            result = poller.result()

            # Parse SDK result into our format
            contents = result.get("contents", [])
            # Collect markdown from ALL content blocks (multi-page docs)
            all_md_parts = []
            fields_raw = {}
            tables_count = 0
            for block in contents:
                block_md = block.get("markdown", "")
                if block_md:
                    all_md_parts.append(block_md)
                block_fields = block.get("fields", {})
                fields_raw.update(block_fields)
                tables_count += len(block.get("tables", []))
            md = "\n\n".join(all_md_parts)

            flat = self._extract_field_values(fields_raw)
            confs = []
            self._collect_confidences(fields_raw, confs)
            avg_conf = round(sum(confs) / len(confs), 4) if confs else None

            # GPT LLM summary
            gpt_description = ""
            gpt_errors = []
            try:
                gpt_description = self._gpt4_describe(
                    filename, md, system_prompt, user_prompt,
                    gpt_endpoint=gpt_endpoint,
                )
            except Exception as e:
                gpt_errors.append(f"GPT Summary: {e}")

            dt = round(time.time() - t0, 2)
            return {
                "status": "success" if not gpt_errors else "partial",
                "time_seconds": dt,
                "markdown": md,
                "fields": flat,
                "field_count": len(fields_raw),
                "fields_with_values": len(flat),
                "tables_count": tables_count,
                "avg_confidence": avg_conf,
                "gpt_description": gpt_description,
                "errors": gpt_errors if gpt_errors else None,
                "raw_result": result,
            }
        except Exception as e:
            return {
                "status": "error",
                "time_seconds": round(time.time() - t0, 2),
                "error": str(e),
            }

    # ── Helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _extract_field_values(fields_dict: dict) -> dict:
        result = {}
        for name, obj in fields_dict.items():
            if not isinstance(obj, dict):
                continue
            if set(obj.keys()) <= {"type", "valueObject"} and "valueObject" in obj:
                sub = ContentUnderstandingService._extract_field_values(obj["valueObject"])
                if sub:
                    result[name] = sub
            elif set(obj.keys()) <= {"type"}:
                continue
            else:
                val = (
                    obj.get("valueString")
                    or obj.get("valueNumber")
                    or obj.get("valueDate")
                    or obj.get("content")
                    or obj.get("value")
                )
                if val is not None:
                    result[name] = val
                elif "valueObject" in obj:
                    sub = ContentUnderstandingService._extract_field_values(
                        obj["valueObject"]
                    )
                    if sub:
                        result[name] = sub
                elif "valueArray" in obj:
                    arr = []
                    for item in obj["valueArray"]:
                        if isinstance(item, dict) and "valueObject" in item:
                            sub = ContentUnderstandingService._extract_field_values(
                                item["valueObject"]
                            )
                            if sub:
                                arr.append(sub)
                        elif isinstance(item, dict):
                            v = (
                                item.get("valueString")
                                or item.get("content")
                                or item.get("value")
                            )
                            if v:
                                arr.append(v)
                    if arr:
                        result[name] = arr
        return result

    @staticmethod
    def _collect_confidences(obj, confs: list):
        if isinstance(obj, dict):
            if "confidence" in obj:
                confs.append(obj["confidence"])
            for v in obj.values():
                ContentUnderstandingService._collect_confidences(v, confs)
        elif isinstance(obj, list):
            for x in obj:
                ContentUnderstandingService._collect_confidences(x, confs)
