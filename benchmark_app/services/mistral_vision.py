"""
Mistral Doc AI — Azure-hosted OCR service via REST API.
Uses Mistral Document AI OCR endpoint (/v1/ocr) for document extraction.
Auth: API key or DefaultAzureCredential fallback.
"""

import time
import base64
import re
import logging
import httpx
from azure.identity import DefaultAzureCredential
from config import (
    MISTRAL_DOC_AI_ENDPOINT,
    MISTRAL_DOC_AI_KEY,
    MISTRAL_DOC_AI_MODEL,
)

logger = logging.getLogger(__name__)


class MistralVisionService:
    """Analyse documents with Azure-hosted Mistral Document AI OCR."""

    def __init__(self):
        self.api_key = MISTRAL_DOC_AI_KEY
        self.model = MISTRAL_DOC_AI_MODEL

        # Build the OCR endpoint URL
        base = MISTRAL_DOC_AI_ENDPOINT.rstrip("/")
        if "/ocr" in base:
            self.ocr_url = base
        elif "/v1/" in base:
            self.ocr_url = base.rsplit("/v1/", 1)[0] + "/v1/ocr"
        else:
            # Azure AI services style: append /providers/mistral/azure/ocr
            self.ocr_url = base + "/providers/mistral/azure/ocr"

        # DefaultAzureCredential for bearer-token fallback
        self.credential = DefaultAzureCredential()
        self._token = None

    # ── Auth helpers ────────────────────────────────────────────────────

    def _get_bearer_token(self) -> str:
        if self._token is None or time.time() > self._token.expires_on - 120:
            self._token = self.credential.get_token(
                "https://cognitiveservices.azure.com/.default"
            )
        return self._token.token

    def _auth_strategies(self) -> list[dict]:
        """Return a list of header dicts to try in order."""
        strategies = []
        if self.api_key:
            # Azure-style api-key header
            strategies.append({
                "name": "ApiKey",
                "headers": {
                    "Content-Type": "application/json",
                    "api-key": self.api_key,
                    "Accept": "application/json",
                },
            })
            # Bearer-key (Mistral platform style)
            strategies.append({
                "name": "BearerKey",
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                },
            })
        # Azure AD token fallback
        try:
            ad_token = self._get_bearer_token()
            strategies.append({
                "name": "AzureAD",
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {ad_token}",
                    "Accept": "application/json",
                },
            })
        except Exception as e:
            logger.debug("Azure AD token unavailable: %s", e)
        return strategies

    # ── OCR call ────────────────────────────────────────────────────────

    def _call_ocr(self, file_bytes: bytes, mime: str) -> dict:
        """Call the Mistral OCR endpoint with base64-encoded document."""
        b64 = base64.b64encode(file_bytes).decode("utf-8")
        body = {
            "model": self.model,
            "document": {
                "type": "document_url",
                "document_url": f"data:{mime};base64,{b64}",
            },
            "include_image_base64": False,
        }

        strategies = self._auth_strategies()
        if not strategies:
            raise RuntimeError(
                "No authentication available. Set MISTRAL_DOC_AI_KEY or run 'az login'."
            )

        last_error = None
        for strat in strategies:
            try:
                logger.info("Trying OCR: %s @ %s", strat["name"], self.ocr_url)
                resp = httpx.post(
                    self.ocr_url, json=body, headers=strat["headers"], timeout=180
                )
                if resp.status_code == 200:
                    logger.info("OCR success via %s", strat["name"])
                    return resp.json()
                last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
                logger.warning("OCR %s failed: %s", strat["name"], last_error)
            except Exception as exc:
                last_error = str(exc)
                logger.warning("OCR %s error: %s", strat["name"], exc)

        raise RuntimeError(f"All OCR auth methods failed. Last: {last_error}")

    # ── Public API ──────────────────────────────────────────────────────

    def analyze(
        self,
        file_bytes: bytes,
        filename: str,
        mime: str = "application/pdf",
        prompts: dict = None,
    ) -> dict:
        """
        Run Mistral Document AI OCR on a file and return structured results.
        Returns a result dict consistent with the other benchmark services.
        """
        t0 = time.time()
        errors = []
        full_markdown = ""
        fields = {}
        description = ""
        page_count = 0
        tables_count = 0

        try:
            raw = self._call_ocr(file_bytes, mime)
            raw_result = raw  # Keep full OCR response for raw display

            # Parse pages from response
            raw_pages = raw.get("pages") or raw.get("data", {}).get("pages", [])
            page_count = len(raw_pages)
            page_markdowns = []
            for rp in raw_pages:
                md = rp.get("markdown", "") or rp.get("content", "")
                page_markdowns.append(md)
                tables_count += len(re.findall(
                    r"(<table[\s\S]*?</table>)", md, re.IGNORECASE
                ))
                # Count markdown tables (lines starting with |)
                tables_count += len(re.findall(r"^\|.+\|$", md, re.MULTILINE))

            full_markdown = "\n\n---\n\n".join(page_markdowns)

            # Extract key-value fields from markdown
            fields = self._parse_fields(full_markdown)

            # Build a summary description from the first page
            description = self._extract_summary(full_markdown)

        except Exception as e:
            errors.append(f"Mistral Document AI OCR: {e}")
            raw_result = None

        dt = round(time.time() - t0, 2)
        error_str = "; ".join(errors) if errors else None
        return {
            "status": "success" if not errors else "error",
            "time_seconds": dt,
            "markdown": full_markdown,
            "fields": fields,
            "field_count": len(fields),
            "fields_with_values": len(fields),
            "tables_count": tables_count,
            "page_count": page_count,
            "avg_confidence": None,
            "mistral_description": description,
            "error": error_str,
            "errors": errors if errors else None,
            "raw_result": raw_result,
        }

    # ── Parsing helpers ─────────────────────────────────────────────────

    @staticmethod
    def _extract_summary(text: str) -> str:
        """Extract a summary from the OCR markdown (key lines or first 500 chars)."""
        lines = text.split("\n")
        summary_lines = []
        for line in lines:
            if any(kw in line.lower() for kw in [
                "invoice", "total", "date", "due", "bill to",
                "amount", "recipient", "issuer", "from:", "to:",
            ]):
                summary_lines.append(line.strip())
        if summary_lines:
            return "\n".join(summary_lines[:15])
        return text[:500] + ("..." if len(text) > 500 else "")

    @staticmethod
    def _parse_fields(text: str) -> dict:
        """Best-effort extraction of key-value pairs from OCR markdown."""
        fields = {}
        # Pattern: **Key**: Value  or  Key: Value  or  - Key: Value
        pattern = re.compile(
            r"(?:^|\n)\s*[-•*]*\s*\*{0,2}([A-Za-z0-9 /]+?)\*{0,2}\s*:\s*(.+)",
            re.MULTILINE,
        )
        for m in pattern.finditer(text):
            key = m.group(1).strip()
            val = m.group(2).strip().rstrip("|").strip()
            if key and val and 2 < len(key) < 40:
                fields[key] = val
        return fields
