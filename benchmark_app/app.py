"""
🏁 Document Processing Benchmark — Streamlit App

Compare Azure Content Understanding vs Document Intelligence + GPT (Argus) vs Mistral AI
on your own documents. Upload, pick a model, and get side-by-side results.
"""

import os
import sys
import json
import time
import requests
import streamlit as st
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Make sure our package is importable ────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import PREBUILT_ANALYZERS, SUPPORTED_EXTENSIONS, CU_ENDPOINT, GPT_MODELS, get_gpt_endpoint
from utils.comparison import (
    build_comparison_table,
    build_field_comparison,
    compute_summary_stats,
    get_mime_type,
)
from prompts import (
    GPT_SYSTEM_PROMPT, GPT_USER_PROMPT,
    MISTRAL_SYSTEM_PROMPT, MISTRAL_USER_PROMPT,
    CU_GPT_SYSTEM_PROMPT, CU_GPT_USER_PROMPT,
)

# ═══════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="📄 Doc Processing Benchmark",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main .block-container { max-width: 1400px; padding-top: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 1.2rem; color: white;
        text-align: center; margin-bottom: 0.5rem;
    }
    .metric-card h3 { margin: 0; font-size: 1.8rem; }
    .metric-card p  { margin: 0; font-size: 0.85rem; opacity: 0.85; }
    .pipeline-header {
        padding: 0.6rem 1rem; border-radius: 8px; font-weight: 600;
        margin-bottom: 0.8rem; font-size: 1.1rem;
    }
    .cu-header  { background: #e3f2fd; color: #1565c0; }
    .di-header  { background: #e8f5e9; color: #2e7d32; }
    .mis-header { background: #fff3e0; color: #e65100; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/compare.png",
        width=64,
    )
    st.title("⚙️ Settings")

    st.subheader("1️⃣  Prebuilt Model(s)")
    selected_models = st.multiselect(
        "Analyzer models",
        options=list(PREBUILT_ANALYZERS.keys()),
        default=["prebuilt-layout"],
        format_func=lambda x: PREBUILT_ANALYZERS[x],
        help="Select one or more models. CU and Doc Intelligence "
             "will run with each. Mistral uses its own OCR model.",
    )
    if not selected_models:
        st.warning("Select at least one model.")

    st.subheader("2️⃣  LLM Model")
    selected_llm = st.selectbox(
        "GPT model for summaries",
        options=list(GPT_MODELS.keys()),
        format_func=lambda x: GPT_MODELS[x],
        help="Used by CU and DocIntel pipelines for the text summary step.",
    )

    st.subheader("3️⃣  Pipelines to Run")
    # Check if Content Understanding is available (requires endpoint)
    cu_available = bool(CU_ENDPOINT)
    run_cu = st.checkbox(
        "🔵 Azure Content Understanding", 
        value=cu_available,
        disabled=not cu_available,
        help="Requires Azure CU endpoint configuration" if not cu_available else None,
    )
    if not cu_available:
        st.caption("⚠️ Content Understanding requires CU endpoint")
    
    run_di = st.checkbox(f"🟢 Document Intelligence + {GPT_MODELS.get(selected_llm, selected_llm)}", value=True)
    run_mi = st.checkbox("🟠 Mistral Document AI", value=True)

    st.divider()
    st.subheader("4️⃣  LLM Prompts")
    st.caption("Edit prompts below ↓ (in the main area)")

    st.divider()
    st.caption(
        "Selected pipelines run **in parallel** for maximum speed. "
        "Results are compared side-by-side."
    )
    st.divider()
    st.caption("Built for the Azure Content Understanding benchmark project.")

# ── Prompt Customization Section (main area — full width) ───────────────
with st.expander("📝 Customize Analysis Prompts", expanded=False):
    st.caption("Edit prompts to customize how LLMs analyze your documents. Use `{filename}` as placeholder for the document name.")
    
    prompt_col1, prompt_col2, prompt_col3 = st.columns(3)
    
    with prompt_col1:
        st.markdown("**🟢 GPT-4.1 (Argus) Prompts:**")
        gpt_system = st.text_area(
            "System prompt",
            value=GPT_SYSTEM_PROMPT,
            height=200,
            key="gpt_system",
            help="Instructions that define how the AI should behave"
        )
        gpt_user = st.text_area(
            "User prompt",
            value=GPT_USER_PROMPT,
            height=250,
            key="gpt_user",
            help="The actual analysis request. Use {filename} for document name"
        )
    
    with prompt_col2:
        st.markdown("**🟠 Mistral Document AI Prompts:**")
        mistral_system = st.text_area(
            "System prompt",
            value=MISTRAL_SYSTEM_PROMPT,
            height=200,
            key="mistral_system",
            help="Instructions for Mistral OCR and analysis"
        )
        mistral_user = st.text_area(
            "User prompt",
            value=MISTRAL_USER_PROMPT,
            height=250,
            key="mistral_user",
            help="The analysis request. Use {filename} for document name"
        )
    
    with prompt_col3:
        st.markdown("**🔵 Content Understanding GPT Prompts:**")
        cu_system = st.text_area(
            "System prompt",
            value=CU_GPT_SYSTEM_PROMPT,
            height=200,
            key="cu_system",
            help="Instructions for Content Understanding's GPT summary"
        )
        cu_user = st.text_area(
            "User prompt",
            value=CU_GPT_USER_PROMPT,
            height=250,
            key="cu_user",
            help="Summary request. Use {filename} for document name"
        )

# Store prompts in session state for access during benchmark
prompts = {
    "gpt_system": st.session_state.get("gpt_system", GPT_SYSTEM_PROMPT),
    "gpt_user": st.session_state.get("gpt_user", GPT_USER_PROMPT),
    "mistral_system": st.session_state.get("mistral_system", MISTRAL_SYSTEM_PROMPT),
    "mistral_user": st.session_state.get("mistral_user", MISTRAL_USER_PROMPT),
    "cu_system": st.session_state.get("cu_system", CU_GPT_SYSTEM_PROMPT),
    "cu_user": st.session_state.get("cu_user", CU_GPT_USER_PROMPT),
}

# ═══════════════════════════════════════════════════════════════════════
# Lazy-load services (cached so they're only initialized once)
# ═══════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🔌 Connecting to Azure Content Understanding…")
def get_cu_service():
    from services.content_understanding import ContentUnderstandingService
    return ContentUnderstandingService()


@st.cache_resource(show_spinner="🔌 Connecting to Document Intelligence + GPT…")
def get_di_service():
    from services.doc_intel_gpt import DocIntelGPTService
    return DocIntelGPTService()


@st.cache_resource(show_spinner="🔌 Connecting to Mistral Doc AI…")
def get_mi_service():
    from services.mistral_vision import MistralVisionService
    return MistralVisionService()


# ═══════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════
st.title("📄 Document Processing Benchmark")
st.markdown(
    "Compare **Azure Content Understanding**, **Document Intelligence + GPT-4.1 (Argus)**, "
    "and **Mistral Document AI** on your documents — all at once."
)

# ═══════════════════════════════════════════════════════════════════════
# Helper function to download images from URLs
# ═══════════════════════════════════════════════════════════════════════
def download_image_from_url(url: str) -> tuple[bytes, str]:
    """Download an image from a URL and return (bytes, filename)."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    # Extract filename from URL
    filename = url.split("/")[-1].split("?")[0]
    if not filename or "." not in filename:
        filename = "document.jpg"
    return response.content, filename

# ═══════════════════════════════════════════════════════════════════════
# File Upload + URL Input
# ═══════════════════════════════════════════════════════════════════════
input_tab1, input_tab2 = st.tabs(["📂 Upload Files", "🔗 Enter URLs"])

with input_tab1:
    uploaded_files = st.file_uploader(
        "Upload one or more documents",
        type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        help="Supported: JPG, PNG, BMP, TIFF, PDF",
    )

with input_tab2:
    url_input = st.text_area(
        "Enter document URLs (one per line)",
        placeholder="https://example.com/document1.jpg\nhttps://example.com/document2.png",
        help="Enter URLs of images/documents to process",
    )
    
# Combine uploaded files and URL downloads
documents = []

if uploaded_files:
    for f in uploaded_files:
        documents.append({
            "name": f.name,
            "bytes": f.getvalue(),
            "type": f.type,
            "source": "upload"
        })

if url_input:
    urls = [u.strip() for u in url_input.strip().split("\n") if u.strip()]
    if urls:
        with st.spinner(f"Downloading {len(urls)} document(s) from URLs..."):
            for url in urls:
                try:
                    file_bytes, filename = download_image_from_url(url)
                    documents.append({
                        "name": filename,
                        "bytes": file_bytes,
                        "type": get_mime_type(filename),
                        "source": "url",
                        "url": url
                    })
                except Exception as e:
                    st.error(f"Failed to download {url}: {e}")

if not documents:
    st.info("👆 Upload documents or enter URLs to get started.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════
# Preview documents
# ═══════════════════════════════════════════════════════════════════════
with st.expander(f"📎 Documents to process ({len(documents)})", expanded=False):
    cols = st.columns(min(len(documents), 5))
    for idx, doc in enumerate(documents):
        with cols[idx % len(cols)]:
            mime_type = doc.get("type", "")
            if mime_type and mime_type.startswith("image"):
                try:
                    st.image(doc["bytes"], caption=doc["name"])
                except Exception:
                    st.write(f"🖼️ {doc['name']} ({len(doc['bytes']) / 1024:.0f} KB)")
            else:
                st.write(f"📄 {doc['name']} ({len(doc['bytes']) / 1024:.0f} KB)")
            if doc.get("source") == "url":
                st.caption("From URL")

# ═══════════════════════════════════════════════════════════════════════
# 🚀 Run Benchmark
# ═══════════════════════════════════════════════════════════════════════
if st.button("🚀  Run Benchmark", type="primary", use_container_width=True):
    if not any([run_cu, run_di, run_mi]):
        st.error("Please select at least one pipeline in the sidebar.")
        st.stop()
    if not selected_models and (run_cu or run_di):
        st.error("Please select at least one prebuilt model in the sidebar.")
        st.stop()

    all_doc_results = []
    progress = st.progress(0, text="Starting benchmark…")
    total_tasks = len(documents)

    for file_idx, doc in enumerate(documents):
        file_bytes = doc["bytes"]
        filename = doc["name"]
        mime = get_mime_type(filename)

        st.divider()
        st.subheader(f"📄 {filename}")

        # Show document preview
        preview_col, results_col = st.columns([1, 3])
        with preview_col:
            if mime.startswith("image"):
                try:
                    st.image(file_bytes, caption=filename)
                except Exception:
                    st.write(f"🖼️ {filename} ({len(file_bytes) / 1024:.0f} KB)")
            else:
                st.write(f"📄 {filename} ({len(file_bytes) / 1024:.0f} KB)")

        # ── Run pipelines in parallel ───────────────────────────────────
        results = {}
        futures = {}
        multi_model = len(selected_models) > 1
        gpt_endpoint = get_gpt_endpoint(selected_llm)
        llm_label = GPT_MODELS.get(selected_llm, selected_llm)
        with ThreadPoolExecutor(max_workers=6) as executor:
            for model_id in selected_models:
                short_label = model_id.replace("prebuilt-", "").capitalize()
                suffix = f" [{short_label}]" if multi_model else ""
                if run_cu:
                    pipeline_label = f"🔵 Content Understanding{suffix}"
                    try:
                        svc = get_cu_service()
                        futures[
                            executor.submit(svc.analyze, file_bytes, filename, model_id, mime,
                                            prompts["cu_system"], prompts["cu_user"],
                                            gpt_endpoint)
                        ] = pipeline_label
                    except Exception as e:
                        results[pipeline_label] = {
                            "status": "error",
                            "error": f"Service init failed: {e}",
                            "time_seconds": 0,
                        }
                if run_di:
                    pipeline_label = f"🟢 DocIntel + {llm_label}{suffix}"
                    try:
                        svc = get_di_service()
                        futures[
                            executor.submit(svc.analyze, file_bytes, filename, model_id, mime,
                                            {"gpt_system": prompts["gpt_system"], "gpt_user": prompts["gpt_user"]},
                                            gpt_endpoint)
                        ] = pipeline_label
                    except Exception as e:
                        results[pipeline_label] = {
                            "status": "error",
                            "error": f"Service init failed: {e}",
                            "time_seconds": 0,
                        }
            if run_mi:
                pipeline_label = "🟠 Mistral Document AI"
                try:
                    svc = get_mi_service()
                    futures[
                        executor.submit(svc.analyze, file_bytes, filename, mime,
                                        {"mistral_system": prompts["mistral_system"], "mistral_user": prompts["mistral_user"]})
                    ] = pipeline_label
                except Exception as e:
                    results[pipeline_label] = {
                        "status": "error",
                        "error": f"Service init failed: {e}",
                        "time_seconds": 0,
                    }

            with results_col:
                status_placeholder = st.empty()
                status_placeholder.info(
                    f"⏳ Running {len(futures)} pipeline(s) in parallel…"
                )

            for future in as_completed(futures):
                pipeline_name = futures[future]
                try:
                    results[pipeline_name] = future.result()
                except Exception as e:
                    results[pipeline_name] = {
                        "status": "error",
                        "error": str(e),
                        "time_seconds": 0,
                    }

        with results_col:
            status_placeholder.success(
                f"✅ All pipelines completed for {filename}"
            )

        # ── Metric cards ────────────────────────────────────────────────
        metric_cols = st.columns(len(results))
        for col, (pname, res) in zip(metric_cols, results.items()):
            with col:
                t = res.get("time_seconds", "—")
                fv = res.get("fields_with_values", 0)
                conf = res.get("avg_confidence")
                conf_str = f"{conf:.1%}" if conf else "N/A"
                st.markdown(
                    f"""<div class="metric-card">
                    <p>{pname}</p>
                    <h3>{t}s</h3>
                    <p>⏱ Time</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.metric("Fields extracted", fv)
                st.metric("Avg confidence", conf_str)

        # ── Comparison table ────────────────────────────────────────────
        st.markdown("#### 📊 Pipeline Comparison")
        comp_rows = build_comparison_table(results)
        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        # ── Field-by-field comparison ───────────────────────────────────
        field_comp = build_field_comparison(results)
        if field_comp:
            st.markdown("#### 🔍 Field-by-Field Comparison")
            df_fields = pd.DataFrame(field_comp).T
            df_fields.index.name = "Field"
            st.dataframe(df_fields, use_container_width=True)

        # ── Detailed outputs (tabs) ─────────────────────────────────────
        st.markdown("#### 📝 Detailed Outputs")
        tabs = st.tabs(list(results.keys()))
        for tab, (pname, res) in zip(tabs, results.items()):
            with tab:
                if res.get("status") == "error":
                    # Check both 'error' (singular) and 'errors' (list) keys
                    err_msg = res.get("error")
                    if not err_msg and res.get("errors"):
                        err_msg = "; ".join(res["errors"])
                    st.error(f"❌ Error: {err_msg or 'Unknown error'}")
                    continue

                # GPT / Mistral description
                desc = res.get("gpt_description") or res.get("mistral_description")
                if desc:
                    st.markdown("**🤖 AI Description:**")
                    st.info(desc)

                # Markdown output
                md = res.get("markdown", "")
                if md:
                    with st.expander("📄 Markdown output", expanded=True):
                        raw_tab, rendered_tab = st.tabs(["📝 Raw Markdown", "👁️ Rendered Preview"])
                        with raw_tab:
                            st.code(md[:5000], language="markdown")
                        with rendered_tab:
                            st.markdown(md[:5000], unsafe_allow_html=True)

                # Raw fields
                fields = res.get("fields", {})
                if fields:
                    with st.expander(f"📋 Extracted fields ({len(fields)})", expanded=False):
                        st.json(fields)

                # Raw API result (paragraphs, polygons, tables, etc.)
                raw = res.get("raw_result")
                if raw:
                    with st.expander("🔬 Raw API Result (paragraphs, polygons, tables…)", expanded=False):
                        st.code(json.dumps(raw, indent=2, default=str)[:20000], language="json")

                # Errors / warnings
                errs = res.get("errors")
                if errs:
                    for e in errs:
                        st.warning(e)

        # Store for batch summary
        all_doc_results.append({"filename": filename, "results": results})
        progress.progress(
            (file_idx + 1) / total_tasks,
            text=f"Processed {file_idx + 1}/{total_tasks} documents",
        )

    # ═══════════════════════════════════════════════════════════════════
    # 📈 Batch Summary (if multiple docs)
    # ═══════════════════════════════════════════════════════════════════
    if len(all_doc_results) > 1:
        st.divider()
        st.header("📈 Batch Summary")
        summary = compute_summary_stats(all_doc_results)
        if summary:
            st.dataframe(
                pd.DataFrame(summary).T.rename_axis("Pipeline"),
                use_container_width=True,
            )

        # Chart: time comparison
        st.markdown("#### ⏱ Processing Time per Document")
        chart_data = []
        for doc in all_doc_results:
            for pipeline, res in doc["results"].items():
                chart_data.append(
                    {
                        "Document": doc["filename"],
                        "Pipeline": pipeline,
                        "Time (s)": res.get("time_seconds", 0),
                    }
                )
        if chart_data:
            df_chart = pd.DataFrame(chart_data)
            st.bar_chart(
                df_chart.pivot(index="Document", columns="Pipeline", values="Time (s)")
            )

    # ── Download results ────────────────────────────────────────────────
    st.divider()
    # Strip raw_result from download (too large) but keep everything else
    download_data = []
    for doc in all_doc_results:
        clean_results = {}
        for pname, res in doc["results"].items():
            clean_results[pname] = {k: v for k, v in res.items() if k != "raw_result"}
        download_data.append({"filename": doc["filename"], "results": clean_results})
    results_json = json.dumps(download_data, indent=2, ensure_ascii=False, default=str)
    st.download_button(
        "📥 Download Full Results (JSON)",
        data=results_json,
        file_name="benchmark_results.json",
        mime="application/json",
        use_container_width=True,
    )

    st.balloons()
