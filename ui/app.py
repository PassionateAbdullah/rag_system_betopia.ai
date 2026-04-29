"""Streamlit test UI for the Betopia RAG MVP.

Two tabs:
  1. Upload & Ingest: drop files; auto-runs ingest_uploaded_file().
  2. Query: ask a question; returns the raw EvidencePackage that an outer
     agent would consume.

This is a developer evaluation harness, not the production chat UI. It hits
the same code paths the backend will: ingest_uploaded_file() and
run_rag_tool(). No LLM is called — the RAG layer does not generate answers.

Run:
    streamlit run ui/app.py
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import traceback
from typing import Any

import streamlit as st

from rag.config import load_config
from rag.embeddings.default_provider import build_embedding_provider
from rag.errors import IngestionError
from rag.ingestion.file_loader import supported_exts
from rag.ingestion.upload import ingest_uploaded_file
from rag.pipeline.run import run_rag_tool
from rag.storage import build_postgres_store
from rag.types import IngestUploadInput, RagInput
from rag.vector.qdrant_client import QdrantStore

st.set_page_config(page_title="Betopia RAG MVP — Test Harness", layout="wide")


# ---------- shared resources (built once per session) ----------

@st.cache_resource(show_spinner="Loading config + embedding model...")
def get_resources() -> dict[str, Any]:
    cfg = load_config()
    embedder = build_embedding_provider(cfg)
    store = QdrantStore(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        collection=cfg.qdrant_collection,
        vector_size=embedder.dim,
    )
    store.ensure_collection()
    pg = None
    try:
        pg = build_postgres_store(cfg)
        if pg is not None:
            pg.migrate()
    except Exception:
        pg = None
    return {"cfg": cfg, "embedder": embedder, "store": store, "postgres": pg}


def _qdrant_status(store: QdrantStore) -> dict[str, Any]:
    return store.info()


# ---------- session state ----------

if "ingest_log" not in st.session_state:
    st.session_state.ingest_log: list[dict[str, Any]] = []
if "query_history" not in st.session_state:
    st.session_state.query_history: list[dict[str, Any]] = []


# ---------- sidebar ----------

with st.sidebar:
    st.title("Betopia RAG — Test")
    st.caption("Evaluation harness. RAG returns EvidencePackage only. No LLM answer.")

    st.subheader("Workspace")
    workspace_id = st.text_input("workspaceId", value="default")
    user_id = st.text_input("userId", value="ui_user")

    st.subheader("Query defaults")
    max_chunks = st.slider("maxChunks", 1, 20, 8)
    max_tokens = st.slider("maxTokens", 500, 8000, 4000, step=250)
    debug_default = st.checkbox("debug payload", value=True)

    st.divider()

    res = get_resources()
    cfg = res["cfg"]
    info = _qdrant_status(res["store"])
    st.subheader("Qdrant")
    st.code(
        f"url:        {cfg.qdrant_url}\n"
        f"collection: {cfg.qdrant_collection}\n"
        f"points:     {info.get('vectors_count', '?')}\n"
        f"status:     {info.get('status', '?')}",
        language="text",
    )
    pg = res.get("postgres")
    st.subheader("Postgres")
    if pg is None:
        st.code("disabled (POSTGRES_URL unset) — Qdrant-only mode", language="text")
    else:
        pg_info = pg.info()
        st.code(
            f"documents: {pg_info.get('documents', '?')}\n"
            f"chunks:    {pg_info.get('chunks', '?')}\n"
            f"status:    {'ok' if pg_info.get('ok') else 'error'}",
            language="text",
        )
    st.subheader("Pipeline")
    st.code(
        f"reranker:    {cfg.reranker_provider}\n"
        f"compression: {cfg.compression_provider}\n"
        f"hybrid:      {bool(pg is not None and cfg.enable_hybrid_retrieval)}\n"
        f"rewriter:    {cfg.query_rewriter}",
        language="text",
    )
    st.subheader("Embedding")
    st.code(
        f"provider: {cfg.embedding_provider}\n"
        f"model:    {res['embedder'].model_name}\n"
        f"dim:      {res['embedder'].dim}",
        language="text",
    )


# ---------- helpers ----------

def _save_upload_to_tempfile(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1].lower() or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def _supported_exts_label() -> str:
    return ", ".join(sorted(supported_exts()))


# ---------- tabs ----------

tab_upload, tab_query, tab_history = st.tabs(
    ["📤 Upload & Ingest", "💬 Query", "🕘 Session log"]
)


# === Upload tab =============================================================

with tab_upload:
    st.header("Upload & auto-ingest")
    st.write(
        "Drop one or more files. Each file is saved to a temp path and "
        "auto-ingested through `ingest_uploaded_file`. Supported: "
        f"`{_supported_exts_label()}`."
    )

    accepted = [ext.lstrip(".") for ext in supported_exts()]
    uploads = st.file_uploader(
        "Files",
        type=accepted,
        accept_multiple_files=True,
    )

    if uploads:
        if st.button(f"Ingest {len(uploads)} file(s)", type="primary"):
            for upload in uploads:
                started = time.time()
                tmp_path: str | None = None
                with st.chat_message("system", avatar="📥"):
                    st.write(f"Ingesting **{upload.name}**...")
                try:
                    tmp_path = _save_upload_to_tempfile(upload)
                    result = ingest_uploaded_file(
                        IngestUploadInput(
                            file_path=tmp_path,
                            workspace_id=workspace_id,
                            user_id=user_id,
                            title=upload.name,
                            url=f"upload://{upload.name}",
                        ),
                        config=res["cfg"],
                        embedder=res["embedder"],
                        store=res["store"],
                        postgres=res.get("postgres"),
                    )
                    elapsed_ms = int((time.time() - started) * 1000)
                    payload = result.to_dict()
                    payload["_elapsedMs"] = elapsed_ms
                    st.session_state.ingest_log.append(
                        {"ok": True, "name": upload.name, "result": payload}
                    )
                    with st.chat_message("assistant", avatar="✅"):
                        st.markdown(
                            f"**Ingested `{upload.name}`** "
                            f"→ {result.chunks_created} chunks "
                            f"in {elapsed_ms} ms"
                        )
                        st.json(payload, expanded=False)
                except IngestionError as e:
                    err = e.to_dict()
                    st.session_state.ingest_log.append(
                        {"ok": False, "name": upload.name, "error": err}
                    )
                    with st.chat_message("assistant", avatar="❌"):
                        st.error(
                            f"**Ingest failed for `{upload.name}`** "
                            f"at stage `{e.stage}`: {e.reason}"
                        )
                        st.json(err)
                except Exception as e:
                    st.session_state.ingest_log.append(
                        {"ok": False, "name": upload.name, "error": str(e)}
                    )
                    with st.chat_message("assistant", avatar="❌"):
                        st.error(f"Unexpected error: {e}")
                        st.code(traceback.format_exc())
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

    st.divider()
    st.subheader("Session ingest log")
    if not st.session_state.ingest_log:
        st.caption("No uploads yet.")
    else:
        for entry in reversed(st.session_state.ingest_log):
            label = "✅" if entry["ok"] else "❌"
            with st.expander(f"{label}  {entry['name']}"):
                st.json(entry.get("result") or entry.get("error"))


# === Query tab ==============================================================

with tab_query:
    st.header("Query the knowledge base")
    st.write(
        "Returns the **EvidencePackage** that the outer agent would consume. "
        "No natural-language answer is generated here — that is the agent's job."
    )

    query_text = st.text_input(
        "Query",
        placeholder="e.g. what is the system vision?",
        key="query_input",
    )
    run = st.button("Run query", type="primary", disabled=not query_text.strip())

    if run and query_text.strip():
        started = time.time()
        try:
            pkg = run_rag_tool(
                RagInput(
                    query=query_text,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    max_tokens=max_tokens,
                    max_chunks=max_chunks,
                    debug=debug_default,
                ),
                config=res["cfg"],
                embedder=res["embedder"],
                store=res["store"],
                postgres=res.get("postgres"),
            )
            elapsed_ms = int((time.time() - started) * 1000)
            out = pkg.to_dict()

            st.session_state.query_history.append(
                {"query": query_text, "result": out, "elapsedMs": elapsed_ms}
            )

            n_evidence = len(out["evidence"])
            n_context = len(out["context_for_agent"])
            n_citations = len(out.get("citations") or [])
            confidence = out.get("confidence", 0.0)
            gaps = out.get("coverage_gaps", []) or []
            usage = out.get("usage") or {}
            debug_payload = out.get("debug") or {}

            with st.chat_message("user", avatar="🧑"):
                st.write(query_text)

            rewrote = out["rewritten_query"] != out["original_query"]
            with st.chat_message("assistant", avatar="🧠"):
                msg = (
                    f"**EvidencePackage ready** in {elapsed_ms} ms · "
                    f"confidence **{confidence:.2f}**\n\n"
                )
                if rewrote:
                    msg += (
                        f"- rewrote: `{out['original_query']}` → "
                        f"`{out['rewritten_query']}`\n"
                    )
                msg += (
                    f"- context items for agent: **{n_context}**\n"
                    f"- raw evidence chunks: **{n_evidence}**\n"
                    f"- citations: **{n_citations}**\n"
                )
                if usage:
                    msg += (
                        f"- estimated tokens: **{usage.get('estimatedTokens', '?')}**"
                        f" / {usage.get('maxTokens', '?')}\n"
                    )
                if gaps:
                    msg += f"- coverage gaps: **{len(gaps)}**\n"
                st.markdown(msg)

                if gaps:
                    with st.expander("Coverage gaps", expanded=False):
                        for g in gaps:
                            st.write(f"• {g}")

            # --- context_for_agent: what the outer agent will actually see ---
            st.subheader("Context for agent (compressed)")
            if n_context == 0:
                st.warning(
                    "No context selected. Confirm the workspace was ingested "
                    "and the embedder dim matches the Qdrant collection."
                )
            for i, ctx in enumerate(out["context_for_agent"]):
                section = ctx.get("sectionTitle") or "(no section)"
                title = ctx.get("title") or ctx.get("sourceId")
                with st.expander(
                    f"#{i + 1} · score {ctx['score']:.4f} · "
                    f"{section} · {title}",
                    expanded=(i == 0),
                ):
                    st.markdown(f"**sectionTitle:** `{section}`")
                    st.markdown(f"**sourceId:** `{ctx['sourceId']}`")
                    st.markdown(f"**chunkId:** `{ctx['chunkId']}`")
                    st.markdown(f"**url:** `{ctx['url']}`")
                    st.text(ctx["text"])

            # --- raw evidence trail ---
            with st.expander("Raw evidence trail (pre-compression)"):
                for i, ev in enumerate(out["evidence"]):
                    section = ev.get("sectionTitle") or "(no section)"
                    st.markdown(
                        f"**#{i + 1}** · vector `{ev['score']:.4f}` · "
                        f"rerank `{ev['rerankScore']:.4f}` · {section}"
                    )
                    st.text(ev["text"][:600] + ("..." if len(ev["text"]) > 600 else ""))
                    sigs = (ev.get("metadata") or {}).get("rerankSignals")
                    if sigs:
                        st.caption(f"signals: {sigs}")
                    st.divider()

            # --- citations (the agent renders these next to its answer) ---
            if n_citations:
                with st.expander(f"Citations ({n_citations})", expanded=False):
                    for i, cit in enumerate(out["citations"]):
                        section = cit.get("section") or "(no section)"
                        page = cit.get("page")
                        page_str = f" · page {page}" if page is not None else ""
                        st.markdown(
                            f"**#{i + 1}** `{cit['sourceId']}` / "
                            f"`{cit['chunkId']}` · {section}{page_str}"
                        )
                        st.caption(cit.get("title") or "")
                        st.caption(cit.get("url") or "")

            # --- retrieval trace ---
            with st.expander("Retrieval trace"):
                st.json(out.get("retrieval_trace") or {})

            # --- debug latency breakdown ---
            if debug_payload:
                with st.expander("Debug payload", expanded=False):
                    st.json(debug_payload)

            # --- raw JSON ---
            st.subheader("Full EvidencePackage JSON")
            st.code(json.dumps(out, indent=2, ensure_ascii=False), language="json")

        except Exception as e:
            st.error(f"Query failed: {e}")
            st.code(traceback.format_exc())


# === History tab ============================================================

with tab_history:
    st.header("Session log")
    st.caption("Lives only in this Streamlit session.")
    st.subheader(f"Queries ({len(st.session_state.query_history)})")
    for i, entry in enumerate(reversed(st.session_state.query_history)):
        with st.expander(
            f"#{len(st.session_state.query_history) - i} · {entry['query']}"
        ):
            st.caption(f"{entry['elapsedMs']} ms")
            st.json(entry["result"])

    st.subheader(f"Ingests ({len(st.session_state.ingest_log)})")
    for entry in reversed(st.session_state.ingest_log):
        label = "✅" if entry["ok"] else "❌"
        with st.expander(f"{label}  {entry['name']}"):
            st.json(entry.get("result") or entry.get("error"))
