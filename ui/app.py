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
from rag.ingestion.file_loader import SUPPORTED_EXTS
from rag.ingestion.upload import ingest_uploaded_file
from rag.pipeline.run import run_rag_tool
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
    return {"cfg": cfg, "embedder": embedder, "store": store}


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
    return ", ".join(sorted(SUPPORTED_EXTS))


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

    accepted = [ext.lstrip(".") for ext in SUPPORTED_EXTS]
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
        placeholder="e.g. How does Betopia pricing work?",
        key="query_input",
    )
    col_run, col_debug = st.columns([1, 4])
    with col_run:
        run = st.button("Run query", type="primary", disabled=not query_text.strip())
    with col_debug:
        include_debug = st.checkbox("Include debug block", value=debug_default)

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
                    debug=include_debug,
                ),
                config=res["cfg"],
                embedder=res["embedder"],
                store=res["store"],
            )
            elapsed_ms = int((time.time() - started) * 1000)
            out = pkg.to_dict()

            st.session_state.query_history.append(
                {"query": query_text, "result": out, "elapsedMs": elapsed_ms}
            )

            n_evidence = len(out["evidence"])
            n_citations = len(out["citations"])
            with st.chat_message("user", avatar="🧑"):
                st.write(query_text)
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(
                    f"**RAG layer returned an EvidencePackage** "
                    f"(in {elapsed_ms} ms).\n\n"
                    f"- rewrittenQuery: `{out['rewrittenQuery']}`\n"
                    f"- evidence chunks: **{n_evidence}**\n"
                    f"- distinct citations: **{n_citations}**\n"
                    f"- estimated tokens: **{out['usage']['estimatedTokens']}** / "
                    f"{out['usage']['maxTokens']}\n\n"
                    "Below is what the outer agent would see."
                )

            st.subheader("Evidence chunks (sorted by score)")
            if n_evidence == 0:
                st.warning(
                    "No evidence retrieved. Confirm the workspace was ingested "
                    "and the embedder dim matches the Qdrant collection."
                )
            for i, ev in enumerate(out["evidence"]):
                title = ev.get("title") or ev.get("sourceId")
                with st.expander(
                    f"#{i + 1} · score {ev['score']:.4f} · {title}", expanded=(i == 0)
                ):
                    st.markdown(f"**sourceId:** `{ev['sourceId']}`")
                    st.markdown(f"**chunkId:** `{ev['chunkId']}`")
                    st.markdown(f"**url:** `{ev['url']}`")
                    st.text(ev["text"])
                    st.json(ev["metadata"])

            st.subheader("Citations")
            st.json(out["citations"], expanded=False)

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
