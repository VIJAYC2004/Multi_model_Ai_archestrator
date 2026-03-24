# app.py

import asyncio
from typing import List, Dict

import streamlit as st

from orchestrator import query_models_with_web
from models_config import (
    MODELS,
    get_default_sources,
    get_task_models,
    classify_task_from_question,
)
from multimodal_utils import build_multimodal_context
from rag_store import index_documents, query_rag  # NEW


st.set_page_config(
    page_title="Multi‑Modal Multi‑Model Web‑Search AI (Local)",
    layout="wide",
)

st.title("🧠 Multi‑Modal Multi‑Model Web‑Search AI (Local, Open‑Source)")

st.markdown(
    "This app:\n"
    "- Uses **local open‑source LLMs via Ollama**.\n"
    "- Supports **text, audio, images, video, and files** as inputs.\n"
    "- Does **live web search** to fetch information.\n"
    "- Lets you **index your own PDFs/TXTs and query them (RAG)**.\n"
    "- Queries **multiple models in parallel** and returns one summarized answer.\n"
)

tab_web, tab_docs = st.tabs(["🌐 Web + Multi‑LLM", "📂 My Documents (RAG)"])

# -----------------------------------------------------------------------------
# 🌐 Web + Multi‑LLM tab
# -----------------------------------------------------------------------------

with tab_web:
    # Sidebar – settings + Perplexity‑style “New Thread” and modes
    with st.sidebar:
        st.header("Settings")

        # Mode selector (like Fast / Balanced / Max)
        mode = st.radio(
            "Mode",
            ["Fast", "Balanced", "Max quality"],
            index=1,
            help="Fast = fewer/smaller models; Max = more models, slower, better quality.",
        )

        # Derive default model list from mode
        all_model_names = [m.name for m in MODELS]

        if mode == "Fast":
            default_source_models: List[str] = [m.name for m in MODELS if m.tier == "fast"]
        elif mode == "Max quality":
            default_source_models = [m.name for m in MODELS if m.tier == "max"]
            if not default_source_models:
                # fallback to all if no 'max' tier defined
                default_source_models = all_model_names
        else:  # Balanced
            default_source_models = get_default_sources()

        selected_models = st.multiselect(
            "Source models (queried in parallel)",
            options=all_model_names,
            default=[m for m in default_source_models if m in all_model_names],
            help="These models receive the web‑augmented prompt.",
        )

        max_pages = st.slider(
            "Number of web pages",
            min_value=1,
            max_value=5,
            value=3,
            help="How many top search results to fetch and use as context.",
        )

        show_sources = st.checkbox("Show web sources box", value=True)
        show_per_model = st.checkbox("Show per‑model answers (debug)", value=False)

        st.markdown("---")
        answer_style = st.selectbox(
            "Answer style",
            ["Default", "Beginner explanation", "Step‑by‑step", "Very short"],
            index=0,
            help="Controls how the final answer is written.",
        )

        st.markdown("---")
        st.subheader("Thread")

        if st.button("🆕 New Thread"):
            st.session_state.chat_messages = []
            st.session_state.media_state = {}
            st.rerun()

        st.markdown("---")
        st.markdown("**Available models:**")
        for m in MODELS:
            st.markdown(f"- `{m.name}` — {m.role}, {m.tier}: {m.description}")

    # State – chat messages + media
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []  # list of {"role": "user"/"assistant", "content": str}

    if "media_state" not in st.session_state:
        st.session_state.media_state = {
            "audio_file": None,
            "image_files": None,
            "video_file": None,
            "other_files": None,
            "extra_text": "",
        }

    # Advanced multimodal inputs
    with st.expander("🔧 Advanced inputs: audio / images / video / files", expanded=False):
        st.subheader("Your input")

        col_text, col_audio = st.columns([2, 1])

        with col_text:
            text_prompt = st.text_area(
                "Optional extra text for the next question (besides chat input):",
                height=100,
            )

        with col_audio:
            audio_file = st.file_uploader(
                "Audio / mic recording (WAV/MP3/OGG/M4A)",
                type=["wav", "mp3", "ogg", "m4a"],
                accept_multiple_files=False,
                help="Upload a recording from your microphone.",
            )

        col_img, col_vid = st.columns(2)

        with col_img:
            image_files = st.file_uploader(
                "Images",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                help="Screenshots, photos, etc.",
            )

        with col_vid:
            video_file = st.file_uploader(
                "Video",
                type=["mp4", "mov", "mkv"],
                accept_multiple_files=False,
                help="Short video (a few seconds recommended).",
            )

        other_files = st.file_uploader(
            "Other files (PDF, TXT, MD)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="Documents to use as extra context.",
        )

        st.session_state.media_state = {
            "audio_file": audio_file,
            "image_files": image_files,
            "video_file": video_file,
            "other_files": other_files,
            "extra_text": text_prompt.strip() if text_prompt else "",
        }

    # Chat UI
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask anything (web + multi‑model)...")

    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        media_state = st.session_state.media_state

        multimodal_context = build_multimodal_context(
            audio_file=media_state.get("audio_file"),
            image_files=media_state.get("image_files"),
            video_file=media_state.get("video_file"),
            other_files=media_state.get("other_files"),
        )

        extra_text = media_state.get("extra_text") or ""

        base_text = user_input
        if extra_text:
            base_text += "\n\nAdditional notes from user:\n" + extra_text

        if multimodal_context:
            user_question_with_media = (
                f"{base_text}\n\n"
                f"Additional context extracted from the user's media inputs:\n"
                f"{multimodal_context}"
            )
        else:
            user_question_with_media = base_text

        task = classify_task_from_question(user_question_with_media)
        if selected_models:
            used_models = selected_models
        else:
            used_models = get_task_models(task, mode=mode)

        style_instruction = ""
        if answer_style == "Beginner explanation":
            style_instruction = (
                "\n\nWrite the answer for a beginner. Avoid jargon and explain concepts simply."
            )
        elif answer_style == "Step‑by‑step":
            style_instruction = (
                "\n\nExplain the answer step‑by‑step with clear numbered points."
            )
        elif answer_style == "Very short":
            style_instruction = (
                "\n\nKeep the answer very short (2‑3 sentences max)."
            )

        styled_question = user_question_with_media + style_instruction

        with st.chat_message("assistant"):
            with st.spinner("Searching web and querying local models..."):
                result = asyncio.run(
                    query_models_with_web(
                        user_question=styled_question,
                        source_models=used_models,
                        max_pages=max_pages,
                    )
                )

                final_answer = result["final_answer"]
                search_results = result.get("search_results", [])
                per_model = result.get("per_model", {})

                # Follow‑up suggestions
                followups_prompt = (
                    "Based on the user's question and the answer above, "
                    "suggest 3 short, useful follow‑up questions the user could ask next. "
                    "Return them as a plain list separated by newline."
                )

                try:
                    followups_result = asyncio.run(
                        query_models_with_web(
                            user_question=styled_question + "\n\n" + followups_prompt,
                            source_models=used_models,
                            max_pages=1,
                        )
                    )
                    followups_raw = followups_result["final_answer"]
                    followup_lines = [
                        line.strip("-• ").strip()
                        for line in followups_raw.splitlines()
                        if line.strip()
                    ][:3]
                except Exception:
                    followup_lines = []

                st.markdown(final_answer)

                if followup_lines:
                    st.markdown("**Follow‑up questions:**")
                    cols_fu = st.columns(len(followup_lines))
                    for fu, c in zip(followup_lines, cols_fu):
                        if fu:
                            with c:
                                if st.button(fu):
                                    st.session_state.chat_messages.append(
                                        {"role": "user", "content": fu}
                                    )
                                    st.rerun()

                if show_sources and search_results:
                    with st.expander("Sources", expanded=True):
                        for i, r in enumerate(search_results, start=1):
                            url = r.get("url", "")
                            title = r.get("title", url)
                            if url:
                                st.markdown(f"{i}. [{title}]({url})")

                if show_per_model and per_model:
                    st.markdown("---")
                    st.markdown("**Per‑model answers (debug view):**")
                    cols = st.columns(max(1, len(per_model)))
                    for (name, ans), c in zip(per_model.items(), cols):
                        with c:
                            st.markdown(f"##### `{name}`")
                            st.markdown(ans)

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": final_answer}
        )

# -----------------------------------------------------------------------------
# 📂 My Documents (RAG) tab
# -----------------------------------------------------------------------------

with tab_docs:
    st.subheader("Ask questions about your documents (local RAG)")

    if "rag_indexed" not in st.session_state:
        st.session_state.rag_indexed = False

    doc_files = st.file_uploader(
        "Upload PDFs / TXT / MD to index",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Index documents into RAG") and doc_files:
        with st.spinner("Indexing documents..."):
            added = index_documents(doc_files, namespace="user_docs")
        st.success(f"Indexed {added} chunks from {len(doc_files)} files.")
        st.session_state.rag_indexed = added > 0

    rag_query = st.text_input("Ask something about your documents:")

    if rag_query:
        if not st.session_state.get("rag_indexed", False):
            st.warning("Please upload and index documents first.")
        else:
            with st.spinner("Retrieving relevant chunks and querying models..."):
                doc_context, doc_sources = query_rag(
                    rag_query, top_k=5, namespace="user_docs"
                )

                rag_prompt = f"""
You are an assistant that must answer ONLY using the DOCUMENT CONTEXT below.
If the answer is not present, say you are not sure instead of guessing.

USER QUESTION:
{rag_query}

DOCUMENT CONTEXT:
{doc_context}
"""

                task_docs = classify_task_from_question(rag_prompt)
                used_models_docs = get_task_models(task_docs, mode=mode)

                result_docs = asyncio.run(
                    query_models_with_web(
                        user_question=rag_prompt,
                        source_models=used_models_docs,
                        max_pages=1,  # web not really used here
                    )
                )

            st.markdown("### Answer from documents")
            st.markdown(result_docs["final_answer"])

            st.markdown("### Document chunks used")
            for s in doc_sources:
                st.markdown(
                    f"- **{s['filename']}** (chunk {s['chunk_index']}): {s['preview']}..."
                )
