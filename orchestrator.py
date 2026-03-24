# orchestrator.py

import asyncio
from typing import Dict, List, Tuple

import httpx
from ddgs import DDGS          # CHANGED: use ddgs package
import ollama

from models_config import get_default_sources, get_aggregator_model


# ---------- Web search & fetch ----------

async def web_search(query: str, num_results: int = 5) -> List[Dict]:
    """
    Use DuckDuckGo to get web search results.
    """
    results: List[Dict] = []
    # DDGS is synchronous; wrap as simple blocking call.
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=num_results):
            # r keys: "title", "href", "body" etc.
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
            )
    return results


async def fetch_url(url: str, timeout: int = 15) -> str:
    """
    Fetch raw HTML/text from a URL using httpx.
    """
    if not url:
        return ""
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url)
            text = resp.text
            # naive truncation to keep prompt size manageable
            return text[:20000]
    except Exception as e:
        return f"[ERROR fetching {url}: {e}]"


async def build_web_context(query: str, max_pages: int = 3) -> Tuple[str, List[Dict]]:
    """
    Search the web and build a text block with top pages' content.
    """
    search_results = await web_search(query)
    top = search_results[:max_pages]

    contents = await asyncio.gather(*(fetch_url(r["url"]) for r in top))

    blocks: List[str] = []
    for r, text in zip(top, contents):
        blocks.append(
            f"TITLE: {r['title']}\n"
            f"URL: {r['url']}\n"
            f"SNIPPET: {r['snippet']}\n"
            f"CONTENT:\n{text}\n"
        )

    context = "\n\n--- WEB PAGE ---\n\n".join(blocks)
    return context, top


# ---------- Local LLM calls via Ollama ----------

async def call_ollama_model(model: str, prompt: str) -> str:
    """
    Call a local LLM via Ollama in a thread so Streamlit doesn't block.
    """
    loop = asyncio.get_event_loop()

    def _run():
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful, precise AI assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp["message"]["content"]

    return await loop.run_in_executor(None, _run)


# ---------- Multi-model + aggregation ----------

async def query_models_with_web(
    user_question: str,
    source_models: List[str] | None = None,
    max_pages: int = 3,
) -> Dict:
    """
    Orchestrate web search + multi-model queries + aggregation.
    Returns final answer + per-model answers + search results.
    """
    if source_models is None:
        source_models = get_default_sources()

    # 1. Build web context from live search
    web_context, search_results = await build_web_context(user_question, max_pages=max_pages)

    base_prompt = f"""
You are an AI assistant answering a user's question using ONLY the information
from the WEB CONTEXT below. If something is not clearly supported by the context,
say that you are not sure instead of hallucinating.

USER QUESTION:
{user_question}

WEB CONTEXT (search results and page extracts):
{web_context}
"""

    # 2. Fan-out to multiple local models in parallel
    tasks = {m: call_ollama_model(m, base_prompt) for m in source_models}
    per_model: Dict[str, str] = {}

    for name, coro in tasks.items():
        try:
            per_model[name] = await asyncio.wait_for(coro, timeout=120)
        except Exception as e:
            per_model[name] = f"[ERROR from {name}: {e}]"

    # 3. Aggregate all model answers with a strong reasoning model
    answers_block = "\n\n".join(
        f"### MODEL: {name}\n{text}\n" for name, text in per_model.items()
    )

    aggregator_model = get_aggregator_model()

    aggregator_prompt = f"""
You are a strong reasoning model.

Your job:
- Read the user question.
- Read the answers from several local models.
- Produce ONE final answer that is accurate, concise, and well structured.
- Use bullet points when helpful.
- Prefer statements that are clearly supported by the web context.
- If models disagree or the web data conflicts, explain the uncertainty briefly.
- When possible, reference the source URLs explicitly that appear in the context.

USER QUESTION:
{user_question}

ANSWERS FROM MODELS:
{answers_block}
"""

    final_answer = await call_ollama_model(aggregator_model, aggregator_prompt)

    return {
        "final_answer": final_answer,
        "per_model": per_model,
        "search_results": search_results,
    }
