Multi‑Modal AI Archestrator 

A research‑assistant‑style AI that answers user queries using **live web search** and **local document RAG** (PDF/TXT/MD), with a Perplexity‑inspired UI. It supports **text, audio, images, video, and files** as inputs, routes tasks automatically (general/code/math), and queries **multiple local LLMs** (Llama, Mistral, Qwen, DeepSeek) via Ollama for a single summarized answer.

## Key Features

- Web search tab with live DuckDuckGo + page fetching and citations.  
- Documents (RAG) tab for indexing and querying your own PDFs/TXTs locally.  
- Multi‑model orchestration with task‑based model selection and answer aggregation.  
- Perplexity‑style chat UI with sources box, answer‑style modes, and follow‑up suggestions.  
- Fully local and open‑source: no cloud API keys, everything runs on your machine.

## Tech Stack

- **Frontend / UI**: Streamlit  
- **LLMs**: Ollama (local models via `ollama` Python client)  
- **Search & RAG**: `ddgs` (DuckDuckGo) + `httpx` + `PyPDF2` + `sentence‑transformers` + `chromadb`  
- **Utils**: `asyncio`, `httpx`, `Pillow`

## How to Run

1. Install Ollama and pull models:
   ```bash
   ollama pull llama3.2:latest
   ollama pull mistral:latest
   ollama pull qwen2.5:7b
   ollama pull deepseek-r1:7b
Start Ollama:

bash
ollama serve
Install Python dependencies:

bash
cd multi_llm_websearch_app
pip install -r requirements.txt
Run the app:

bash
python -m streamlit run app.py
Open http://localhost:8501 in your browser.

Files
app.py – main UI with two tabs: Web + Multi‑LLM and My Documents (RAG).

orchestrator.py – multi‑model + web search orchestration.

models_config.py – model definitions, task routing, and aggregator selection.

multimodal_utils.py – handling audio, images, video, and file inputs.

rag_store.py – PDF/TXT indexing and RAG querying with Chroma and embeddings