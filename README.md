 How to run
Install and start Ollama (follow their docs, then):​

bash
# examples – you can change models if you want
ollama pull llama3.2:latest
ollama pull mistral:latest
ollama pull qwen2.5:7b-instruct
ollama pull deepseek-r1:7b


ollama pull qwen2.5-coder:7b
ollama pull qwen2.5:7b-math

ollama serve
Install Python deps:

bash
cd multi_llm_websearch_app
pip install -r requirements.txt
Run the app:

bash

python -m streamlit run app.py

Open the URL Streamlit prints (usually http://localhost:8501), ask a question, and the app will:

search the internet,

fetch top pages,

send the context to multiple local models,

show a single summarized answer.

If you want to extend this (logging to DB, rate limiting, or connecting your Flutter app instead of Streamlit), that can be added on top of this structure without changing the core orchestration.

