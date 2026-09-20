# YouTube Video Q&A (RAG)

Ask questions about any YouTube video by chatting with its transcript. Fetches the video's captions, embeds them into a local vector store, and answers questions using retrieval-augmented generation (RAG) — FastAPI backend + Streamlit chat frontend.

## Why this project
I wanted to build a RAG pipeline end-to-end on a real, messy data source (auto-generated captions, inconsistent URL formats, no fixed document structure) rather than clean PDFs — and wire it up as an actual API + chat UI instead of just a notebook demo.

## How it works

1. **Ingestion** — given a YouTube link, the video ID is parsed from the URL (handles `/watch`, `youtu.be`, `/shorts/`, `/embed/`, `/live/` links) and the transcript is fetched via `youtube-transcript-api`.
2. **Chunking** — the transcript is joined into one string and split into ~600-character chunks (50-character overlap) with `RecursiveCharacterTextSplitter`.
3. **Embedding + storage** — each chunk is embedded locally with `all-MiniLM-L6-v2` (sentence-transformers) and stored in a **Chroma** collection scoped to that specific video ID — so switching videos doesn't mix context between them.
4. **Retrieval + generation** — on each question, the top-2 most similar chunks are retrieved and passed, along with the question, into a prompt sent to **Gemini 2.5 Flash Lite** via a LangChain LCEL chain. The prompt instructs the model to answer only from the retrieved context, say "I don't know" otherwise, and keep answers to three sentences.

## Architecture
```
User → Streamlit chat UI
     → POST /video_link_from → FastAPI backend
         → extract_video_id() → fetch transcript → chunk → embed → Chroma (per-video collection)
     → POST /predict (question) → FastAPI backend
         → retriever (top-2 chunks) → prompt → Gemini 2.5 Flash Lite → answer
     ← rendered in chat UI
```

## Project structure
```
Youtube_Rag/
├── requirements.txt
├── Youtube_Q_A_bot.py           # standalone CLI version
├── backend/
│   ├── fastapi_main.py          # FastAPI app — /video_link_from and /predict endpoints
│   └── Youtube_Q_A_bot.py       # RAG pipeline as reusable functions (ingestion → chunking → embedding → retrieval → generation)
└── frontend/
    └── streamlit_app.py         # Streamlit chat UI
```

## Running it

**1. Set up your API key** — create a `.env` file in `backend/`:
```
GOOGLE_API_KEY=your-gemini-api-key
```

**2. Backend:**
```bash
cd Youtube_Rag/backend
pip install -r ../requirements.txt
uvicorn fastapi_main:app --reload
```

**3. Frontend** (in a separate terminal):
```bash
cd Youtube_Rag/frontend
streamlit run streamlit_app.py
```
Note: the frontend currently points at `http://backend:8000`, which assumes a Docker Compose setup. For local (non-Docker) use, change this to `http://127.0.0.1:8000` in `streamlit_app.py`, or better, read it from an environment variable (see "What I'd improve").

## Tech Stack
Python · LangChain · Google Gemini (`gemini-2.5-flash-lite`) · ChromaDB · sentence-transformers (`all-MiniLM-L6-v2`) · FastAPI · Streamlit · `youtube-transcript-api`

## Bugs found and fixed
While building this I found and fixed two real issues, not just style nits:
- **Cross-video contamination:** the vector store originally wrote every video into the same default Chroma collection, so questions about one video could retrieve chunks from a previously loaded video. Fixed by scoping each video into its own collection (`video_{video_id}`).
- **Brittle video ID parsing:** the original code extracted the video ID with string-splitting, which broke on `/shorts/`, `/embed/`, `/live/` links and on `?v=ID&t=30s` URLs. Replaced with proper URL parsing (`urllib.parse`).
- Also removed a hardcoded API key from source (now read from `.env`) and replaced a `sys.exit()` inside the FastAPI handler (which would crash the worker process) with a proper `HTTPException`.

## Current limitations
- **English only** — both the transcript fetch and the embedding model are English-focused; videos with only non-English captions aren't handled (no Whisper fallback for missing captions).
- **Small context window (k=2)** — works well for specific factual questions, but broad questions like "summarize this video" will do poorly, especially on longer videos, since the model only sees ~1,200 characters of retrieved context.
- **No timestamp linking** — timestamps are discarded during chunking, so answers can't be linked back to a specific point in the video (would require keeping each snippet's `start` time as chunk metadata).
- **Single video/user at a time** — the backend holds `vector_db` and `rag_chain` as module-level globals, so all users share one active video and conversation state.
- **No conversation memory** — each question is answered independently; the frontend only stores history for display, not for context.
- **Docker-dependent setup** — the frontend's backend URL and Chroma's persist path assume a container environment; no `Dockerfile`/`docker-compose.yml` is included yet.

## What I'd improve
- Add a `Dockerfile` + `docker-compose.yml` so the Docker-shaped assumptions in the code (backend hostname, persist path) actually work out of the box
- Read the backend URL from an environment variable instead of hardcoding it in the frontend
- Add a Whisper-based fallback for videos with no captions available
- Store timestamps as chunk metadata to support "jump to this point in the video" answers
- Move from module-level global state to per-session state for multi-user support
- Swap the deprecated `langchain_community` Chroma/embeddings wrappers for `langchain-chroma` / `langchain-huggingface`
- Add basic tests for `extract_video_id()` across all URL formats
