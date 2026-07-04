# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for Python dependency management (Python >=3.13). There is no lint config, no test suite, and no build step in this repo.

```bash
# Install dependencies
uv sync

# Run the app (from repo root)
./run.sh
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

Requires a `.env` file in the repo root with `OPENAI_API_KEY=...` (see `.env.example`).

App runs at `http://localhost:8000` (frontend) and `http://localhost:8000/docs` (Swagger/OpenAPI).

On every startup, `backend/app.py` loads all documents from `../docs` into ChromaDB (`./backend/chroma_db`), skipping courses whose title already exists in the store — so re-running the server doesn't re-embed existing content. To force a full rebuild, delete the `chroma_db` directory.

## Architecture

Full-stack RAG chatbot: vanilla JS/HTML frontend + FastAPI backend, ChromaDB for vector storage, OpenAI (Chat Completions, default model `gpt-4o-mini`) for generation. Course content lives in `docs/*.txt`.

### Request flow

`frontend/script.js` → `POST /api/query` (`backend/app.py`) → `RAGSystem.query()` (`backend/rag_system.py`) → `AIGenerator.generate_response()` (`backend/ai_generator.py`).

The AI generator calls OpenAI **with a `search_course_content` tool attached** (converted from the internal Anthropic-style tool definition into OpenAI's `{"type": "function", "function": {...}}` shape in `AIGenerator._to_openai_tools`), letting the model itself decide whether to search:
- General knowledge questions → model answers directly, no tool call.
- Course-specific questions → model returns `finish_reason == "tool_calls"`, `ToolManager` dispatches to `CourseSearchTool.execute()` (`backend/search_tools.py`), which queries `VectorStore.search()`, and results are sent back as `role: "tool"` messages in a **second, tool-free** API call to synthesize the final answer. The system prompt enforces **one search per query maximum**.

Sources for the last search are tracked on the tool instance (`CourseSearchTool.last_sources`) and pulled/reset by `RAGSystem.query()` after each turn — this is stateful and single-threaded; a new search overwrites the previous one.

Conversation history is in-memory only (`SessionManager`, `backend/session_manager.py`), keyed by a `session_id` the frontend holds in a JS variable (not persisted across page reloads), capped at `MAX_HISTORY` exchanges (`config.py`).

### Vector store layout (`backend/vector_store.py`)

Two ChromaDB collections, both using the `all-MiniLM-L6-v2` sentence-transformer embedding function:
- `course_catalog` — one doc per course (title as ID), used only to fuzzy-resolve a user-supplied `course_name` to an exact title via semantic search (`_resolve_course_name`). Lesson metadata is stored as a JSON string (`lessons_json`) since Chroma metadata values must be scalars.
- `course_content` — the actual chunked lesson text, filterable by `course_title` and/or `lesson_number` via a Chroma `where` clause.

`get_course_link()` / `get_lesson_link()` exist on `VectorStore` but are currently unused by the search tool — sources returned to the UI are plain strings like `"Course Title - Lesson 2"` with no clickable link.

### Document ingestion (`backend/document_processor.py`)

Expects course `.txt` files in a fixed format:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <lesson title>
Lesson Link: <url>
<lesson content...>

Lesson 1: <lesson title>
...
```
Despite `RAGSystem.add_course_folder` filtering for `.pdf`/`.docx`/`.txt`, only plain-text parsing is implemented — PDF/DOCX files will not be extracted correctly.

Chunking is sentence-aware (regex-based sentence splitting, careful of abbreviations), packing sentences up to `CHUNK_SIZE` chars with a sliding-window overlap of `CHUNK_OVERLAP` chars (`config.py`, defaults 800/100). The first chunk of each lesson is prefixed with `"Lesson {N} content: "` for embedding context; note the *last* lesson in a file is handled by separate trailing-code that prefixes **every** chunk with `"Course {title} Lesson {N} content: "` instead — an inconsistency between the two code paths, not an intentional distinction.

### Config

All tunables (`OPENAI_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MAX_RESULTS`, `MAX_HISTORY`, `CHROMA_PATH`, `EMBEDDING_MODEL`) live in one dataclass: `backend/config.py`.
