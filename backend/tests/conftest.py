"""Shared fixtures for the backend test suite.

backend/*.py modules use bare imports (e.g. `from vector_store import VectorStore`)
rather than package-relative ones, so `backend/` itself must be on sys.path.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from models import Course, CourseChunk, Lesson
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults, VectorStore

# ---------------------------------------------------------------------------
# Sample course data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_courses():
    return [
        Course(
            title="Intro to Testing",
            course_link="https://example.com/intro-to-testing",
            instructor="Ada Lovelace",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="Why Test?",
                    lesson_link="https://example.com/intro-to-testing/lesson-1",
                ),
                # Deliberately missing lesson_link to exercise the None-link path.
                Lesson(lesson_number=2, title="Writing Assertions", lesson_link=None),
            ],
        ),
        Course(
            title="Advanced Fixtures",
            course_link="https://example.com/advanced-fixtures",
            instructor="Grace Hopper",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="Fixture Scopes",
                    lesson_link="https://example.com/advanced-fixtures/lesson-1",
                ),
            ],
        ),
    ]


@pytest.fixture
def sample_chunks():
    return [
        CourseChunk(
            content="Lesson 1 content: Testing verifies behavior instead of hoping it works.",
            course_title="Intro to Testing",
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="Assertions compare actual output to expected output.",
            course_title="Intro to Testing",
            lesson_number=2,
            chunk_index=1,
        ),
        CourseChunk(
            content="Lesson 1 content: Fixtures can be scoped to function, class, module, or session.",
            course_title="Advanced Fixtures",
            lesson_number=1,
            chunk_index=0,
        ),
    ]


# ---------------------------------------------------------------------------
# Fake (pure unit) vector store
# ---------------------------------------------------------------------------


class FakeVectorStore:
    """Hand-rolled stand-in for VectorStore exposing only what CourseSearchTool touches."""

    def __init__(self):
        self.search_return: SearchResults | None = None
        self.lesson_links: dict = {}
        self.course_links: dict = {}
        self.last_search_kwargs = None

    def search(
        self,
        query: str,
        course_name: str | None = None,
        lesson_number: int | None = None,
    ) -> SearchResults:
        self.last_search_kwargs = {
            "query": query,
            "course_name": course_name,
            "lesson_number": lesson_number,
        }
        return self.search_return

    def get_lesson_link(self, course_title: str, lesson_number: int) -> str | None:
        return self.lesson_links.get((course_title, lesson_number))

    def get_course_link(self, course_title: str) -> str | None:
        return self.course_links.get(course_title)


@pytest.fixture
def fake_vector_store():
    return FakeVectorStore()


@pytest.fixture
def course_search_tool_fake(fake_vector_store):
    return CourseSearchTool(fake_vector_store)


@pytest.fixture
def tool_manager_fake(course_search_tool_fake):
    manager = ToolManager()
    manager.register_tool(course_search_tool_fake)
    return manager


# ---------------------------------------------------------------------------
# Real (integration) vector store backed by a temp ChromaDB instance
# ---------------------------------------------------------------------------


@pytest.fixture
def real_vector_store(tmp_path, sample_courses, sample_chunks):
    store = VectorStore(str(tmp_path / "chroma_db"), "all-MiniLM-L6-v2", max_results=5)
    for course in sample_courses:
        store.add_course_metadata(course)
    store.add_course_content(sample_chunks)
    return store


@pytest.fixture
def course_search_tool_real(real_vector_store):
    return CourseSearchTool(real_vector_store)


@pytest.fixture
def tool_manager_real(course_search_tool_real):
    manager = ToolManager()
    manager.register_tool(course_search_tool_real)
    return manager


# ---------------------------------------------------------------------------
# OpenAI response builders (SimpleNamespace mirrors real SDK attribute access
# and raises AttributeError on typos, unlike MagicMock's auto-vivification).
# ---------------------------------------------------------------------------


def make_tool_call(id: str, name: str, arguments_json_str: str):
    return SimpleNamespace(
        id=id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments_json_str),
    )


def make_message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls, role="assistant")


def make_completion(finish_reason: str, message):
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish_reason, message=message)]
    )


@pytest.fixture
def openai_builders():
    return SimpleNamespace(
        tool_call=make_tool_call,
        message=make_message,
        completion=make_completion,
    )


@pytest.fixture
def mock_openai_client():
    """A fake OpenAI client exposing only client.chat.completions.create(...)."""
    create = Mock()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return client


@pytest.fixture
def stub_tool_manager():
    return Mock(spec=ToolManager)


# ---------------------------------------------------------------------------
# FastAPI test app (backend/app.py can't be imported directly in tests: it
# instantiates a real RAGSystem at import time, registers a startup event
# that reads ../docs, and mounts StaticFiles(directory="../frontend") - a
# path that only resolves when cwd is backend/. This mirrors its route
# bodies/models against an injected mock RAGSystem and a fixture-controlled
# static directory instead.)
# ---------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class SourceItem(BaseModel):
    text: str
    link: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: list[str]


def create_test_app(rag_system, static_dir) -> FastAPI:
    """Builds a FastAPI app mirroring backend/app.py's API routes against the
    given rag_system, with static files served from static_dir instead of
    the real ../frontend."""
    app = FastAPI(title="Course Materials RAG System - Test")

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def clear_session(session_id: str):
        try:
            rag_system.session_manager.clear_session(session_id)
            return {"success": True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


@pytest.fixture
def mock_rag_system():
    rag_system = Mock()
    rag_system.session_manager = Mock()
    rag_system.session_manager.create_session.return_value = "test-session-id"
    rag_system.query.return_value = (
        "Test answer",
        [{"text": "Intro to Testing - Lesson 1", "link": "https://example.com/intro-to-testing/lesson-1"}],
    )
    rag_system.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Intro to Testing", "Advanced Fixtures"],
    }
    return rag_system


@pytest.fixture
def test_app(mock_rag_system, tmp_path):
    static_dir = tmp_path / "frontend"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html><body>Test Frontend</body></html>")
    return create_test_app(mock_rag_system, static_dir)


@pytest.fixture
def client(test_app):
    return TestClient(test_app)
