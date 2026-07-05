"""API endpoint tests for the FastAPI layer (backend/app.py's routes),
exercised against the test app built in conftest.py (see create_test_app)
with a mocked RAGSystem - no real ChromaDB, embedding model, or OpenAI calls."""


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

def test_query_with_session_id_returns_answer_and_sources(client, mock_rag_system):
    response = client.post("/api/query", json={"query": "Why do we test?", "session_id": "existing-session"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Test answer"
    assert body["session_id"] == "existing-session"
    assert body["sources"] == [
        {"text": "Intro to Testing - Lesson 1", "link": "https://example.com/intro-to-testing/lesson-1"}
    ]
    mock_rag_system.query.assert_called_once_with("Why do we test?", "existing-session")


def test_query_without_session_id_creates_one(client, mock_rag_system):
    response = client.post("/api/query", json={"query": "Why do we test?"})

    assert response.status_code == 200
    assert response.json()["session_id"] == "test-session-id"
    mock_rag_system.session_manager.create_session.assert_called_once()
    mock_rag_system.query.assert_called_once_with("Why do we test?", "test-session-id")


def test_query_missing_query_field_returns_422(client, mock_rag_system):
    response = client.post("/api/query", json={"session_id": "existing-session"})

    assert response.status_code == 422
    mock_rag_system.query.assert_not_called()


def test_query_raising_exception_returns_500(client, mock_rag_system):
    mock_rag_system.query.side_effect = RuntimeError("vector store unavailable")

    response = client.post("/api/query", json={"query": "Why do we test?"})

    assert response.status_code == 500
    assert response.json()["detail"] == "vector store unavailable"


def test_query_source_with_none_link_round_trips(client, mock_rag_system):
    mock_rag_system.query.return_value = ("Assertions compare values.", [{"text": "Intro to Testing - Lesson 2", "link": None}])

    response = client.post("/api/query", json={"query": "What are assertions?"})

    assert response.status_code == 200
    assert response.json()["sources"] == [{"text": "Intro to Testing - Lesson 2", "link": None}]


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------

def test_get_course_stats_success(client):
    response = client.get("/api/courses")

    assert response.status_code == 200
    assert response.json() == {
        "total_courses": 2,
        "course_titles": ["Intro to Testing", "Advanced Fixtures"],
    }


def test_get_course_stats_raising_exception_returns_500(client, mock_rag_system):
    mock_rag_system.get_course_analytics.side_effect = RuntimeError("chroma down")

    response = client.get("/api/courses")

    assert response.status_code == 500
    assert response.json()["detail"] == "chroma down"


# ---------------------------------------------------------------------------
# DELETE /api/session/{session_id}
# ---------------------------------------------------------------------------

def test_clear_session_success(client, mock_rag_system):
    response = client.delete("/api/session/some-session-id")

    assert response.status_code == 200
    assert response.json() == {"success": True}
    mock_rag_system.session_manager.clear_session.assert_called_once_with("some-session-id")


def test_clear_session_raising_exception_returns_500(client, mock_rag_system):
    mock_rag_system.session_manager.clear_session.side_effect = RuntimeError("session store unavailable")

    response = client.delete("/api/session/some-session-id")

    assert response.status_code == 500
    assert response.json()["detail"] == "session store unavailable"


# ---------------------------------------------------------------------------
# GET / (static mount)
# ---------------------------------------------------------------------------

def test_root_serves_index_html(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "Test Frontend" in response.text


def test_unknown_static_path_returns_404(client):
    response = client.get("/does-not-exist.js")

    assert response.status_code == 404
