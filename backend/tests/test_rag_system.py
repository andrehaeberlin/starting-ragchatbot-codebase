"""End-to-end tests for RAGSystem.query()'s handling of content-related
questions (backend/rag_system.py), using a real seeded VectorStore with only
the OpenAI network boundary mocked out."""

import json

import pytest

from config import Config
from rag_system import RAGSystem


@pytest.fixture
def rag_system_real(tmp_path, sample_courses, sample_chunks):
    config = Config(CHROMA_PATH=str(tmp_path / "chroma_db"), OPENAI_API_KEY="test-key")
    system = RAGSystem(config)
    for course in sample_courses:
        system.vector_store.add_course_metadata(course)
    system.vector_store.add_course_content(sample_chunks)
    return system


@pytest.fixture
def rag_system_empty(tmp_path):
    config = Config(
        CHROMA_PATH=str(tmp_path / "chroma_db_empty"), OPENAI_API_KEY="test-key"
    )
    return RAGSystem(config)


def _mock_client(rag_system, mock_openai_client):
    rag_system.ai_generator.client = mock_openai_client
    return mock_openai_client


def test_content_question_returns_answer_and_sources_tuple_shape(
    rag_system_real, mock_openai_client, openai_builders
):
    _mock_client(rag_system_real, mock_openai_client)
    tool_call = openai_builders.tool_call(
        "call_1",
        "search_course_content",
        json.dumps(
            {"query": "why test", "course_name": "Intro to Testing", "lesson_number": 1}
        ),
    )
    first_response = openai_builders.completion(
        "tool_calls", openai_builders.message(content=None, tool_calls=[tool_call])
    )
    second_response = openai_builders.completion(
        "stop", openai_builders.message(content="Testing verifies behavior.")
    )
    mock_openai_client.chat.completions.create.side_effect = [
        first_response,
        second_response,
    ]

    answer, sources = rag_system_real.query("Why do we test?")

    assert answer == "Testing verifies behavior."
    assert sources == [
        {
            "text": "Intro to Testing - Lesson 1",
            "link": "https://example.com/intro-to-testing/lesson-1",
        }
    ]


def test_general_knowledge_question_no_tool_call_returns_empty_sources(
    rag_system_real, mock_openai_client, openai_builders
):
    _mock_client(rag_system_real, mock_openai_client)
    response = openai_builders.completion(
        "stop", openai_builders.message(content="2 + 2 = 4.")
    )
    mock_openai_client.chat.completions.create.return_value = response

    answer, sources = rag_system_real.query("What is 2 + 2?")

    assert answer == "2 + 2 = 4."
    assert sources == []


def test_sources_reset_between_successive_queries(
    rag_system_real, mock_openai_client, openai_builders
):
    _mock_client(rag_system_real, mock_openai_client)
    tool_call = openai_builders.tool_call(
        "call_1",
        "search_course_content",
        json.dumps(
            {"query": "why test", "course_name": "Intro to Testing", "lesson_number": 1}
        ),
    )
    first_response = openai_builders.completion(
        "tool_calls", openai_builders.message(content=None, tool_calls=[tool_call])
    )
    second_response = openai_builders.completion(
        "stop", openai_builders.message(content="Answer one.")
    )
    mock_openai_client.chat.completions.create.side_effect = [
        first_response,
        second_response,
    ]
    _, sources_one = rag_system_real.query("Why do we test?")
    assert len(sources_one) == 1

    third_response = openai_builders.completion(
        "stop", openai_builders.message(content="Answer two.")
    )
    mock_openai_client.chat.completions.create.side_effect = [third_response]
    _, sources_two = rag_system_real.query("What is 2 + 2?")

    assert sources_two == []


def test_session_history_threads_across_turns(
    rag_system_real, mock_openai_client, openai_builders
):
    _mock_client(rag_system_real, mock_openai_client)
    session_id = rag_system_real.session_manager.create_session()

    response_one = openai_builders.completion(
        "stop", openai_builders.message(content="First answer.")
    )
    mock_openai_client.chat.completions.create.side_effect = [response_one]
    rag_system_real.query("First question?", session_id=session_id)

    response_two = openai_builders.completion(
        "stop", openai_builders.message(content="Second answer.")
    )
    mock_openai_client.chat.completions.create.side_effect = [response_two]
    rag_system_real.query("Second question?", session_id=session_id)

    second_call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    system_message = second_call_kwargs["messages"][0]["content"]
    assert "First question?" in system_message
    assert "First answer." in system_message


def test_no_session_id_does_not_persist_history(
    rag_system_real, mock_openai_client, openai_builders
):
    _mock_client(rag_system_real, mock_openai_client)
    response = openai_builders.completion(
        "stop", openai_builders.message(content="An answer.")
    )
    mock_openai_client.chat.completions.create.return_value = response

    rag_system_real.query("A question with no session?", session_id=None)

    assert rag_system_real.session_manager.sessions == {}


def test_malformed_tool_arguments_degrades_gracefully_full_stack(
    rag_system_real, mock_openai_client, openai_builders
):
    """Full-stack regression test: malformed tool-call arguments must not crash
    RAGSystem.query() - previously this raised uncaught all the way up to what
    app.py flattens into the 500 the frontend shows as 'Query failed'."""
    _mock_client(rag_system_real, mock_openai_client)
    tool_call = openai_builders.tool_call(
        "call_1", "search_course_content", '{"query": "incomplete'
    )
    first_response = openai_builders.completion(
        "tool_calls", openai_builders.message(content=None, tool_calls=[tool_call])
    )
    second_response = openai_builders.completion(
        "stop", openai_builders.message(content="I couldn't process that search.")
    )
    mock_openai_client.chat.completions.create.side_effect = [
        first_response,
        second_response,
    ]

    answer, sources = rag_system_real.query("Why do we test?")

    assert answer == "I couldn't process that search."
    assert sources == []


def test_content_question_no_matching_course_in_store(
    rag_system_empty, mock_openai_client, openai_builders
):
    _mock_client(rag_system_empty, mock_openai_client)
    tool_call = openai_builders.tool_call(
        "call_1",
        "search_course_content",
        json.dumps({"query": "anything", "course_name": "Nonexistent Course"}),
    )
    first_response = openai_builders.completion(
        "tool_calls", openai_builders.message(content=None, tool_calls=[tool_call])
    )
    second_response = openai_builders.completion(
        "stop", openai_builders.message(content="I couldn't find that course.")
    )
    mock_openai_client.chat.completions.create.side_effect = [
        first_response,
        second_response,
    ]

    answer, sources = rag_system_empty.query("Tell me about the Nonexistent Course")

    assert answer == "I couldn't find that course."
    assert sources == []


def test_link_resolution_when_lesson_link_is_none(
    rag_system_real, mock_openai_client, openai_builders
):
    tool_call = openai_builders.tool_call(
        "call_1",
        "search_course_content",
        json.dumps(
            {
                "query": "assertions",
                "course_name": "Intro to Testing",
                "lesson_number": 2,
            }
        ),
    )
    _mock_client(rag_system_real, mock_openai_client)
    first_response = openai_builders.completion(
        "tool_calls", openai_builders.message(content=None, tool_calls=[tool_call])
    )
    second_response = openai_builders.completion(
        "stop", openai_builders.message(content="Assertions compare values.")
    )
    mock_openai_client.chat.completions.create.side_effect = [
        first_response,
        second_response,
    ]

    answer, sources = rag_system_real.query("What are assertions?")

    assert answer == "Assertions compare values."
    assert sources == [{"text": "Intro to Testing - Lesson 2", "link": None}]
