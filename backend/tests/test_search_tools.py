"""Tests for CourseSearchTool.execute() and ToolManager (backend/search_tools.py)."""
import pytest

from vector_store import SearchResults


# ---------------------------------------------------------------------------
# CourseSearchTool.execute() - success formatting (real store)
# ---------------------------------------------------------------------------

def test_execute_returns_formatted_results_no_filters(course_search_tool_real):
    result = course_search_tool_real.execute(query="testing behavior")
    assert "[Intro to Testing" in result or "[Advanced Fixtures" in result


def test_execute_with_course_name_filter_real_store(course_search_tool_real):
    result = course_search_tool_real.execute(query="fixtures", course_name="Advanced Fixtures")
    assert "Advanced Fixtures" in result
    assert "Intro to Testing" not in result


def test_execute_with_lesson_number_filter_real_store(course_search_tool_real):
    result = course_search_tool_real.execute(query="assertions", course_name="Intro to Testing", lesson_number=2)
    assert "Lesson 2" in result


def test_execute_with_combined_course_and_lesson_filter_real_store(course_search_tool_real):
    result = course_search_tool_real.execute(
        query="fixture scopes", course_name="Advanced Fixtures", lesson_number=1
    )
    assert "Advanced Fixtures" in result
    assert "Lesson 1" in result


# ---------------------------------------------------------------------------
# CourseSearchTool.execute() - empty results (fake store, exact message text)
# ---------------------------------------------------------------------------

def test_execute_empty_results_no_filters_message(course_search_tool_fake, fake_vector_store):
    fake_vector_store.search_return = SearchResults(documents=[], metadata=[], distances=[])
    result = course_search_tool_fake.execute(query="anything")
    assert result == "No relevant content found."


def test_execute_empty_results_with_course_name_message_text(course_search_tool_fake, fake_vector_store):
    fake_vector_store.search_return = SearchResults(documents=[], metadata=[], distances=[])
    result = course_search_tool_fake.execute(query="anything", course_name="Intro to Testing")
    assert result == "No relevant content found in course 'Intro to Testing'."


def test_execute_empty_results_with_lesson_number_message_text(course_search_tool_fake, fake_vector_store):
    fake_vector_store.search_return = SearchResults(documents=[], metadata=[], distances=[])
    result = course_search_tool_fake.execute(query="anything", lesson_number=3)
    assert result == "No relevant content found in lesson 3."


def test_execute_empty_results_with_both_filters_message_text(course_search_tool_fake, fake_vector_store):
    fake_vector_store.search_return = SearchResults(documents=[], metadata=[], distances=[])
    result = course_search_tool_fake.execute(query="anything", course_name="Intro to Testing", lesson_number=3)
    assert result == "No relevant content found in course 'Intro to Testing' in lesson 3."


# ---------------------------------------------------------------------------
# CourseSearchTool.execute() - error propagation
# ---------------------------------------------------------------------------

def test_execute_propagates_vector_store_error_string_verbatim(course_search_tool_fake, fake_vector_store):
    fake_vector_store.search_return = SearchResults.empty("Search error: boom")
    result = course_search_tool_fake.execute(query="anything")
    assert result == "Search error: boom"


def test_execute_course_name_not_found_returns_no_match_error(course_search_tool_fake, fake_vector_store):
    fake_vector_store.search_return = SearchResults.empty("No course found matching 'Nonexistent Course'")
    result = course_search_tool_fake.execute(query="anything", course_name="Nonexistent Course")
    assert result == "No course found matching 'Nonexistent Course'"


# ---------------------------------------------------------------------------
# last_sources shape and lifecycle
# ---------------------------------------------------------------------------

def test_last_sources_lesson_case_shape(course_search_tool_fake, fake_vector_store):
    fake_vector_store.search_return = SearchResults(
        documents=["some lesson text"],
        metadata=[{"course_title": "Intro to Testing", "lesson_number": 1}],
        distances=[0.1],
    )
    fake_vector_store.lesson_links[("Intro to Testing", 1)] = "https://example.com/lesson-1"

    course_search_tool_fake.execute(query="anything")

    assert course_search_tool_fake.last_sources == [
        {"text": "Intro to Testing - Lesson 1", "link": "https://example.com/lesson-1"}
    ]


def test_last_sources_course_link_fallback_no_lesson_number(course_search_tool_fake, fake_vector_store):
    fake_vector_store.search_return = SearchResults(
        documents=["some course-level text"],
        metadata=[{"course_title": "Intro to Testing"}],
        distances=[0.1],
    )
    fake_vector_store.course_links["Intro to Testing"] = "https://example.com/course"

    course_search_tool_fake.execute(query="anything")

    assert course_search_tool_fake.last_sources == [
        {"text": "Intro to Testing", "link": "https://example.com/course"}
    ]


def test_last_sources_overwritten_not_accumulated_across_calls(course_search_tool_fake, fake_vector_store):
    fake_vector_store.search_return = SearchResults(
        documents=["doc one"],
        metadata=[{"course_title": "Intro to Testing", "lesson_number": 1}],
        distances=[0.1],
    )
    course_search_tool_fake.execute(query="first")
    first_sources = course_search_tool_fake.last_sources
    assert len(first_sources) == 1

    fake_vector_store.search_return = SearchResults(
        documents=["doc two"],
        metadata=[{"course_title": "Advanced Fixtures", "lesson_number": 1}],
        distances=[0.1],
    )
    course_search_tool_fake.execute(query="second")

    assert len(course_search_tool_fake.last_sources) == 1
    assert course_search_tool_fake.last_sources[0]["text"] == "Advanced Fixtures - Lesson 1"


def test_last_sources_cleared_on_subsequent_error_or_empty_result(course_search_tool_fake, fake_vector_store):
    """Regression test: a subsequent error/empty search must clear stale sources
    from a prior successful call rather than leaving them exposed via
    get_last_sources()."""
    fake_vector_store.search_return = SearchResults(
        documents=["doc one"],
        metadata=[{"course_title": "Intro to Testing", "lesson_number": 1}],
        distances=[0.1],
    )
    course_search_tool_fake.execute(query="first")
    assert len(course_search_tool_fake.last_sources) == 1

    fake_vector_store.search_return = SearchResults.empty("Search error: boom")
    course_search_tool_fake.execute(query="second")

    assert course_search_tool_fake.last_sources == []


# ---------------------------------------------------------------------------
# Tool definition shape
# ---------------------------------------------------------------------------

def test_get_tool_definition_shape(course_search_tool_fake):
    definition = course_search_tool_fake.get_tool_definition()
    assert definition["name"] == "search_course_content"
    assert definition["input_schema"]["type"] == "object"
    assert definition["input_schema"]["required"] == ["query"]
    assert set(definition["input_schema"]["properties"].keys()) == {"query", "course_name", "lesson_number"}


# ---------------------------------------------------------------------------
# ToolManager
# ---------------------------------------------------------------------------

def test_tool_manager_register_tool_keyed_by_name(tool_manager_fake):
    assert "search_course_content" in tool_manager_fake.tools


def test_tool_manager_register_tool_missing_name_raises_value_error(tool_manager_fake):
    class BadTool:
        def get_tool_definition(self):
            return {"description": "no name here"}

        def execute(self, **kwargs):
            return ""

    with pytest.raises(ValueError):
        tool_manager_fake.register_tool(BadTool())


def test_tool_manager_get_tool_definitions_aggregates_all_registered(tool_manager_fake):
    definitions = tool_manager_fake.get_tool_definitions()
    assert len(definitions) == 1
    assert definitions[0]["name"] == "search_course_content"


def test_tool_manager_execute_tool_dispatches_to_correct_tool(tool_manager_fake, fake_vector_store):
    fake_vector_store.search_return = SearchResults(documents=[], metadata=[], distances=[])
    result = tool_manager_fake.execute_tool("search_course_content", query="anything")
    assert result == "No relevant content found."


def test_tool_manager_execute_tool_unknown_name_returns_error_string_not_raise(tool_manager_fake):
    result = tool_manager_fake.execute_tool("nonexistent_tool", query="anything")
    assert result == "Tool 'nonexistent_tool' not found"


def test_tool_manager_get_last_sources_scans_all_registered_tools(tool_manager_fake, fake_vector_store):
    fake_vector_store.search_return = SearchResults(
        documents=["doc"],
        metadata=[{"course_title": "Intro to Testing", "lesson_number": 1}],
        distances=[0.1],
    )
    tool_manager_fake.execute_tool("search_course_content", query="anything")
    assert tool_manager_fake.get_last_sources() == [
        {"text": "Intro to Testing - Lesson 1", "link": None}
    ]


def test_tool_manager_reset_sources_clears_all_tools(tool_manager_fake, fake_vector_store):
    fake_vector_store.search_return = SearchResults(
        documents=["doc"],
        metadata=[{"course_title": "Intro to Testing", "lesson_number": 1}],
        distances=[0.1],
    )
    tool_manager_fake.execute_tool("search_course_content", query="anything")
    assert tool_manager_fake.get_last_sources() != []

    tool_manager_fake.reset_sources()
    assert tool_manager_fake.get_last_sources() == []
