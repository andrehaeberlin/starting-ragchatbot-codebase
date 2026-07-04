"""Tests proving/disproving whether AIGenerator correctly drives the
search_course_content tool-calling contract (backend/ai_generator.py)."""
import json

import pytest

from ai_generator import AIGenerator


@pytest.fixture
def generator(mock_openai_client):
    gen = AIGenerator(api_key="test-key", model="gpt-4o-mini")
    gen.client = mock_openai_client
    return gen


# ---------------------------------------------------------------------------
# Direct-answer path (no tool call)
# ---------------------------------------------------------------------------

def test_direct_answer_path_never_touches_tool_manager(generator, stub_tool_manager, openai_builders):
    response = openai_builders.completion("stop", openai_builders.message(content="Paris is the capital of France."))
    generator.client.chat.completions.create.return_value = response

    result = generator.generate_response(
        query="What is the capital of France?",
        tools=[{"name": "search_course_content", "description": "d", "input_schema": {}}],
        tool_manager=stub_tool_manager,
    )

    assert result == "Paris is the capital of France."
    stub_tool_manager.execute_tool.assert_not_called()
    assert generator.client.chat.completions.create.call_count == 1


def test_no_tools_passed_means_no_tools_key_in_api_params(generator, openai_builders):
    response = openai_builders.completion("stop", openai_builders.message(content="Some answer."))
    generator.client.chat.completions.create.return_value = response

    generator.generate_response(query="General question")

    call_kwargs = generator.client.chat.completions.create.call_args.kwargs
    assert "tools" not in call_kwargs
    assert "tool_choice" not in call_kwargs


def test_conversation_history_appears_in_system_message(generator, openai_builders):
    response = openai_builders.completion("stop", openai_builders.message(content="Some answer."))
    generator.client.chat.completions.create.return_value = response

    generator.generate_response(query="Follow-up question", conversation_history="User: Hi\nAssistant: Hello!")

    call_kwargs = generator.client.chat.completions.create.call_args.kwargs
    system_message = call_kwargs["messages"][0]
    assert system_message["role"] == "system"
    assert "User: Hi\nAssistant: Hello!" in system_message["content"]


# ---------------------------------------------------------------------------
# Tool-call path - happy paths
# ---------------------------------------------------------------------------

def test_tool_call_invokes_execute_tool_with_parsed_arguments(generator, stub_tool_manager, openai_builders):
    tool_call = openai_builders.tool_call("call_1", "search_course_content", json.dumps({"query": "X", "course_name": "Y"}))
    first_response = openai_builders.completion("tool_calls", openai_builders.message(content=None, tool_calls=[tool_call]))
    second_response = openai_builders.completion("stop", openai_builders.message(content="Final answer."))
    generator.client.chat.completions.create.side_effect = [first_response, second_response]
    stub_tool_manager.execute_tool.return_value = "tool result text"

    generator.generate_response(
        query="Content question",
        tools=[{"name": "search_course_content", "description": "d", "input_schema": {}}],
        tool_manager=stub_tool_manager,
    )

    stub_tool_manager.execute_tool.assert_called_once_with("search_course_content", query="X", course_name="Y")


def test_final_response_returned_after_tool_execution(generator, stub_tool_manager, openai_builders):
    tool_call = openai_builders.tool_call("call_1", "search_course_content", json.dumps({"query": "X"}))
    first_response = openai_builders.completion("tool_calls", openai_builders.message(content=None, tool_calls=[tool_call]))
    second_response = openai_builders.completion("stop", openai_builders.message(content="Final answer."))
    generator.client.chat.completions.create.side_effect = [first_response, second_response]
    stub_tool_manager.execute_tool.return_value = "tool result text"

    result = generator.generate_response(
        query="Content question",
        tools=[{"name": "search_course_content", "description": "d", "input_schema": {}}],
        tool_manager=stub_tool_manager,
    )

    assert result == "Final answer."
    assert generator.client.chat.completions.create.call_count == 2


def test_second_call_assistant_message_echoes_tool_calls_exactly(generator, stub_tool_manager, openai_builders):
    tool_call = openai_builders.tool_call("call_1", "search_course_content", json.dumps({"query": "X"}))
    first_response = openai_builders.completion("tool_calls", openai_builders.message(content=None, tool_calls=[tool_call]))
    second_response = openai_builders.completion("stop", openai_builders.message(content="Final answer."))
    generator.client.chat.completions.create.side_effect = [first_response, second_response]
    stub_tool_manager.execute_tool.return_value = "tool result text"

    generator.generate_response(
        query="Content question",
        tools=[{"name": "search_course_content", "description": "d", "input_schema": {}}],
        tool_manager=stub_tool_manager,
    )

    second_call_messages = generator.client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assistant_msg = second_call_messages[2]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["tool_calls"] == [
        {"id": "call_1", "type": "function", "function": {"name": "search_course_content", "arguments": json.dumps({"query": "X"})}}
    ]


def test_second_call_tool_result_message_follows_assistant_message(generator, stub_tool_manager, openai_builders):
    tool_call = openai_builders.tool_call("call_1", "search_course_content", json.dumps({"query": "X"}))
    first_response = openai_builders.completion("tool_calls", openai_builders.message(content=None, tool_calls=[tool_call]))
    second_response = openai_builders.completion("stop", openai_builders.message(content="Final answer."))
    generator.client.chat.completions.create.side_effect = [first_response, second_response]
    stub_tool_manager.execute_tool.return_value = "tool result text"

    generator.generate_response(
        query="Content question",
        tools=[{"name": "search_course_content", "description": "d", "input_schema": {}}],
        tool_manager=stub_tool_manager,
    )

    second_call_messages = generator.client.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_msg = second_call_messages[3]
    assert tool_msg == {"role": "tool", "tool_call_id": "call_1", "content": "tool result text"}


def test_second_call_omits_tools_and_tool_choice_keys(generator, stub_tool_manager, openai_builders):
    tool_call = openai_builders.tool_call("call_1", "search_course_content", json.dumps({"query": "X"}))
    first_response = openai_builders.completion("tool_calls", openai_builders.message(content=None, tool_calls=[tool_call]))
    second_response = openai_builders.completion("stop", openai_builders.message(content="Final answer."))
    generator.client.chat.completions.create.side_effect = [first_response, second_response]
    stub_tool_manager.execute_tool.return_value = "tool result text"

    generator.generate_response(
        query="Content question",
        tools=[{"name": "search_course_content", "description": "d", "input_schema": {}}],
        tool_manager=stub_tool_manager,
    )

    second_call_kwargs = generator.client.chat.completions.create.call_args_list[1].kwargs
    assert "tools" not in second_call_kwargs
    assert "tool_choice" not in second_call_kwargs


def test_multiple_parallel_tool_calls_all_executed_in_order(generator, stub_tool_manager, openai_builders):
    tool_call_1 = openai_builders.tool_call("call_1", "search_course_content", json.dumps({"query": "A"}))
    tool_call_2 = openai_builders.tool_call("call_2", "search_course_content", json.dumps({"query": "B"}))
    first_response = openai_builders.completion(
        "tool_calls", openai_builders.message(content=None, tool_calls=[tool_call_1, tool_call_2])
    )
    second_response = openai_builders.completion("stop", openai_builders.message(content="Final answer."))
    generator.client.chat.completions.create.side_effect = [first_response, second_response]
    stub_tool_manager.execute_tool.side_effect = ["result A", "result B"]

    generator.generate_response(
        query="Content question",
        tools=[{"name": "search_course_content", "description": "d", "input_schema": {}}],
        tool_manager=stub_tool_manager,
    )

    assert stub_tool_manager.execute_tool.call_args_list == [
        (("search_course_content",), {"query": "A"}),
        (("search_course_content",), {"query": "B"}),
    ]
    second_call_messages = generator.client.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
    assert tool_messages == [
        {"role": "tool", "tool_call_id": "call_1", "content": "result A"},
        {"role": "tool", "tool_call_id": "call_2", "content": "result B"},
    ]


def test_none_content_serializes_as_none_not_string(generator, stub_tool_manager, openai_builders):
    tool_call = openai_builders.tool_call("call_1", "search_course_content", json.dumps({"query": "X"}))
    first_response = openai_builders.completion("tool_calls", openai_builders.message(content=None, tool_calls=[tool_call]))
    second_response = openai_builders.completion("stop", openai_builders.message(content="Final answer."))
    generator.client.chat.completions.create.side_effect = [first_response, second_response]
    stub_tool_manager.execute_tool.return_value = "tool result text"

    generator.generate_response(
        query="Content question",
        tools=[{"name": "search_course_content", "description": "d", "input_schema": {}}],
        tool_manager=stub_tool_manager,
    )

    second_call_messages = generator.client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert second_call_messages[2]["content"] is None


def test_tool_calls_finish_reason_but_no_tool_manager_falls_through_to_direct_content(generator, openai_builders):
    tool_call = openai_builders.tool_call("call_1", "search_course_content", json.dumps({"query": "X"}))
    response = openai_builders.completion("tool_calls", openai_builders.message(content="fallback text", tool_calls=[tool_call]))
    generator.client.chat.completions.create.return_value = response

    result = generator.generate_response(query="Content question", tool_manager=None)

    assert result == "fallback text"
    assert generator.client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# Regression tests: a single malformed tool call must degrade gracefully to a
# final answer instead of crashing the whole request (previously these raised
# json.JSONDecodeError / TypeError uncaught, which matched "content questions
# fail" - see backend/ai_generator.py::_handle_tool_execution).
# ---------------------------------------------------------------------------

def test_malformed_tool_call_arguments_json_degrades_gracefully(generator, stub_tool_manager, openai_builders):
    tool_call = openai_builders.tool_call("call_1", "search_course_content", '{"query": "incomplete')
    first_response = openai_builders.completion("tool_calls", openai_builders.message(content=None, tool_calls=[tool_call]))
    second_response = openai_builders.completion("stop", openai_builders.message(content="I couldn't process that search."))
    generator.client.chat.completions.create.side_effect = [first_response, second_response]

    result = generator.generate_response(
        query="Content question",
        tools=[{"name": "search_course_content", "description": "d", "input_schema": {}}],
        tool_manager=stub_tool_manager,
    )

    assert result == "I couldn't process that search."
    stub_tool_manager.execute_tool.assert_not_called()
    second_call_messages = generator.client.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_msg = second_call_messages[3]
    assert tool_msg["role"] == "tool"
    assert "Tool execution error" in tool_msg["content"]


def test_tool_call_with_unexpected_argument_key_degrades_gracefully_end_to_end(
    generator, tool_manager_real, openai_builders
):
    tool_call = openai_builders.tool_call(
        "call_1", "search_course_content", json.dumps({"query": "x", "bogus_field": "y"})
    )
    first_response = openai_builders.completion("tool_calls", openai_builders.message(content=None, tool_calls=[tool_call]))
    second_response = openai_builders.completion("stop", openai_builders.message(content="I couldn't process that search."))
    generator.client.chat.completions.create.side_effect = [first_response, second_response]

    result = generator.generate_response(
        query="Content question",
        tools=[{"name": "search_course_content", "description": "d", "input_schema": {}}],
        tool_manager=tool_manager_real,
    )

    assert result == "I couldn't process that search."
    second_call_messages = generator.client.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_msg = second_call_messages[3]
    assert tool_msg["role"] == "tool"
    assert "Tool execution error" in tool_msg["content"]


def test_openai_api_error_propagates_uncaught(generator, stub_tool_manager):
    class FakeAPIError(Exception):
        pass

    generator.client.chat.completions.create.side_effect = FakeAPIError("rate limited")

    with pytest.raises(FakeAPIError):
        generator.generate_response(query="Any question", tool_manager=stub_tool_manager)
