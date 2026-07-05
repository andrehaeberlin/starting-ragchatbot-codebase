import json
from typing import Any

from openai import OpenAI


class AIGenerator:
    """Handles interactions with OpenAI's Chat Completions API for generating responses"""

    # Maximum number of sequential tool-calling rounds per user query
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- You may search up to **two times** in a row when answering a single question, but treat that as the exception, not the default
- Only use a second search when the first search's results reveal something that must itself be searched for to answer the question — e.g. finding a lesson's topic/title first, then searching for that topic elsewhere; or when the question has multiple distinct parts spanning different courses/lessons that one search cannot cover
- Do **not** search twice out of habit, to "double-check" a result, or when the first search already answers the question — stop and answer as soon as you have enough information
- Synthesize search results into accurate, fact-based responses
- If a search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(
        self, api_key: str, model: str, max_tool_rounds: int = MAX_TOOL_ROUNDS
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tool_rounds = max_tool_rounds

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: str | None = None,
        tools: list | None = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response, allowing up to `max_tool_rounds` sequential rounds
        of tool calling. Each round is a separate API request so the model can
        reason about the previous round's tool results before deciding whether
        to search again.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use (Anthropic-style tool definitions)
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        round_num = 0
        while True:
            round_num += 1

            # Snapshot messages per call - `messages` keeps growing across
            # rounds, and recording a live reference (rather than a copy)
            # would make earlier calls appear to retroactively include later
            # rounds' messages to anything inspecting call history.
            api_params = {**self.base_params, "messages": list(messages)}
            if tools and round_num <= self.max_tool_rounds:
                api_params["tools"] = self._to_openai_tools(tools)
                api_params["tool_choice"] = "auto"
                api_params["parallel_tool_calls"] = False

            response = self.client.chat.completions.create(**api_params)
            message = response.choices[0].message
            has_tool_calls = response.choices[0].finish_reason == "tool_calls" and bool(
                message.tool_calls
            )

            # Termination (b): no tool calls requested - this is the final answer
            if not has_tool_calls or not tool_manager:
                return message.content

            messages.append(self._build_assistant_tool_call_message(message))

            # Execute all tool calls and add results as individual tool messages.
            # Malformed arguments or unexpected tool signatures degrade gracefully
            # and do not stop the round loop; only genuinely unexpected/unhandled
            # exceptions from the tool layer count as a hard failure.
            hard_failure = False
            for tool_call in message.tool_calls:
                tool_result, failed = self._execute_tool_call(tool_call, tool_manager)
                hard_failure = hard_failure or failed

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )

            # Termination (a) or (c): rounds exhausted or a tool call hard-failed -
            # fall through to a final, tools-omitted call to synthesize an answer
            if hard_failure or round_num >= self.max_tool_rounds:
                break

        final_params = {**self.base_params, "messages": list(messages)}
        final_response = self.client.chat.completions.create(**final_params)
        return final_response.choices[0].message.content

    @staticmethod
    def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Anthropic-style tool definitions to OpenAI's function-calling format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ]

    @staticmethod
    def _build_assistant_tool_call_message(message) -> dict[str, Any]:
        """Build the assistant message that echoes back the model's tool_calls,
        required before any role:"tool" result messages can be sent."""
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ],
        }

    @staticmethod
    def _execute_tool_call(tool_call, tool_manager) -> tuple[str, bool]:
        """
        Execute a single tool call, returning (result_content, hard_failure).

        Malformed arguments (bad JSON) or a tool signature mismatch (unexpected
        argument keys) are degraded to an error message the model can react to
        without counting as a "tool call fails" termination. Any other
        unexpected exception from the tool layer is a hard failure.
        """
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            return f"Tool execution error: {e}", False

        try:
            return (
                tool_manager.execute_tool(tool_call.function.name, **arguments),
                False,
            )
        except TypeError as e:
            return f"Tool execution error: {e}", False
        except Exception as e:
            return f"Tool execution error: {e}", True
