import json
from openai import OpenAI
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with OpenAI's Chat Completions API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

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

    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

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
            {"role": "user", "content": query}
        ]

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": messages
        }

        # Add tools if available
        if tools:
            api_params["tools"] = self._to_openai_tools(tools)
            api_params["tool_choice"] = "auto"

        # Get response from OpenAI
        response = self.client.chat.completions.create(**api_params)
        message = response.choices[0].message

        # Handle tool execution if needed
        if response.choices[0].finish_reason == "tool_calls" and tool_manager:
            return self._handle_tool_execution(message, messages, tool_manager)

        # Return direct response
        return message.content

    @staticmethod
    def _to_openai_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Anthropic-style tool definitions to OpenAI's function-calling format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            }
            for tool in tools
        ]

    def _handle_tool_execution(self, message, messages: List[Dict[str, Any]], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            message: The assistant message containing tool_calls
            messages: Conversation messages so far (system + user)
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = messages.copy()

        # Add AI's tool call message
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in message.tool_calls
            ]
        })

        # Execute all tool calls and add results as individual tool messages
        for tool_call in message.tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            tool_result = tool_manager.execute_tool(tool_call.function.name, **arguments)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages
        }

        # Get final response
        final_response = self.client.chat.completions.create(**final_params)
        return final_response.choices[0].message.content
