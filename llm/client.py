"""
LLMClient – thin wrapper around the Anthropic Messages API.

Supports:
  - Single-turn completion
  - Agentic tool-use loops (runs until no more tool calls)
"""
from __future__ import annotations
import json
import logging
from typing import Any, Callable

import anthropic

import config

logger = logging.getLogger(__name__)

ToolHandler = Callable[[str, dict], dict]


class LLMClient:
    def __init__(
        self,
        model: str = config.LLM_MODEL,
        max_tokens: int = config.LLM_MAX_TOKENS,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def complete(self, system: str, messages: list[dict]) -> str:
        """Single-turn completion (no tool use)."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )
        return response.content[0].text

    def run_agent_loop(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        tool_handler: ToolHandler,
        max_iterations: int = 10,
    ) -> tuple[str, list[dict]]:
        """
        Run an agentic tool-use loop.

        Args:
            system:         System prompt for the agent.
            messages:       Conversation history (will be mutated in place).
            tools:          Anthropic-format tool schemas.
            tool_handler:   Callable(tool_name, tool_input) → result_dict.
            max_iterations: Safety limit on the number of LLM calls.

        Returns:
            (final_text_response, updated_messages)
        """
        for iteration in range(max_iterations):
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=messages,
                tools=tools,
            )
            logger.debug("Iteration %d – stop_reason: %s", iteration, response.stop_reason)

            # Append assistant turn
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # Extract final text
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text, messages
                return "", messages

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        logger.info("Calling tool: %s", block.name)
                        try:
                            result = tool_handler(block.name, block.input)
                        except Exception as exc:
                            result = {"error": str(exc)}
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, default=str),
                        })
                messages.append({"role": "user", "content": tool_results})
            else:
                # Unexpected stop reason
                break

        logger.warning("Agent loop exceeded max_iterations=%d", max_iterations)
        return "", messages
