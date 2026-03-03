"""
BaseAgent – abstract base class for all MASAI agents.

Each concrete agent:
  1. Declares its tools via `get_tool_schemas()`.
  2. Implements `handle_tool_call()` to dispatch to the right tool instance.
  3. Calls `self._run_loop()` with a prepared system prompt and messages.
"""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import Any

from llm.client import LLMClient
from memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for all MASAI domain agents."""

    agent_name: str = "BaseAgent"

    def __init__(self, llm: LLMClient, memory: MemoryManager) -> None:
        self.llm = llm
        self.memory = memory

    # ── Subclass interface ────────────────────────────────────────────────

    @abstractmethod
    def get_tool_schemas(self) -> list[dict]:
        """Return Anthropic-format tool schemas this agent exposes."""

    @abstractmethod
    def handle_tool_call(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Dispatch tool_name to the appropriate tool and return the result."""

    # ── Shared runner ──────────────────────────────────────────────────────

    def _run_loop(
        self,
        system_prompt: str,
        user_message: str,
        extra_messages: list[dict] | None = None,
    ) -> str:
        messages: list[dict] = extra_messages or []
        messages.append({"role": "user", "content": user_message})

        response, _ = self.llm.run_agent_loop(
            system=system_prompt,
            messages=messages,
            tools=self.get_tool_schemas(),
            tool_handler=self.handle_tool_call,
        )
        return response

    def __repr__(self) -> str:
        return f"<{self.agent_name}>"
