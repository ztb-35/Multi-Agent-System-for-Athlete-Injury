"""
Base class for all MASAI tools.

Each concrete tool class must:
  1. Define `name`, `description`, and `input_schema`
  2. Implement `run(input: dict) -> dict`

The `to_anthropic_schema()` method converts the tool to the format expected
by the Anthropic Messages API (tool_use).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    name: str
    description: str
    input_schema: dict  # JSON Schema

    @abstractmethod
    def run(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool and return a structured result."""

    def to_anthropic_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
