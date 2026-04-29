"""
app/agent/tools/base_tool.py

Base class cho mọi tool trong hệ thống.
OCP: thêm tool mới → kế thừa AgentTool, đăng ký vào ToolRegistry.
ISP: mỗi tool chỉ biết interface nó cần, không biết các tool khác.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from app.agent.state.agent_state import AgentState, ToolCall
from app.core.interfaces.vector_port import VectorPort
from app.shared.logging.logger import get_logger

log = get_logger(__name__)


@dataclass
class ToolResult:
    success: bool
    data: Any
    summary: str       # 1-2 câu mô tả kết quả — để update state
    error: str = ""


class AgentTool(ABC):
    """Base cho mọi tool. SRP: mỗi tool chỉ làm 1 việc."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Mô tả để LLM có thể chọn tool (nếu dùng LLM routing)."""
        ...

    @abstractmethod
    async def run(self, state: AgentState) -> ToolResult:
        ...

    async def execute(self, state: AgentState) -> tuple[ToolResult, ToolCall]:
        """Wrapper thêm timing và audit logging."""
        start = time.monotonic()
        try:
            result = await self.run(state)
            duration = int((time.monotonic() - start) * 1000)
            call = ToolCall(
                tool_name=self.name,
                input_summary=self._summarize_input(state),
                output_summary=result.summary,
                duration_ms=duration,
                success=result.success,
            )
            log.info(
                "tool_executed",
                tool=self.name,
                duration_ms=duration,
                success=result.success,
            )
            return result, call
        except Exception as e:
            duration = int((time.monotonic() - start) * 1000)
            log.error("tool_error", tool=self.name, error=str(e))
            result = ToolResult(success=False, data=None, summary="", error=str(e))
            call = ToolCall(
                tool_name=self.name,
                input_summary=self._summarize_input(state),
                output_summary=f"ERROR: {e}",
                duration_ms=duration,
                success=False,
            )
            return result, call

    def _summarize_input(self, state: AgentState) -> str:
        return f"query={state['raw_query'][:80]!r}"


class ToolRegistry:
    """
    Registry pattern — không dùng if-else chuỗi.
    OCP: thêm tool = register(), không sửa code khác.
    """

    def __init__(self, vdb: Optional[VectorPort] = None) -> None:
        self._tools: dict[str, AgentTool] = {}
        self._vdb = vdb

    def register(self, tool: AgentTool) -> None:
        self._tools[tool.name] = tool
        log.info("tool_registered", name=tool.name)

    def get_vdb(self) -> VectorPort:
        if not self._vdb:
            raise ValueError("Vector DB not initialized in ToolRegistry")
        return self._vdb


    def get(self, name: str) -> AgentTool | None:
        return self._tools.get(name)

    def all(self) -> list[AgentTool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())