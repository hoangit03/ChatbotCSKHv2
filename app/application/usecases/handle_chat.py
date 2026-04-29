"""
app/application/usecases/handle_chat.py

Use Case: CB-02 — Xử lý hội thoại qua Agent LangGraph.

Nhận ChatRequest → khởi tạo AgentState → invoke graph → trả ChatResponse.
Class này KHÔNG biết gì về LangGraph internals — chỉ gọi graph.invoke().

SRP: mapping request/response + invoke graph, không logic business.
DIP: nhận compiled graph qua constructor.
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Optional

from app.agent.state.agent_state import AgentState, SourceRef, ToolCall, make_initial_state
from app.shared.logging.logger import get_logger

log = get_logger(__name__)


# ── Request / Response DTOs ───────────────────────────────────────

@dataclass
class ChatRequest:
    message: str
    session_id: Optional[str] = None      # None → tạo mới
    project_name: Optional[str] = None    # filter theo dự án cụ thể
    # Thông tin khách hàng (dùng cho booking intent)
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None


@dataclass
class SourceRefDTO:
    document_code: str
    document_name: str
    doc_group: str
    excerpt: str
    page: Optional[int] = None


@dataclass
class ToolCallDTO:
    tool_name: str
    input_summary: str
    output_summary: str
    duration_ms: int
    success: bool


@dataclass
class ChatResponse:
    session_id: str
    answer: str
    intent: str
    sources: list[SourceRefDTO] = field(default_factory=list)
    tool_calls: list[ToolCallDTO] = field(default_factory=list)
    fallback: bool = False
    fallback_reason: str = ""
    was_injected: bool = False
    project_name: Optional[str] = None    # Dự án được detect thực tế
    response_time_ms: int = 0


# ── Use Case ──────────────────────────────────────────────────────

class HandleChatUseCase:
    """
    Orchestrates CB-02: tạo state → invoke LangGraph → map response.
    Graph được inject qua __init__ (DIP).
    """

    def __init__(self, agent_graph, history_store=None) -> None:
        # agent_graph là compiled LangGraph (kiểu CompiledStateGraph)
        # Không type-hint cụ thể để tránh circular import
        self._graph = agent_graph
        self._history = history_store

    async def execute(self, req: ChatRequest) -> ChatResponse:
        session_id = req.session_id or _new_session_id()
        t0 = time.monotonic()

        log.info(
            "chat_start",
            session_id=session_id,
            project=req.project_name,
            msg_len=len(req.message),
        )

        # ── Khởi tạo state ────────────────────────────────────────
        state = make_initial_state(
            session_id=session_id,
            raw_query=req.message,
            project_name=req.project_name,
        )

        # Load history và context nếu có
        if self._history:
            # 1. Load chat messages
            history = await self._history.get_history(session_id)
            state["messages"] = history
            
            # 2. Load persistent context (project_name)
            ctx = await self._history.get_context(session_id)
            cached_project = ctx.get("project_name")
            
            # Ưu tiên project truyền từ request (nếu có và hợp lệ)
            # Nếu request rỗng/default -> dùng project từ cache
            if not req.project_name or req.project_name.lower() in ["", "string", "none"]:
                state["project_name"] = cached_project
            else:
                state["project_name"] = req.project_name
                
            log.debug("chat_session_loaded", session=session_id, history=len(history), project=state["project_name"])

        # Gắn thêm thông tin khách hàng nếu có (dùng cho booking)
        if req.customer_name or req.customer_phone:
            state["sales_data"] = {
                "customer_name":  req.customer_name or "",
                "customer_phone": req.customer_phone or "",
            }

        # ── Invoke LangGraph ──────────────────────────────────────
        try:
            final_state: AgentState = await self._graph.ainvoke(state)
        except Exception as e:
            log.error("chat_graph_error", session_id=session_id, error=str(e))
            return ChatResponse(
                session_id=session_id,
                answer=(
                    "Xin lỗi, hệ thống đang gặp sự cố. "
                    "Vui lòng thử lại hoặc liên hệ Sales để được hỗ trợ."
                ),
                intent="unknown",
                fallback=True,
                fallback_reason=str(e),
                response_time_ms=_ms(t0),
            )

        # ── Map state → response ──────────────────────────────────
        response = ChatResponse(
            session_id=session_id,
            answer=final_state.get("final_answer", ""),
            intent=final_state.get("intent", "unknown"),
            sources=_map_sources(final_state.get("sources", [])),
            tool_calls=_map_tool_calls(final_state.get("tool_calls", [])),
            fallback=final_state.get("fallback", False),
            fallback_reason=final_state.get("fallback_reason", ""),
            was_injected=final_state.get("was_injected", False),
            project_name=final_state.get("project_name"),
            response_time_ms=_ms(t0),
        )

        # Lưu history (user msg & assistant answer)
        if self._history:
            await self._history.append(session_id, "user", req.message)
            await self._history.append(session_id, "assistant", response.answer)
            # Lưu lại project_name thực tế sau khi agent xử lý (có thể agent đã detect được project mới)
            if response.project_name:
                await self._history.set_context(session_id, {"project_name": response.project_name})

        log.info(
            "chat_done",
            session_id=session_id,
            intent=response.intent,
            fallback=response.fallback,
            sources=len(response.sources),
            tool_calls=len(response.tool_calls),
            ms=response.response_time_ms,
        )

        return response


# ── Helpers ───────────────────────────────────────────────────────

def _new_session_id() -> str:
    return f"sess_{secrets.token_urlsafe(10)}"


def _ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


def _map_sources(sources: list) -> list[SourceRefDTO]:
    result = []
    for s in sources:
        if isinstance(s, SourceRef):
            result.append(SourceRefDTO(
                document_code=s.document_code,
                document_name=s.document_name,
                doc_group=s.doc_group,
                excerpt=s.excerpt,
                page=s.page,
            ))
        elif isinstance(s, dict):
            result.append(SourceRefDTO(**s))
    return result


def _map_tool_calls(calls: list) -> list[ToolCallDTO]:
    result = []
    for c in calls:
        if isinstance(c, ToolCall):
            result.append(ToolCallDTO(
                tool_name=c.tool_name,
                input_summary=c.input_summary,
                output_summary=c.output_summary,
                duration_ms=c.duration_ms,
                success=c.success,
            ))
        elif isinstance(c, dict):
            result.append(ToolCallDTO(**c))
    return result