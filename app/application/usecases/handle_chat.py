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
import asyncio

from app.agent.state.agent_state import AgentState, SourceRef, ToolCall, make_initial_state
from app.shared.logging.logger import get_logger
from app.infrastructure.cache.pg_history import save_chat_message_async

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
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    role_level: Optional[str] = None


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
    suggested_questions: list[str] = field(default_factory=list)  # 3 câu hỏi gợi ý tiếp theo
    sales_data: dict = field(default_factory=dict)  # Raw sales data cho UI


# ── Use Case ──────────────────────────────────────────────────────

class HandleChatUseCase:
    """
    Orchestrates CB-02: tạo state → invoke LangGraph → map response.
    Graph được inject qua __init__ (DIP).
    """

    def __init__(self, agent_graph, history_store=None, activity_logger=None) -> None:
        self._graph = agent_graph
        self._history = history_store
        self._activity_log = activity_logger  # UserActivityLogger (optional)

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
        # Xác định user_type từ role_level header
        # role_level "1" = khách hàng, > "1" = nội bộ sale
        _role = req.role_level or "1"
        user_type = "sale" if _role.isdigit() and int(_role) > 1 else "customer"

        state = make_initial_state(
            session_id=session_id,
            raw_query=req.message,
            project_name=req.project_name,
            user_type=user_type,
        )

        # Load history và context nếu có
        if self._history:
            # 1. Load chat messages
            history = await self._history.get_history(session_id, limit=20)
            state["messages"] = history
            
            # 2. Load persistent context (project_name + customer journey)
            ctx = await self._history.get_context(session_id)
            cached_project = ctx.get("project_name")
            
            # Ưu tiên project truyền từ request (nếu có và hợp lệ)
            if not req.project_name or req.project_name.lower() in ["", "string", "none"]:
                state["project_name"] = cached_project
            else:
                state["project_name"] = req.project_name

            # 3. Restore customer journey state (fix stage/USP reset bug)
            if ctx.get("customer_stage"):
                state["customer_stage"] = ctx["customer_stage"]
            if ctx.get("usps_used"):
                state["usps_used"] = ctx["usps_used"]
            if ctx.get("appointment_booked"):
                state["appointment_booked"] = ctx["appointment_booked"]

            log.debug(
                "chat_session_loaded",
                session=session_id,
                history=len(history),
                project=state["project_name"],
                stage=state.get("customer_stage"),
            )

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
                    "Dạ, hiện tại hệ thống đang xử lý quá nhiều yêu cầu nên phản hồi chậm. "
                    "Anh/chị vui lòng để lại số điện thoại để chuyên viên tư vấn gọi lại hỗ trợ mình ngay nhé."
                ),
                intent="unknown",
                fallback=True,
                fallback_reason=str(e),
                response_time_ms=_ms(t0),
            )

        # ── Map state → response ──────────────────────────────────
        raw_intent = final_state.get("intent", "unknown")
        intent_str = raw_intent.value if hasattr(raw_intent, "value") else str(raw_intent)
        response = ChatResponse(
            session_id=session_id,
            answer=final_state.get("final_answer", ""),
            intent=intent_str,
            sources=_map_sources(final_state.get("sources", [])),
            tool_calls=_map_tool_calls(final_state.get("tool_calls", [])),
            fallback=final_state.get("fallback", False),
            fallback_reason=final_state.get("fallback_reason", ""),
            was_injected=final_state.get("was_injected", False),
            project_name=final_state.get("project_name"),
            response_time_ms=_ms(t0),
            suggested_questions=final_state.get("suggested_questions", []),
            sales_data=final_state.get("sales_data", {}),
        )

        # Lưu history (user msg & assistant answer)
        if self._history:
            await self._history.append(session_id, "user", req.message)
            await self._history.append(session_id, "assistant", response.answer)
            asyncio.create_task(save_chat_message_async(session_id, "user", req.message, req.user_id, req.tenant_id))
            asyncio.create_task(save_chat_message_async(session_id, "assistant", response.answer, req.user_id, req.tenant_id))
            
            # Persist toàn bộ context sau mỗi request (fix stage/USP reset)
            ctx_to_save: dict = {}
            if response.project_name:
                ctx_to_save["project_name"] = response.project_name
            if final_state.get("customer_stage"):
                ctx_to_save["customer_stage"] = final_state["customer_stage"]
            if final_state.get("usps_used"):
                ctx_to_save["usps_used"] = final_state["usps_used"]
            if final_state.get("appointment_booked"):
                ctx_to_save["appointment_booked"] = final_state["appointment_booked"]
            if ctx_to_save:
                await self._history.set_context(session_id, ctx_to_save)

        # Ghi user activity audit log
        if self._activity_log:
            try:
                await self._activity_log.log_chat_event(req, response)
            except Exception as log_err:
                log.warning("activity_log_failed", error=str(log_err))

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

    async def execute_stream(self, req: ChatRequest):
        """
        [NEW] Stream mode for CB-02.
        Yields SSE chunks: `data: {"text": "..."}`
        """
        import json
        session_id = req.session_id or _new_session_id()
        t0 = time.monotonic()

        log.info(
            "chat_stream_start",
            session_id=session_id,
            project=req.project_name,
            msg_len=len(req.message),
        )

        queue = asyncio.Queue()
        _role = req.role_level or "1"
        user_type = "sale" if _role.isdigit() and int(_role) > 1 else "customer"

        state = make_initial_state(
            session_id=session_id,
            raw_query=req.message,
            project_name=req.project_name,
            user_type=user_type,
        )
        state["stream_queue"] = queue

        # Load history and context
        if self._history:
            history = await self._history.get_history(session_id, limit=20)
            state["messages"] = history
            ctx = await self._history.get_context(session_id)
            cached_project = ctx.get("project_name")
            if not req.project_name or req.project_name.lower() in ["", "string", "none"]:
                state["project_name"] = cached_project
            else:
                state["project_name"] = req.project_name
            # Restore customer journey state (stream mode)
            if ctx.get("customer_stage"):
                state["customer_stage"] = ctx["customer_stage"]
            if ctx.get("usps_used"):
                state["usps_used"] = ctx["usps_used"]
            if ctx.get("appointment_booked"):
                state["appointment_booked"] = ctx["appointment_booked"]

        if req.customer_name or req.customer_phone:
            state["sales_data"] = {
                "customer_name":  req.customer_name or "",
                "customer_phone": req.customer_phone or "",
            }

        # Khởi tạo task báo cáo "Đang kiểm tra" nếu LLM quá chậm (> 3s)
        state["real_token_emitted"] = False
        
        async def filler_task(q: asyncio.Queue, s: dict):
            await asyncio.sleep(3.0)
            if not s.get("real_token_emitted"):
                filler_text = "Dạ em đang kiểm tra thông tin, anh/chị đợi một chút nhé...\n\n"
                words = filler_text.split(" ")
                for i, word in enumerate(words):
                    if s.get("real_token_emitted"): 
                        break
                    # Stream từng từ một cách tự nhiên
                    chunk_text = word + " " if i < len(words) - 1 else word
                    await q.put({"type": "filler_token", "content": chunk_text})
                    await asyncio.sleep(0.15)

        filler_bg = asyncio.create_task(filler_task(queue, state))

        # Run graph in background task
        graph_task = asyncio.create_task(self._graph.ainvoke(state))

        # Stream from queue
        while True:
            chunk = await queue.get()
            if chunk["type"] == "done":
                break
            elif chunk["type"] == "filler_token":
                yield f"data: {json.dumps({'text': chunk['content'], 'session_id': session_id})}\n\n"
            elif chunk["type"] == "token":
                state["real_token_emitted"] = True
                yield f"data: {json.dumps({'text': chunk['content'], 'session_id': session_id})}\n\n"
            elif chunk["type"] == "suggestions":
                yield f"data: {json.dumps({'suggested_questions': chunk['content'], 'session_id': session_id})}\n\n"
        
        # Hủy task mồi nếu nó vẫn đang chạy
        if not filler_bg.done():
            filler_bg.cancel()

        # Wait for graph to finish completely
        final_state = await graph_task
        
        # Map response
        raw_intent = final_state.get("intent", "unknown")
        intent_str = raw_intent.value if hasattr(raw_intent, "value") else str(raw_intent)
        response = ChatResponse(
            session_id=session_id,
            answer=final_state.get("final_answer", ""),
            intent=intent_str,
            sources=_map_sources(final_state.get("sources", [])),
            tool_calls=_map_tool_calls(final_state.get("tool_calls", [])),
            fallback=final_state.get("fallback", False),
            fallback_reason=final_state.get("fallback_reason", ""),
            was_injected=final_state.get("was_injected", False),
            project_name=final_state.get("project_name"),
            response_time_ms=_ms(t0),
            suggested_questions=final_state.get("suggested_questions", []),
            sales_data=final_state.get("sales_data", {}),
        )

        # Save history (stream mode)
        if self._history:
            await self._history.append(session_id, "user", req.message)
            await self._history.append(session_id, "assistant", response.answer)
            asyncio.create_task(save_chat_message_async(session_id, "user", req.message, req.user_id, req.tenant_id))
            asyncio.create_task(save_chat_message_async(session_id, "assistant", response.answer, req.user_id, req.tenant_id))

            ctx_to_save: dict = {}
            if response.project_name:
                ctx_to_save["project_name"] = response.project_name
            if final_state.get("customer_stage"):
                ctx_to_save["customer_stage"] = final_state["customer_stage"]
            if final_state.get("usps_used"):
                ctx_to_save["usps_used"] = final_state["usps_used"]
            if final_state.get("appointment_booked"):
                ctx_to_save["appointment_booked"] = final_state["appointment_booked"]
            if ctx_to_save:
                await self._history.set_context(session_id, ctx_to_save)

        # Send raw sources/tool calls at the end
        metadata_chunk = {
            "sources": [{"doc": s.document_name, "excerpt": s.excerpt} for s in response.sources],
            "intent": response.intent,
            "session_id": session_id
        }
        yield f"data: {json.dumps(metadata_chunk)}\n\n"
        yield "data: [DONE]\n\n"


# ── Helpers ───────────────────────────────────────────────────────

def _new_session_id() -> str:
    import uuid
    return str(uuid.uuid4())


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