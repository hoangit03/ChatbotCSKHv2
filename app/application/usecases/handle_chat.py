"""
app/application/usecases/handle_chat.py

Use Cases:
  - HandleCustomerChatUseCase  → dùng CustomerGraph (Luồng A)
  - HandleSaleChatUseCase      → dùng SaleGraph     (Luồng B)

DTOs dùng chung: ChatRequest, ChatResponse.

v2 — Tách 2 UseCase riêng biệt:
  - Customer: persist project_name vào session
  - Sale: persist project_name vào session (không có customer_stage / usps_used)
  - Xóa bỏ role_level / user_type mapping (mỗi UseCase biết rõ loại của mình)
"""
from __future__ import annotations

import asyncio
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from app.agent.state.agent_state import AgentState, SourceRef, ToolCall, make_initial_state
from app.agent.state.sale_state import SaleAgentState, make_sale_initial_state
from app.infrastructure.cache.pg_history import save_chat_message_async
from app.shared.logging.logger import get_logger

log = get_logger(__name__)


# ── Shared DTOs ───────────────────────────────────────────────────

@dataclass
class ChatRequest:
    message: str
    session_id: Optional[str]     = None
    project_name: Optional[str]   = None
    customer_name: Optional[str]  = None
    customer_phone: Optional[str] = None
    user_id: Optional[str]        = None
    tenant_id: Optional[str]      = None
    # Deprecated — giữ lại để không break caller cũ
    role_level: Optional[str]     = None


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
    sources: list[SourceRefDTO]          = field(default_factory=list)
    tool_calls: list[ToolCallDTO]        = field(default_factory=list)
    fallback: bool                       = False
    fallback_reason: str                 = ""
    was_injected: bool                   = False
    project_name: Optional[str]         = None
    response_time_ms: int               = 0
    suggested_questions: list[str]       = field(default_factory=list)
    sales_data: dict                     = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────

def _new_session_id() -> str:
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


def _is_invalid_project(name: Optional[str]) -> bool:
    return not name or name.lower() in ("", "string", "none", "unknown")


def _build_customer_response(session_id: str, final_state: AgentState, t0: float) -> ChatResponse:
    raw_intent = final_state.get("intent", "unknown")
    intent_str = raw_intent.value if hasattr(raw_intent, "value") else str(raw_intent)
    return ChatResponse(
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


def _build_sale_response(session_id: str, final_state: SaleAgentState, t0: float) -> ChatResponse:
    return ChatResponse(
        session_id=session_id,
        answer=final_state.get("final_answer", ""),
        intent="sale_inquiry",  # Sale không có intent classifier
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


# ── HandleCustomerChatUseCase ─────────────────────────────────────

class HandleCustomerChatUseCase:
    """
    Use Case cho Luồng A (Khách hàng).
    Dùng CustomerGraph. Persist project_name trong session.
    """

    def __init__(self, agent_graph, history_store=None, activity_logger=None):
        self._graph         = agent_graph
        self._history       = history_store
        self._activity_log  = activity_logger

    async def execute(self, req: ChatRequest) -> ChatResponse:
        session_id = req.session_id or _new_session_id()
        t0 = time.monotonic()
        log.info("customer_chat_start", session_id=session_id, project=req.project_name)

        state = make_initial_state(
            session_id=session_id,
            raw_query=req.message,
            project_name=req.project_name if not _is_invalid_project(req.project_name) else None,
            customer_name=req.customer_name,
            customer_phone=req.customer_phone,
        )

        if self._history:
            history = await self._history.get_history(session_id, limit=20)
            state["messages"] = history
            ctx = await self._history.get_context(session_id)
            if _is_invalid_project(req.project_name):
                state["project_name"] = ctx.get("project_name")
            else:
                state["project_name"] = req.project_name

        if req.customer_name or req.customer_phone:
            state["sales_data"] = {
                "customer_name":  req.customer_name or "",
                "customer_phone": req.customer_phone or "",
            }

        try:
            final_state: AgentState = await self._graph.ainvoke(state)
        except Exception as e:
            log.error("customer_chat_graph_error", session_id=session_id, error=str(e))
            return ChatResponse(
                session_id=session_id,
                answer="Dạ, hệ thống đang quá tải. Anh/chị vui lòng để lại SĐT để chuyên viên gọi lại nhé.",
                intent="unknown",
                fallback=True,
                fallback_reason=str(e),
                response_time_ms=_ms(t0),
            )

        response = _build_customer_response(session_id, final_state, t0)

        if self._history:
            await self._history.append(session_id, "user", req.message)
            await self._history.append(session_id, "assistant", response.answer)
            asyncio.create_task(save_chat_message_async(session_id, "user",      req.message,    req.user_id, req.tenant_id))
            asyncio.create_task(save_chat_message_async(session_id, "assistant", response.answer, req.user_id, req.tenant_id))

            ctx_to_save: dict = {}
            if response.project_name:
                ctx_to_save["project_name"] = response.project_name
            if ctx_to_save:
                await self._history.set_context(session_id, ctx_to_save)

        if self._activity_log:
            try:
                await self._activity_log.log_chat_event(req, response)
            except Exception as log_err:
                log.warning("activity_log_failed", error=str(log_err))

        log.info("customer_chat_done", session_id=session_id,
                 intent=response.intent, fallback=response.fallback, ms=response.response_time_ms)
        return response

    async def execute_stream(self, req: ChatRequest):
        """Stream mode — yields SSE chunks."""
        import json
        session_id = req.session_id or _new_session_id()
        t0 = time.monotonic()
        log.info("customer_chat_stream_start", session_id=session_id)

        queue = asyncio.Queue()
        state = make_initial_state(
            session_id=session_id,
            raw_query=req.message,
            project_name=req.project_name if not _is_invalid_project(req.project_name) else None,
            customer_name=req.customer_name,
            customer_phone=req.customer_phone,
        )
        state["stream_queue"] = queue

        if self._history:
            history = await self._history.get_history(session_id, limit=20)
            state["messages"] = history
            ctx = await self._history.get_context(session_id)
            if _is_invalid_project(req.project_name):
                state["project_name"] = ctx.get("project_name")
            else:
                state["project_name"] = req.project_name

        if req.customer_name or req.customer_phone:
            state["sales_data"] = {
                "customer_name":  req.customer_name or "",
                "customer_phone": req.customer_phone or "",
            }

        async for chunk in _stream_graph(self._graph, state, queue, session_id, req, self._history, t0):
            yield chunk


# ── HandleSaleChatUseCase ─────────────────────────────────────────

class HandleSaleChatUseCase:
    """
    Use Case cho Luồng B (Sale nội bộ).
    Dùng SaleGraph. Không persist customer_stage / usps_used.
    """

    def __init__(self, agent_graph, history_store=None, activity_logger=None):
        self._graph        = agent_graph
        self._history      = history_store
        self._activity_log = activity_logger

    async def execute(self, req: ChatRequest) -> ChatResponse:
        session_id = req.session_id or _new_session_id()
        t0 = time.monotonic()
        log.info("sale_chat_start", session_id=session_id, project=req.project_name)

        state = make_sale_initial_state(
            session_id=session_id,
            raw_query=req.message,
            project_name=req.project_name if not _is_invalid_project(req.project_name) else None,
            customer_name=req.customer_name,
            customer_phone=req.customer_phone,
        )

        if self._history:
            history = await self._history.get_history(session_id, limit=20)
            state["messages"] = history
            ctx = await self._history.get_context(session_id)
            if _is_invalid_project(req.project_name):
                state["project_name"] = ctx.get("project_name")
            else:
                state["project_name"] = req.project_name

        if req.customer_name or req.customer_phone:
            state["sales_data"] = {
                "customer_name":  req.customer_name or "",
                "customer_phone": req.customer_phone or "",
            }

        try:
            final_state: SaleAgentState = await self._graph.ainvoke(state)
        except Exception as e:
            log.error("sale_chat_graph_error", session_id=session_id, error=str(e))
            return ChatResponse(
                session_id=session_id,
                answer="Hệ thống đang xử lý, vui lòng thử lại.",
                intent="unknown",
                fallback=True,
                fallback_reason=str(e),
                response_time_ms=_ms(t0),
            )

        response = _build_sale_response(session_id, final_state, t0)

        if self._history:
            await self._history.append(session_id, "user", req.message)
            await self._history.append(session_id, "assistant", response.answer)
            asyncio.create_task(save_chat_message_async(session_id, "user",      req.message,    req.user_id, req.tenant_id))
            asyncio.create_task(save_chat_message_async(session_id, "assistant", response.answer, req.user_id, req.tenant_id))

            ctx_to_save: dict = {}
            if response.project_name:
                ctx_to_save["project_name"] = response.project_name
            if ctx_to_save:
                await self._history.set_context(session_id, ctx_to_save)

        log.info("sale_chat_done", session_id=session_id,
                 fallback=response.fallback, ms=response.response_time_ms)
        return response

    async def execute_stream(self, req: ChatRequest):
        """Stream mode cho Sale."""
        import json
        session_id = req.session_id or _new_session_id()
        t0 = time.monotonic()
        log.info("sale_chat_stream_start", session_id=session_id)

        queue = asyncio.Queue()
        state = make_sale_initial_state(
            session_id=session_id,
            raw_query=req.message,
            project_name=req.project_name if not _is_invalid_project(req.project_name) else None,
            customer_name=req.customer_name,
            customer_phone=req.customer_phone,
        )
        state["stream_queue"] = queue

        if self._history:
            history = await self._history.get_history(session_id, limit=20)
            state["messages"] = history
            ctx = await self._history.get_context(session_id)
            if _is_invalid_project(req.project_name):
                state["project_name"] = ctx.get("project_name")
            else:
                state["project_name"] = req.project_name

        if req.customer_name or req.customer_phone:
            state["sales_data"] = {
                "customer_name":  req.customer_name or "",
                "customer_phone": req.customer_phone or "",
            }

        async for chunk in _stream_graph(self._graph, state, queue, session_id, req, self._history, t0):
            yield chunk


# ── Shared stream helper ──────────────────────────────────────────

async def _stream_graph(graph, state, queue, session_id, req, history, t0):
    """Generator helper chung cho cả Customer và Sale stream."""
    import json

    # Filler khi xử lý lâu > 3s
    async def filler_task(q):
        try:
            await asyncio.sleep(3.0)
            filler = "Dạ em đang kiểm tra thông tin, anh/chị đợi một chút nhé...\n\n"
            for i, word in enumerate(filler.split(" ")):
                chunk = word + (" " if i < len(filler.split(" ")) - 1 else "")
                await q.put({"type": "filler_token", "content": chunk})
                await asyncio.sleep(0.12)
        except asyncio.CancelledError:
            pass

    filler_bg  = asyncio.create_task(filler_task(queue))
    graph_task = asyncio.create_task(graph.ainvoke(state))

    def _on_done(task):
        try:
            if task.exception():
                log.error("stream_graph_task_exception", error=str(task.exception()))
                try:
                    queue.put_nowait({"type": "token", "content": "Hệ thống đang bận, vui lòng thử lại."})
                except Exception:
                    pass
        except asyncio.CancelledError:
            pass
        finally:
            try:
                queue.put_nowait({"type": "done"})
            except Exception:
                pass

    graph_task.add_done_callback(_on_done)

    real_token_emitted = False
    final_state = None

    try:
        while True:
            chunk = await queue.get()
            if chunk["type"] == "done":
                break
            elif chunk["type"] == "token":
                if not real_token_emitted:
                    real_token_emitted = True
                    if not filler_bg.done():
                        filler_bg.cancel()
                yield f"data: {json.dumps({'text': chunk['content'], 'session_id': session_id})}\n\n"
            elif chunk["type"] == "filler_token":
                yield f"data: {json.dumps({'text': chunk['content'], 'session_id': session_id})}\n\n"
            elif chunk["type"] == "suggestions":
                yield f"data: {json.dumps({'suggested_questions': chunk['content'], 'session_id': session_id})}\n\n"
    finally:
        if not filler_bg.done():
            filler_bg.cancel()

    task_exc = None
    try:
        task_exc = graph_task.exception()
    except asyncio.CancelledError:
        pass

    if task_exc:
        log.error("stream_graph_crash", error=str(task_exc))
        yield f"data: {json.dumps({'text': 'Hệ thống đang bận, vui lòng thử lại.', 'session_id': session_id})}\n\n"
    else:
        final_state = graph_task.result()
        if not real_token_emitted and final_state:
            fallback_answer = final_state.get("final_answer", "")
            if fallback_answer:
                yield f"data: {json.dumps({'text': fallback_answer, 'session_id': session_id})}\n\n"

    if final_state is None:
        final_state = state

    if history:
        await history.append(session_id, "user", req.message)
        answer = final_state.get("final_answer", "")
        await history.append(session_id, "assistant", answer)
        asyncio.ensure_future(asyncio.shield(save_chat_message_async(session_id, "user",      req.message, req.user_id, req.tenant_id)))
        asyncio.ensure_future(asyncio.shield(save_chat_message_async(session_id, "assistant", answer,      req.user_id, req.tenant_id)))

        ctx_to_save: dict = {}
        if final_state.get("project_name"):
            ctx_to_save["project_name"] = final_state["project_name"]
        if ctx_to_save:
            await history.set_context(session_id, ctx_to_save)

    metadata = {
        "sources":   [{"doc": s.document_name, "excerpt": s.excerpt} for s in _map_sources(final_state.get("sources", []))],
        "intent":    str(final_state.get("intent", "unknown")),
        "session_id": session_id,
    }
    yield f"data: {json.dumps(metadata)}\n\n"
    yield "data: [DONE]\n\n"


# ── Backward-compat alias ─────────────────────────────────────────
HandleChatUseCase = HandleCustomerChatUseCase