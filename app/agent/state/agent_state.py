"""
app/agent/state/agent_state.py

AgentState là trái tim của LangGraph.
Mọi node đọc/ghi vào đây — không dùng biến global.

Tại sao LangGraph tốt hơn LangChain Agent:
  - State machine rõ ràng, mọi bước được kiểm soát
  - Dễ debug: log từng node
  - Controllable: có thể interrupt, replay, branch
  - Không bị "hallucinate tool call" vì graph định sẵn luồng
  - Dễ test từng node riêng biệt
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TypedDict


class Intent(str, Enum):
    """
    Agent phân loại intent trước khi chọn tool.
    Tránh dùng LLM gọi tool sai.
    """
    CUSTOMER_SUPPORT = "customer_support"   # hỏi về dự án, pháp lý, tiện ích
    SALES_INQUIRY    = "sales_inquiry"      # hỏi giá, tồn kho, đặt cọc
    BOOKING_INTENT   = "booking_intent"     # muốn đặt cọc/giữ chỗ
    APPOINTMENT_INTENT  = "appointment_intent"  # muốn đặt lịch xem nhà mẫu/sa bàn
    COMPARISON_INTENT   = "comparison_intent"   # đang so sánh với dự án đối thủ
    CHITCHAT         = "chitchat"           # chào hỏi, tán gẫu, câu hỏi chung
    UNKNOWN          = "unknown"


class CustomerStage(str, Enum):
    """
    Ba giai đoạn tâm lý khách hàng theo kịch bản sales.
 
    AWARENESS     — Giai đoạn 1: Mới quan tâm, chưa hiểu dự án.
                    Mục tiêu: Tạo tò mò, chốt lịch hẹn xem sa bàn.
                    Vũ khí USP: vị trí, cầu metro, 0% lãi suất.
                    KHÔNG báo giá chi tiết.
 
    CONSIDERATION — Giai đoạn 2: Đã đến xem, đang đánh giá.
                    Mục tiêu: Đẩy cảm xúc, trải nghiệm đẳng cấp.
                    Vũ khí USP: CBD, kiến trúc, hầm xe, lễ hội ánh sáng.
 
    DECISION      — Giai đoạn 3: Đang cân nhắc chốt, hỏi cụ thể.
                    Mục tiêu: Xóa rủi ro, tạo khan hiếm, chốt deal.
                    Vũ khí USP: pháp lý 2026, bảo lãnh, bảo chứng tăng giá.
    """
    AWARENESS     = "awareness"
    CONSIDERATION = "consideration"
    DECISION      = "decision"

class ScarcityLevel(str, Enum):
    """
    Mức độ khan hiếm căn hộ — dùng để inject FOMO messaging.
    Được tính toán dựa trên số căn còn trống từ inventory/search.
    """
    NONE     = "none"      # > 10 căn — không cần FOMO
    LOW      = "low"       # 6–10 căn — nhắc nhẹ
    MEDIUM   = "medium"    # 3–5 căn — nhấn mạnh
    CRITICAL = "critical"  # ≤ 2 căn  — FOMO mạnh, cần hành động ngay

@dataclass
class SourceRef:
    """Trích dẫn nguồn trả lời."""
    document_code: str
    document_name: str
    doc_group: str
    excerpt: str
    page: Optional[int] = None


@dataclass
class ToolCall:
    """Record một lần gọi tool — để audit/debug."""
    tool_name: str
    input_summary: str
    output_summary: str
    duration_ms: int
    success: bool

@dataclass
class USPItem:
    """
    Một điểm bán hàng độc đáo (Unique Selling Point).
 
    Mỗi dự án sẽ có bộ USP riêng, được load từ config/DB.
    stage: giai đoạn phù hợp để dùng USP này.
    """
    usp_id: str             # Ví dụ: "USP_3", "USP_11"
    title: str              # Ví dụ: "5 năm 0% lãi suất"
    description: str        # Nội dung chi tiết để inject vào prompt
    stage: CustomerStage    # Giai đoạn phù hợp
    priority: int = 0       # Thứ tự ưu tiên (thấp hơn = quan trọng hơn)

@dataclass
class AppointmentResult:
    """
    [NEW] Kết quả đặt lịch hẹn xem nhà mẫu/sa bàn.
    Khác với BookingResult (đặt cọc căn hộ thực).
    """
    success: bool
    appointment_id: str
    message: str
    scheduled_date: Optional[str] = None   # ISO date string
    scheduled_time: Optional[str] = None   # HH:MM
    location: Optional[str] = None         # Tên Sale Gallery / địa chỉ

class AgentState(TypedDict):
    """
    LangGraph State — kế thừa TypedDict để compat với StateGraph.

    Keys:
      messages       : lịch sử hội thoại (list message dạng dict)
      session_id     : định danh session
      project_name   : dự án đang hỏi (optional filter)
      intent         : Intent enum sau khi classify
      raw_query      : query gốc (sau sanitize)
      was_injected   : cờ injection attempt
      rag_results    : list[dict] từ vector search
      qa_result      : QAItem nếu match
      sales_data     : dict từ sales API
      final_answer   : câu trả lời cuối
      sources        : list[SourceRef]
      tool_calls     : list[ToolCall] — audit trail
      fallback       : True nếu không đủ dữ liệu
      fallback_reason: lý do fallback
      iteration      : số iteration hiện tại
      error          : lỗi nếu có
      tool_kwargs    : arguments trích xuất từ LLM cho các tool
      customer_stage          : Giai đoạn tâm lý của khách hàng
      stage_signals           : Dấu hiệu nhận biết giai đoạn (list[str])
      usps_used               : USP IDs đã dùng trong session (tránh lặp)
      appointment_booked      : Đã chốt lịch hẹn xem nhà mẫu chưa
      scarcity_level          : Mức độ khan hiếm căn hộ
      cross_sell_suggestions  : Căn đề xuất cross-sell khi căn đã bán
      booking_confirmation    : [NEW] Khách đã xác nhận booking (multi-turn)
      price_disclosed         : [NEW] Giá đã được tiết lộ chưa (stage guard)
    """
    # ── Core ─────
    messages: list
    session_id: str
    project_name: Optional[str]
    min_role_level: Optional[int]
    intent: Any                  # Intent enum
    raw_query: str
    was_injected: bool

    # ── RAG / QA ────
    rag_results: list
    qa_result: Optional[Any]     # QAItem | None
    qa_hit: bool                 # True nếu QATool tìm thấy match
    query_embedding: Optional[list[float]] # Lưu embedding của query để dùng chung
    
    # ── Sales ──────
    sales_data: dict
    final_answer: str
    sources: list
    tool_calls: list
    fallback: bool
    fallback_reason: str
    iteration: int
    error: Optional[str]

    # ── Project / Customer identity ────
    project_newly_confirmed: bool       # True khi guard vừa switch sang project mới
    customer_name: Optional[str]        # Tên khách hàng (từ ChatRequest)
    customer_phone: Optional[str]       # SĐT khách hàng (từ ChatRequest)
    tool_kwargs: dict                   # Lưu params cho tools
    
    # ── Customer journey / Stage ────
    customer_stage: str                 # CustomerStage enum value
    stage_signals: list                 # list[str] — dấu hiệu nhận biết
    usps_used: list                     # list[str] — USP IDs đã inject
    appointment_booked: bool            # Đã đặt lịch hẹn xem nhà mẫu chưa
    scarcity_level: str                 # ScarcityLevel enum value
    cross_sell_suggestions: list        # list[dict] — căn đề xuất
    booking_confirmation: bool          # Khách đã confirm booking chưa
    price_disclosed: bool               # Giá đã được cho phép tiết lộ chưa


# ── Factory ────────────────────────────────────────────────────────
 
def make_initial_state(
    session_id: str,
    raw_query: str,
    project_name: str | None = None,
    customer_name: str | None = None,
    customer_phone: str | None = None,
    customer_stage: str = CustomerStage.AWARENESS,
    usps_used: list | None = None,
    appointment_booked: bool = False,
    min_role_level: int | None = None,
) -> AgentState:
    return AgentState(
        # Core
        messages=[],
        session_id=session_id,
        project_name=project_name,
        min_role_level=min_role_level,
        intent=Intent.UNKNOWN,
        raw_query=raw_query,
        was_injected=False,
        # RAG / QA
        rag_results=[],
        qa_result=None,
        qa_hit=False,
        query_embedding=None,
        # Sales
        sales_data={},
        final_answer="",
        sources=[],
        tool_calls=[],
        fallback=False,
        fallback_reason="",
        iteration=0,
        error=None,
        project_newly_confirmed=False,
        customer_name=customer_name,
        customer_phone=customer_phone,
        tool_kwargs={},
        customer_stage=customer_stage,
        stage_signals=[],
        usps_used=usps_used or [],
        appointment_booked=appointment_booked,
        scarcity_level=ScarcityLevel.NONE,
        cross_sell_suggestions=[],
        booking_confirmation=False,
        price_disclosed=False,
    )