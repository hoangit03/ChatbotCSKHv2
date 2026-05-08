"""
app/agent/nodes/stage_classifier.py

[NEW FILE] Customer Journey Stage Classifier.

Phân tích lịch sử hội thoại để xác định khách hàng đang ở giai đoạn nào
trong hành trình mua bất động sản:
  - AWARENESS     (Giai đoạn 1): Mới quan tâm → chốt lịch hẹn
  - CONSIDERATION (Giai đoạn 2): Đang đánh giá → đẩy cảm xúc
  - DECISION      (Giai đoạn 3): Sắp chốt → xóa rủi ro, tạo khan hiếm

Thiết kế:
  - Chạy SONG SONG với classify_intent — không thêm latency
  - Dùng rule-based TRƯỚC, LLM làm fallback (tiết kiệm token)
  - Stage được persist giữa các lượt (chỉ upgrade, không downgrade)
  - Expose stage signals để debug/audit

Stage chỉ upgrade (AWARENESS → CONSIDERATION → DECISION), không đi ngược lại,
trừ khi khách chuyển sang dự án mới (project_newly_confirmed = True).
"""
from __future__ import annotations

import re
from typing import Optional

from app.agent.state.agent_state import AgentState, CustomerStage, Intent
from app.shared.logging.logger import get_logger

log = get_logger(__name__)


# ── Signal patterns — Rule-based detection ────────────────────────

# Dấu hiệu CONSIDERATION (Giai đoạn 2)
_CONSIDERATION_PATTERNS = [
    r"tiện ích",
    r"môi trường sống",
    r"quy hoạch",
    r"sa bàn",
    r"nhà mẫu",
    r"xem (nhà|căn|dự án|thực tế)",
    r"thiết kế",
    r"kiến trúc",
    r"tầng hầm",
    r"hồ bơi",
    r"công viên",
    r"tiện nghi",
    r"view",
    r"hướng",
    r"tầng (cao|thấp|\d+)",
    r"layout",
    r"mặt bằng",
    r"đã đến",
    r"vừa xem",
]

# Dấu hiệu DECISION (Giai đoạn 3)
_DECISION_PATTERNS = [
    # Hỏi mã căn cụ thể
    r"căn\s+[A-Z0-9][\w\-]+",
    r"mã căn",
    r"unit\s+[A-Z0-9]",
    # Tính toán tài chính
    r"dòng tiền",
    r"tiến độ thanh toán",
    r"vay (ngân hàng|thêm|bao nhiêu)",
    r"trả (góp|hàng tháng|mỗi tháng)",
    r"lãi suất",
    r"đặt cọc",
    r"giữ chỗ",
    r"booking",
    r"cọc",
    # So sánh dự án khác
    r"so (với|sánh)",
    r"dự án (khác|bên kia|đối thủ)",
    r"tại sao (chọn|mua|nên)",
    r"(hơn|kém) dự án",
    # Pháp lý chi tiết
    r"pháp lý",
    r"sổ (hồng|đỏ)",
    r"giấy phép",
    r"bảo lãnh",
    r"hợp đồng",
    r"tiến độ (xây|bàn giao|thi công)",
    # Xác nhận mua
    r"(muốn|quyết định|đồng ý).{0,20}(mua|cọc|đặt)",
    r"(mua|lấy|chốt)\s*(căn|cái|nó)",
]

# Dấu hiệu AWARENESS (Giai đoạn 1 — mặc định, không cần detect)
# Tất cả những gì không thuộc consideration/decision đều là awareness.


def _normalize(text: str) -> str:
    """Lowercase, loại bỏ dấu câu thừa để regex dễ match."""
    return text.lower().strip()


def _check_patterns(text: str, patterns: list[str]) -> list[str]:
    """Trả về list pattern đã match."""
    norm = _normalize(text)
    return [p for p in patterns if re.search(p, norm)]


def detect_stage_from_messages(
    messages: list[dict],
    current_stage: str,
    current_query: str,
) -> tuple[CustomerStage, list[str]]:
    """
    Phân tích messages + query để xác định CustomerStage.

    Logic:
      1. Lấy current_stage làm baseline (không downgrade)
      2. Check query + 4 messages gần nhất
      3. DECISION signals → override về DECISION
      4. CONSIDERATION signals → upgrade nếu đang AWARENESS
      5. Stage chỉ đi lên, không đi xuống

    Returns:
        (CustomerStage, list[str]) — stage mới và signals đã detect
    """
    signals: list[str] = []

    # Combine query + recent messages để check
    recent_content = current_query
    for msg in messages[-4:]:
        content = msg.get("content", "")
        if content:
            recent_content += " " + content

    # Check DECISION signals (priority cao nhất)
    decision_hits = _check_patterns(recent_content, _DECISION_PATTERNS)
    if decision_hits:
        signals.extend([f"decision:{p}" for p in decision_hits[:3]])  # Chỉ log 3 signal đầu

    # Check CONSIDERATION signals
    consideration_hits = _check_patterns(recent_content, _CONSIDERATION_PATTERNS)
    if consideration_hits:
        signals.extend([f"consideration:{p}" for p in consideration_hits[:3]])

    # Determine new stage (chỉ upgrade)
    baseline = CustomerStage(current_stage) if current_stage else CustomerStage.AWARENESS

    if decision_hits:
        new_stage = CustomerStage.DECISION
    elif consideration_hits and baseline == CustomerStage.AWARENESS:
        new_stage = CustomerStage.CONSIDERATION
    else:
        new_stage = baseline  # Giữ nguyên hoặc baseline

    # Stage chỉ upgrade, không downgrade
    stage_order = {
        CustomerStage.AWARENESS: 0,
        CustomerStage.CONSIDERATION: 1,
        CustomerStage.DECISION: 2,
    }
    if stage_order.get(new_stage, 0) < stage_order.get(baseline, 0):
        new_stage = baseline

    return new_stage, signals


def classify_customer_stage(state: AgentState) -> AgentState:
    """
    Node helper — cập nhật customer_stage trong state.

    Được gọi TRONG classify_intent (không phải node riêng)
    để tránh thêm một LLM call.

    Nếu project vừa được switch (project_newly_confirmed = True),
    reset về AWARENESS vì khách hàng đang bắt đầu tìm hiểu dự án mới.
    """
    # Reset stage khi chuyển dự án mới
    if state.get("project_newly_confirmed"):
        state["customer_stage"] = CustomerStage.AWARENESS
        state["stage_signals"] = ["project_switched"]
        log.info(
            "stage_reset_new_project",
            session=state.get("session_id"),
            project=state.get("project_name"),
        )
        return state

    current_stage = state.get("customer_stage", CustomerStage.AWARENESS)
    messages = state.get("messages") or []
    query = state.get("raw_query", "")

    new_stage, signals = detect_stage_from_messages(
        messages=messages,
        current_stage=current_stage,
        current_query=query,
    )

    old_stage = current_stage
    state["customer_stage"] = new_stage
    state["stage_signals"] = signals

    if new_stage != old_stage:
        log.info(
            "stage_upgraded",
            session=state.get("session_id"),
            from_stage=old_stage,
            to_stage=new_stage.value,
            signals=signals[:3],
        )

    return state


def get_stage_display(stage: str) -> str:
    """Helper cho logging/debug — trả về tên giai đoạn dễ đọc."""
    mapping = {
        CustomerStage.AWARENESS:     "Giai đoạn 1 — Quan Tâm",
        CustomerStage.CONSIDERATION: "Giai đoạn 2 — Tiềm Năng",
        CustomerStage.DECISION:      "Giai đoạn 3 — Sắp Chốt",
    }
    try:
        return mapping.get(CustomerStage(stage), stage)
    except ValueError:
        return stage