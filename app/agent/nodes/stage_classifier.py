"""
app/agent/nodes/stage_classifier.py

DEPRECATED — Không còn sử dụng kể từ v2.

CustomerStage (Awareness/Consideration/Decision) và USP injection
đã bị loại bỏ khỏi hệ thống. Sale và Customer giờ dùng Graph riêng biệt.

File này chỉ giữ lại để tránh ImportError từ code cũ đang chờ cleanup.
Không import module này vào code mới.
"""


def classify_customer_stage(state):  # type: ignore[return]
    """DEPRECATED: không làm gì, trả về state nguyên."""
    return state