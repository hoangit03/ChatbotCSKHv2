"""
app/agent/tools/sales_tool.py

Các tool gọi Sales Backend API.
Tách thành nhiều tool nhỏ (ISP) thay vì một tool lớn:
  - AvailabilityTool  : kiểm tra căn hộ còn trống
  - InventoryTool     : tổng tồn kho dự án
  - PaymentTool       : chính sách thanh toán/vay
  - UnitSearchTool    : tìm căn theo tiêu chí
  - BookingIntentTool : đăng ký đặt cọc/giữ chỗ
"""
from __future__ import annotations

import re
from typing import Optional

from app.agent.state.agent_state import AgentState
from app.agent.tools.base_tool import AgentTool, ToolResult
from app.core.interfaces.sales_api_port import SalesAPIPort
from app.shared.errors.exceptions import SalesAPIError
from app.shared.logging.logger import get_logger

log = get_logger(__name__)


def _project(state: AgentState) -> str:
    return state.get("project_name") or "unknown"


def _fmt_vnd(amount: float) -> str:
    """3_500_000_000 → '3,5 tỷ VNĐ'"""
    if amount >= 1_000_000_000:
        return f"{amount / 1_000_000_000:.1f} tỷ VNĐ"
    if amount >= 1_000_000:
        return f"{amount / 1_000_000:.0f} triệu VNĐ"
    return f"{amount:,.0f} VNĐ"


# ── Tool 1: Availability ──────────────────────────────────────────

class AvailabilityTool(AgentTool):

    def __init__(self, api: SalesAPIPort):
        self._api = api

    @property
    def name(self) -> str:
        return "check_availability"

    @property
    def description(self) -> str:
        return "Kiểm tra căn hộ còn trống không. Dùng khi hỏi 'còn căn không', 'căn X còn chưa'."

    async def run(self, state: AgentState) -> ToolResult:
        # Trích unit_code từ query nếu có (e.g. "căn A1-05")
        unit_code = _extract_unit_code(state["raw_query"])
        try:
            units = await self._api.get_unit_availability(
                project=_project(state),
                unit_code=unit_code,
            )
            available = [u for u in units if u.status == "available"]

            if not available:
                summary = "Không còn căn hộ trống theo tiêu chí yêu cầu."
                state["sales_data"]["availability"] = {"available": 0, "units": []}
            else:
                unit_lines = [
                    f"  • {u.unit_code}: {u.bedrooms}PN, {u.area_m2}m², {_fmt_vnd(u.price_vnd)}"
                    for u in available[:5]
                ]
                summary = f"Còn {len(available)} căn trống:\n" + "\n".join(unit_lines)
                state["sales_data"]["availability"] = {
                    "available": len(available),
                    "units": [
                        {
                            "unit_code": u.unit_code,
                            "bedrooms": u.bedrooms,
                            "area_m2": u.area_m2,
                            "price_vnd": u.price_vnd,
                            "price_formatted": _fmt_vnd(u.price_vnd),
                            "floor": u.floor,
                        }
                        for u in available
                    ],
                }

            return ToolResult(success=True, data=state["sales_data"]["availability"], summary=summary)

        except SalesAPIError as e:
            return ToolResult(success=False, data=None, summary=f"Lỗi API: {e.message}", error=str(e))


# ── Tool 2: Inventory ─────────────────────────────────────────────

class InventoryTool(AgentTool):

    def __init__(self, api: SalesAPIPort):
        self._api = api

    @property
    def name(self) -> str:
        return "get_inventory"

    @property
    def description(self) -> str:
        return "Lấy tổng số căn hộ còn lại của dự án."

    async def run(self, state: AgentState) -> ToolResult:
        try:
            inv = await self._api.get_project_inventory(_project(state))
            state["sales_data"]["inventory"] = {
                "total": inv.total_units,
                "available": inv.available,
                "reserved": inv.reserved,
                "sold": inv.sold,
            }
            summary = (
                f"Dự án {inv.project}: tổng {inv.total_units} căn, "
                f"còn trống {inv.available}, đặt cọc {inv.reserved}, đã bán {inv.sold}."
            )
            return ToolResult(success=True, data=state["sales_data"]["inventory"], summary=summary)
        except SalesAPIError as e:
            return ToolResult(success=False, data=None, summary=f"Lỗi API: {e.message}", error=str(e))


# ── Tool 3: Payment Policy ────────────────────────────────────────

class PaymentTool(AgentTool):

    def __init__(self, api: SalesAPIPort):
        self._api = api

    @property
    def name(self) -> str:
        return "get_payment_policy"

    @property
    def description(self) -> str:
        return "Lấy chính sách thanh toán, vay vốn, trả góp của dự án."

    async def run(self, state: AgentState) -> ToolResult:
        try:
            policies = await self._api.get_payment_policies(_project(state))
            if not policies:
                return ToolResult(success=False, data=None, summary="Không tìm thấy chính sách thanh toán.")

            state["sales_data"]["payment_policies"] = [
                {"name": p.name, "description": p.description} for p in policies
            ]
            summary = f"Tìm được {len(policies)} chính sách thanh toán: " + ", ".join(p.name for p in policies)
            return ToolResult(success=True, data=state["sales_data"]["payment_policies"], summary=summary)
        except SalesAPIError as e:
            return ToolResult(success=False, data=None, summary=f"Lỗi API: {e.message}", error=str(e))


# ── Tool 4: Unit Search ───────────────────────────────────────────

class UnitSearchTool(AgentTool):

    def __init__(self, api: SalesAPIPort):
        self._api = api

    @property
    def name(self) -> str:
        return "search_units"

    @property
    def description(self) -> str:
        return "Tìm căn hộ theo tiêu chí: số phòng ngủ, diện tích, giá tối đa."

    async def run(self, state: AgentState) -> ToolResult:
        query = state["raw_query"]
        bedrooms = _extract_bedrooms(query)
        min_price = _extract_min_price(query)
        max_price = _extract_max_price(query)
        min_area = _extract_min_area(query)
        max_area = _extract_max_area(query)
        floor = _extract_floor(query)
        direction = _extract_direction(query)

        try:
            units = await self._api.search_units(
                project=_project(state),
                bedrooms=bedrooms,
                min_price_vnd=min_price,
                max_price_vnd=max_price,
                min_area_m2=min_area,
                max_area_m2=max_area,
                direction=direction,
                floor=floor,
            )
            if not units:
                return ToolResult(success=False, data=[], summary="Không tìm thấy căn hộ phù hợp.")

            state["sales_data"]["search_results"] = [
                {
                    "unit_code": u.unit_code,
                    "bedrooms": u.bedrooms,
                    "area_m2": u.area_m2,
                    "price_vnd": u.price_vnd,
                    "price_formatted": _fmt_vnd(u.price_vnd),
                    "total_price": _fmt_vnd(u.total_price) if u.total_price else _fmt_vnd(u.price_vnd),
                    "floor": u.floor,
                    "direction": u.direction,
                    "sale_program": u.sale_program,
                }
                for u in units[:10]
            ]
            summary = f"Tìm được {len(units)} căn phù hợp."
            return ToolResult(success=True, data=state["sales_data"]["search_results"], summary=summary)
        except SalesAPIError as e:
            return ToolResult(success=False, data=None, summary=f"Lỗi API: {e.message}", error=str(e))


# ── Tool 5: Booking Intent ────────────────────────────────────────

class BookingIntentTool(AgentTool):
    """Trigger đặt cọc / giữ chỗ sang hệ thống backend."""

    def __init__(self, api: SalesAPIPort):
        self._api = api

    @property
    def name(self) -> str:
        return "booking_intent"

    @property
    def description(self) -> str:
        return "Gửi yêu cầu đặt cọc/giữ chỗ khi khách hàng đã quyết định mua."

    async def run(self, state: AgentState) -> ToolResult:
        # Lấy thông tin khách từ state (đã thu thập trước đó)
        sales_data = state.get("sales_data", {})
        customer_name = sales_data.get("customer_name", "")
        customer_phone = sales_data.get("customer_phone", "")
        unit_code = sales_data.get("selected_unit_code", "")

        if not all([customer_name, customer_phone, unit_code]):
            return ToolResult(
                success=False,
                data=None,
                summary="Cần thu thập đủ thông tin: họ tên, SĐT, và mã căn hộ.",
            )

        try:
            result = await self._api.trigger_booking_intent(
                project=_project(state),
                unit_code=unit_code,
                customer_name=customer_name,
                customer_phone=customer_phone,
            )
            state["sales_data"]["booking"] = {
                "booking_id": result.booking_id,
                "success": result.success,
                "message": result.message,
            }
            summary = f"Đặt cọc {'thành công' if result.success else 'thất bại'}: {result.message}"
            return ToolResult(success=result.success, data=result, summary=summary)
        except SalesAPIError as e:
            return ToolResult(success=False, data=None, summary=f"Lỗi đặt cọc: {e.message}", error=str(e))


# ── Helpers ────────────────────────────────────────────────────────

def _extract_unit_code(text: str) -> Optional[str]:
    m = re.search(r"\b([A-Z]\d{1,2}-\d{2,3})\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def _extract_bedrooms(text: str) -> Optional[int]:
    m = re.search(r"(\d)\s*(?:phòng ngủ|pn|bedroom|br)", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _extract_max_price(text: str) -> Optional[float]:
    # "dưới 5 tỷ", "max 3.5 tỷ", "không quá 4 tỷ"
    m = re.search(
        r"(?:dưới|không quá|tối đa|max|<)\s*(\d+(?:[.,]\d+)?)\s*(tỷ|triệu)",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    val = float(m.group(1).replace(",", "."))
    unit = m.group(2).lower()
    return val * 1_000_000_000 if "tỷ" in unit else val * 1_000_000

def _extract_min_price(text: str) -> Optional[float]:
    m = re.search(r"(?:từ|trên|>)\s*(\d+(?:[.,]\d+)?)\s*(tỷ|triệu)", text, re.IGNORECASE)
    if not m: return None
    val = float(m.group(1).replace(",", "."))
    unit = m.group(2).lower()
    return val * 1_000_000_000 if "tỷ" in unit else val * 1_000_000

def _extract_floor(text: str) -> Optional[str]:
    m = re.search(r"tầng\s*(\d{1,3}[a-zA-Z]?)", text, re.IGNORECASE)
    return m.group(1).upper() if m else None

def _extract_direction(text: str) -> Optional[str]:
    directions = ["đông nam", "tây nam", "đông bắc", "tây bắc", "đông tứ trạch", "tây tứ trạch", "đông", "tây", "nam", "bắc"]
    for d in directions:
        if re.search(rf"\b{d}\b", text, re.IGNORECASE):
            return d.capitalize()
    return None

def _extract_min_area(text: str) -> Optional[float]:
    m = re.search(r"(?:từ|trên|>)\s*(\d+(?:[.,]\d+)?)\s*m2", text, re.IGNORECASE)
    return float(m.group(1).replace(",", ".")) if m else None

def _extract_max_area(text: str) -> Optional[float]:
    m = re.search(r"(?:dưới|không quá|tối đa|<)\s*(\d+(?:[.,]\d+)?)\s*m2", text, re.IGNORECASE)
    return float(m.group(1).replace(",", ".")) if m else None
