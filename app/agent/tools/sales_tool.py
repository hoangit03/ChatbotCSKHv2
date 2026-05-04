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

from app.agent.state.agent_state import AgentState, ScarcityLevel
from app.agent.tools.base_tool import AgentTool, ToolResult
from app.core.interfaces.sales_api_port import SalesAPIPort
from app.shared.errors.exceptions import SalesAPIError
from app.shared.logging.logger import get_logger

log = get_logger(__name__)

# Ngưỡng scarcity — có thể config
_SCARCITY_CRITICAL = 2
_SCARCITY_MEDIUM   = 5
_SCARCITY_LOW      = 10


def _project(state: AgentState) -> str:
    return state.get("project_name") or "unknown"


def _fmt_vnd(amount: float) -> str:
    """3_500_000_000 → '3,5 tỷ VNĐ'"""
    if amount >= 1_000_000_000:
        return f"{amount / 1_000_000_000:.1f} tỷ VNĐ"
    if amount >= 1_000_000:
        return f"{amount / 1_000_000:.0f} triệu VNĐ"
    return f"{amount:,.0f} VNĐ"


def _compute_scarcity(available_count: int) -> str:
    """
    [NEW] Tính ScarcityLevel từ số căn còn trống.
    Được dùng để inject FOMO messaging trong Synthesizer.
    """
    if available_count <= _SCARCITY_CRITICAL:
        return ScarcityLevel.CRITICAL
    if available_count <= _SCARCITY_MEDIUM:
        return ScarcityLevel.MEDIUM
    if available_count <= _SCARCITY_LOW:
        return ScarcityLevel.LOW
    return ScarcityLevel.NONE

def _unit_to_dict(u) -> dict:
    """Convert UnitAvailability → dict chuẩn cho sales_data."""
    return {
        "unit_code":       u.unit_code,
        "bedrooms":        u.bedrooms,
        "area_m2":         u.area_m2,
        "price_vnd":       u.price_vnd,
        "price_formatted": _fmt_vnd(u.price_vnd),
        "total_price":     _fmt_vnd(u.total_price) if u.total_price else _fmt_vnd(u.price_vnd),
        "floor":           u.floor,
        "direction":       u.direction,
        "sale_program":    u.sale_program,
        "maintenance_fee": _fmt_vnd(u.maintenance_fee) if u.maintenance_fee else None,
        "status":          u.status,
    }

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
        kwargs = state.get("tool_kwargs", {}).get(self.name, {})
        unit_code = kwargs.get("unit_code")
        try:
            summary = ""
            units = await self._api.get_unit_availability(
                project=_project(state),
                unit_code=unit_code,
            )
            available = [u for u in units if u.status == "available"]
            unavailable = [u for u in units if u.status in ("reserved", "sold")]
 
            scarcity = _compute_scarcity(len(available))
            state["scarcity_level"] = scarcity

            if not available:
                summary = "Không còn căn hộ trống theo tiêu chí yêu cầu."
                state["sales_data"]["availability"] = {"available": 0, "units": []}
                if unavailable and unit_code:
                    ref = unavailable[0]
                    cross_units = await self._api.search_units(
                        project=_project(state),
                        bedrooms=ref.bedrooms,
                        min_price_vnd=ref.price_vnd * 0.85,
                        max_price_vnd=ref.price_vnd * 1.15,
                        status="available",
                    )
                    if cross_units:
                        state["cross_sell_suggestions"] = [
                            _unit_to_dict(u) for u in cross_units[:3]
                        ]
                        state["sales_data"]["cross_sell"] = state["cross_sell_suggestions"]
                        summary += (
                            f" Tuy nhiên, em tìm được {len(cross_units)} căn "
                            f"tương đương đang còn trống."
                        )
                        log.info(
                            "cross_sell_triggered",
                            unit_code=unit_code,
                            suggestions=len(cross_units),
                        )
            else:
                unit_lines = [
                    f"  • {u.unit_code}: {u.bedrooms}PN, {u.area_m2}m², {_fmt_vnd(u.price_vnd)}"
                    for u in available[:5]
                ]
                
                summary = f"Tìm thấy {len(available)} căn hộ đang còn trống:\n" + "\n".join(unit_lines)

                if scarcity == ScarcityLevel.CRITICAL:
                    summary += f"\n\nCHỈ CÒN {len(available)} CĂN — cần hành động ngay!"
                elif scarcity == ScarcityLevel.MEDIUM:
                    summary += f"\n\nCòn {len(available)} căn — số lượng có hạn."
 
                state["sales_data"]["availability"] = {
                    "available": len(available),
                    "scarcity_level": scarcity,
                    "units": [_unit_to_dict(u) for u in available],
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
            scarcity = _compute_scarcity(inv.available)
            state["scarcity_level"] = scarcity

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

            if scarcity == ScarcityLevel.CRITICAL:
                summary += f" Chỉ còn {inv.available} căn — rất khan hiếm!"
            elif scarcity == ScarcityLevel.MEDIUM:
                summary += f" Còn {inv.available} căn — số lượng có hạn."
 
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
        kwargs = state.get("tool_kwargs", {}).get(self.name, {})
        bedrooms = kwargs.get("bedrooms")
        min_price = kwargs.get("min_price_vnd")
        max_price = kwargs.get("max_price_vnd")
        min_area = kwargs.get("min_area_m2")
        max_area = kwargs.get("max_area_m2")
        floor = kwargs.get("floor")
        direction = kwargs.get("direction")

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

            available_units = [u for u in units if u.status == "available"]
 
            # Tính scarcity từ kết quả search
            scarcity = _compute_scarcity(len(available_units))
            state["scarcity_level"] = scarcity
 
            state["sales_data"]["search_results"] = [_unit_to_dict(u) for u in units[:10]]
            state["sales_data"]["search_scarcity"] = scarcity
 
            summary = f"Tìm được {len(units)} căn phù hợp ({len(available_units)} còn trống)."
            if scarcity == ScarcityLevel.CRITICAL:
                summary += f" Chỉ còn {len(available_units)} căn available!"
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
        # Lấy thông tin từ state
        sales_data = state.get("sales_data", {})
        customer_name = state.get("customer_name") or sales_data.get("customer_name", "")
        customer_phone = state.get("customer_phone") or sales_data.get("customer_phone", "")
        
        kwargs = state.get("tool_kwargs", {}).get(self.name, {})
        unit_code = kwargs.get("unit_code") or sales_data.get("selected_unit_code", "")

        missing = []
        if not customer_name:
            missing.append("họ và tên")
        if not customer_phone:
            missing.append("số điện thoại")
        if not unit_code:
            missing.append("mã căn hộ muốn đặt")
 
        if missing:
            missing_str = ", ".join(missing)
            state["sales_data"]["booking_pending"] = True
            state["sales_data"]["booking_missing_fields"] = missing
            return ToolResult(
                success=False,
                data={"missing": missing},
                summary=(
                    f"Cần thu thập thêm: {missing_str} trước khi đặt cọc. "
                    f"Đây là multi-turn — hỏi từng thông tin một."
                ),
            )

        if not state.get("booking_confirmation"):
            state["sales_data"]["booking_pending"] = True
            state["sales_data"]["booking_confirm_required"] = {
                "unit_code":       unit_code,
                "customer_name":   customer_name,
                "customer_phone":  customer_phone,
            }
            return ToolResult(
                success=False,
                data={"confirm_required": True},
                summary=(
                    f"Cần xác nhận booking: căn {unit_code}, "
                    f"khách {customer_name} — {customer_phone}. "
                    f"Đợi khách xác nhận trước khi gọi API."
                ),
            )

        try:
            result = await self._api.trigger_booking_intent(
                project=_project(state),
                unit_code=unit_code,
                customer_name=customer_name,
                customer_phone=customer_phone,
            )
            state["sales_data"]["booking"] = {
                "booking_id":   result.booking_id,
                "success":      result.success,
                "message":      result.message,
                "unit_code":    unit_code,
                "customer_name": customer_name,
            }
            summary = f"Đặt cọc {'thành công' if result.success else 'thất bại'}: {result.message}"
            log.info(
                "booking_completed",
                session=state.get("session_id"),
                unit_code=unit_code,
                success=result.success,
            )
            return ToolResult(success=result.success, data=result, summary=summary)
        except SalesAPIError as e:
            return ToolResult(success=False, data=None, summary=f"Lỗi đặt cọc: {e.message}", error=str(e))

# ── Tool 6: Appointment ───────────────────────────────────────────
 
class AppointmentTool(AgentTool):
    """
    [NEW] Đặt lịch hẹn xem nhà mẫu / sa bàn.
 
    Đây là tool phục vụ mục tiêu Giai đoạn 1 (AWARENESS):
      "Không bán nhà qua điện thoại, chỉ bán cuộc hẹn."
 
    Flow:
      1. Lấy khung giờ còn trống (get_available_slots)
      2. Slot filling: ngày/giờ mong muốn + số người đi cùng
      3. Xác nhận và book (book_appointment)
      4. Hệ thống tự gửi SMS/Zalo xác nhận cho khách
 
    Khác với BookingIntentTool (đặt cọc căn hộ thực) về nghiệp vụ.
    """
 
    def __init__(self, api: SalesAPIPort):
        self._api = api
 
    @property
    def name(self) -> str:
        return "book_appointment"
 
    @property
    def description(self) -> str:
        return (
            "Đặt lịch hẹn cho khách đến xem nhà mẫu hoặc sa bàn tại Sale Gallery. "
            "Dùng khi khách muốn 'đi xem', 'đặt lịch', 'book hẹn', 'cuối tuần đến được không'."
        )
 
    async def run(self, state: AgentState) -> ToolResult:
        kwargs = state.get("tool_kwargs", {}).get(self.name, {})
        customer_name  = state.get("customer_name")  or state.get("sales_data", {}).get("customer_name", "")
        customer_phone = state.get("customer_phone") or state.get("sales_data", {}).get("customer_phone", "")
        preferred_date = kwargs.get("preferred_date")   # "2026-05-10" hoặc None
        num_guests     = int(kwargs.get("num_guests", 1))
        note           = kwargs.get("note", "")
 
        # ── Slot filling: lấy danh sách khung giờ trống ──────────
        try:
            slots = await self._api.get_available_slots(
                project=_project(state),
                preferred_date=preferred_date,
            )
        except SalesAPIError as e:
            return ToolResult(success=False, data=None, summary=f"Lỗi lấy lịch: {e.message}", error=str(e))
 
        if not slots:
            return ToolResult(
                success=False,
                data=None,
                summary=(
                    "Hiện tại không còn khung giờ trống trong thời gian này. "
                    "Vui lòng thử ngày khác hoặc để lại SĐT để chuyên viên liên hệ."
                ),
            )
 
        # ── Thiếu thông tin khách → slot filling ─────────────────
        missing = []
        if not customer_name:
            missing.append("họ và tên")
        if not customer_phone:
            missing.append("số điện thoại")
 
        if missing:
            # Cung cấp khung giờ gợi ý để khách chọn
            slot_options = [
                f"• {s.date} lúc {s.time_start}–{s.time_end} tại {s.location} ({s.available_spots} chỗ trống)"
                for s in slots[:3]
            ]
            state["sales_data"]["available_slots"] = [
                {
                    "slot_id":         s.slot_id,
                    "date":            s.date,
                    "time_start":      s.time_start,
                    "time_end":        s.time_end,
                    "location":        s.location,
                    "available_spots": s.available_spots,
                }
                for s in slots[:5]
            ]
            missing_str = " và ".join(missing)
            summary = (
                f"Có {len(slots)} khung giờ trống. "
                f"Cần thu thập thêm {missing_str} để hoàn tất đặt lịch.\n"
                + "\n".join(slot_options)
            )
            return ToolResult(success=False, data={"missing": missing, "slots": slots[:5]}, summary=summary)
 
        # ── Đặt lịch với slot đầu tiên phù hợp ──────────────────
        chosen_slot = slots[0]
        slot_id = kwargs.get("slot_id") or chosen_slot.slot_id
 
        try:
            result = await self._api.book_appointment(
                project=_project(state),
                slot_id=slot_id,
                customer_name=customer_name,
                customer_phone=customer_phone,
                num_guests=num_guests,
                note=note,
            )
            state["appointment_booked"] = result.success
            state["sales_data"]["appointment"] = {
                "appointment_id":    result.appointment_id,
                "confirmation_code": result.confirmation_code,
                "success":           result.success,
                "message":           result.message,
                "scheduled_date":    result.scheduled_date,
                "scheduled_time":    result.scheduled_time,
                "location":          result.location,
            }
            summary = (
                f"Đặt lịch hẹn {'thành công' if result.success else 'thất bại'}. "
                f"Mã xác nhận: {result.confirmation_code}. "
                f"Thời gian: {result.scheduled_date} lúc {result.scheduled_time} "
                f"tại {result.location}."
            )
            log.info(
                "appointment_booked",
                session=state.get("session_id"),
                appointment_id=result.appointment_id,
                success=result.success,
            )
            return ToolResult(success=result.success, data=result, summary=summary)
        except SalesAPIError as e:
            return ToolResult(success=False, data=None, summary=f"Lỗi đặt lịch: {e.message}", error=str(e))
 
# ── Tool 7: Project List ──────────────────────────────────────────
 
class ProjectListTool(AgentTool):
    """Lấy danh sách các dự án bất động sản hiện có."""
 
    def __init__(self, api: SalesAPIPort):
        self._api = api
 
    @property
    def name(self) -> str:
        return "list_projects"
 
    @property
    def description(self) -> str:
        return "Lấy danh sách tất cả các dự án bất động sản hiện có."
 
    async def run(self, state: AgentState) -> ToolResult:
        try:
            projects = await self._api.list_all_projects()
            if not projects:
                return ToolResult(success=False, data=[], summary="Hiện tại chưa có dự án nào trong hệ thống.")
            
            # [FIX] Đảm bảo toàn bộ project_name là string để tránh lỗi join()
            projects = [str(p) for p in projects if p]
            
            state["sales_data"]["project_list"] = projects
            summary = f"Hiện tại có {len(projects)} dự án: " + ", ".join(projects)
            return ToolResult(success=True, data=projects, summary=summary)
        except SalesAPIError as e:
            return ToolResult(success=False, data=None, summary=f"Lỗi lấy danh sách dự án: {e.message}", error=str(e))
 
# ── Helpers ────────────────────────────────────────────────────────

# Helper regex functions have been removed. We now rely on LLM Native Tool Calling
# to extract parameters into state["tool_kwargs"].
