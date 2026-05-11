"""
app/agent/tools/sales_tool.py

Các tool gọi Sales Backend API v2.
Endpoints: /endpoint/product, /endpoint/project, /endpoint/consultation.
"""
from __future__ import annotations

from datetime import date as _date
from typing import Optional

from app.agent.state.agent_state import AgentState, ScarcityLevel
from app.agent.tools.base_tool import AgentTool, ToolResult
from app.core.interfaces.sales_api_port import SalesAPIPort, ProjectStatusFilter
from app.shared.errors.exceptions import SalesAPIError
from app.shared.logging.logger import get_logger

log = get_logger(__name__)

# Ngưỡng scarcity
_SCARCITY_CRITICAL = 2
_SCARCITY_MEDIUM   = 5
_SCARCITY_LOW      = 10


def _project_name(state: AgentState) -> str:
    return state.get("project_name") or "unknown"


def _project_id(state: AgentState) -> str:
    return state.get("project_id") or ""


def _fmt_vnd(amount: float) -> str:
    """3_500_000_000 → '3,5 tỷ VNĐ'"""
    if amount >= 1_000_000_000:
        return f"{amount / 1_000_000_000:.1f} tỷ VNĐ"
    if amount >= 1_000_000:
        return f"{amount / 1_000_000:.0f} triệu VNĐ"
    return f"{amount:,.0f} VNĐ"


def _compute_scarcity(available_count: int) -> str:
    if available_count <= _SCARCITY_CRITICAL:
        return ScarcityLevel.CRITICAL
    if available_count <= _SCARCITY_MEDIUM:
        return ScarcityLevel.MEDIUM
    if available_count <= _SCARCITY_LOW:
        return ScarcityLevel.LOW
    return ScarcityLevel.NONE


def _unit_to_dict(u) -> dict:
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
        "status":          u.status,
    }


# ── Tool 1: Availability ──────────────────────────────────────────

class AvailabilityTool(AgentTool):
    """Kiểm tra một căn hộ cụ thể còn trống không."""

    def __init__(self, api: SalesAPIPort):
        self._api = api

    @property
    def name(self) -> str:
        return "check_availability"

    @property
    def description(self) -> str:
        return "Kiểm tra trạng thái của một căn hộ cụ thể theo mã căn (VD: T1-A14-03)."

    @property
    def tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "unit_code": {"type": "string", "description": "Mã căn hộ cần kiểm tra"}
                    },
                    "required": ["unit_code"],
                },
            },
        }

    async def run(self, state: AgentState, tool_kwargs: dict = None) -> ToolResult:
        kwargs = tool_kwargs or {}
        unit_code = kwargs.get("unit_code", "")
        project = _project_name(state)

        try:
            units = await self._api.get_unit_availability(project=project, unit_code=unit_code)
            if not units:
                return ToolResult(success=False, data=[], summary=f"Không tìm thấy căn {unit_code}.")

            unit = units[0]
            unit_dict = _unit_to_dict(unit)
            state["sales_data"]["unit_detail"] = unit_dict
            
            summary = f"Căn {unit_code}: {unit.status}. Tầng {unit.floor}, {unit.bedrooms}PN, giá {_fmt_vnd(unit.price_vnd)}."
            return ToolResult(success=True, data=unit_dict, summary=summary)
        except SalesAPIError as e:
            return ToolResult(success=False, data=None, summary=f"Lỗi: {e.message}")


# ── Tool 2: Inventory ─────────────────────────────────────────────

class InventoryTool(AgentTool):
    """Lấy tổng tồn kho dự án."""

    def __init__(self, api: SalesAPIPort):
        self._api = api

    @property
    def name(self) -> str:
        return "get_inventory"

    @property
    def description(self) -> str:
        return "Lấy thống kê tổng tồn kho dự án (số căn còn trống, đã bán...)."

    @property
    def tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}},
            },
        }

    async def run(self, state: AgentState, tool_kwargs: dict = None) -> ToolResult:
        project = _project_name(state)
        try:
            inv = await self._api.get_project_inventory(project=project)
            scarcity = _compute_scarcity(inv.available)
            state["scarcity_level"] = scarcity
            state["sales_data"]["inventory"] = {
                "total": inv.total_units,
                "available": inv.available,
                "scarcity": scarcity,
            }
            summary = f"Dự án {project} còn {inv.available}/{inv.total_units} căn trống."
            return ToolResult(success=True, data=state["sales_data"]["inventory"], summary=summary)
        except SalesAPIError as e:
            return ToolResult(success=False, data=None, summary=f"Lỗi: {e.message}")


# ── Tool 3: Unit Search ───────────────────────────────────────────

class UnitSearchTool(AgentTool):
    """Tìm căn hộ theo tiêu chí."""

    def __init__(self, api: SalesAPIPort):
        self._api = api

    @property
    def name(self) -> str:
        return "search_units"

    @property
    def description(self) -> str:
        return "Tìm căn hộ theo số phòng ngủ, giá, diện tích, tầng..."

    @property
    def tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bedrooms": {"type": "integer"},
                        "max_price": {"type": "number"},
                        "floor": {"type": "string"},
                    },
                },
            },
        }

    async def run(self, state: AgentState, tool_kwargs: dict = None) -> ToolResult:
        kwargs = tool_kwargs or {}
        project = _project_name(state)
        try:
            units = await self._api.search_units(
                project=project,
                bedrooms=kwargs.get("bedrooms"),
                max_price_vnd=kwargs.get("max_price"),
                floor=kwargs.get("floor"),
            )
            if not units:
                return ToolResult(success=False, data=[], summary="Không tìm thấy căn phù hợp.")

            results = [_unit_to_dict(u) for u in units[:5]]
            state["sales_data"]["search_results"] = results
            summary = f"Tìm thấy {len(units)} căn phù hợp. Các căn nổi bật: " + ", ".join([str(u["unit_code"]) for u in results])
            return ToolResult(success=True, data=results, summary=summary)
        except SalesAPIError as e:
            return ToolResult(success=False, data=None, summary=f"Lỗi: {e.message}")


# ── Tool 4: Consultation ──────────────────────────────────────────

class ConsultationTool(AgentTool):
    """
    Đăng ký tư vấn bán hàng.
    Yêu cầu: name, phoneNumber, projectId, projectName.
    """

    def __init__(self, api: SalesAPIPort):
        self._api = api

    @property
    def name(self) -> str:
        return "register_consultation"

    @property
    def description(self) -> str:
        return "Đăng ký yêu cầu tư vấn. Sale sẽ liên hệ lại với khách hàng."

    @property
    def tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_name": {"type": "string", "description": "Họ và tên của khách hàng (Bắt buộc trích xuất từ hội thoại)"},
                        "customer_phone": {"type": "string", "description": "Số điện thoại của khách hàng (Bắt buộc trích xuất từ hội thoại)"},
                        "email": {"type": "string", "description": "Email khách hàng (nếu có)"},
                        "address": {"type": "string", "description": "Địa chỉ khách hàng (nếu có)"},
                    },
                    "required": ["customer_name", "customer_phone"]
                },
            },
        }

    async def run(self, state: AgentState, tool_kwargs: dict = None) -> ToolResult:
        import re
        kwargs = tool_kwargs or {}
        
        # Lấy thông tin ưu tiên từ LLM trích xuất (kwargs), fallback về state
        name = kwargs.get("customer_name") or state.get("customer_name")
        phone = kwargs.get("customer_phone") or state.get("customer_phone")
        email = kwargs.get("email")
        address = kwargs.get("address")

        # ── Normalize & Validate phone ─────────────────────────────
        if phone:
            # Normalize: xóa khoảng trắng, gạch ngang, dấu chấm, ngoặc
            phone_clean = re.sub(r"[\s\-\.\(\)]+", "", str(phone))
            # Chuẩn hóa +84 → 0
            phone_clean = re.sub(r"^\+84", "0", phone_clean)
            # Validate: 10 chữ số, bắt đầu bằng 0[3-9]
            if re.fullmatch(r"0[3-9]\d{8}", phone_clean):
                phone = phone_clean
            else:
                log.warning("consultation_invalid_phone", raw=str(phone)[:20], session=state.get("session_id"))
                phone = None  # Coi như chưa có → trigger slot filling

        p_id = _project_id(state)
        p_name = _project_name(state)

        # Nếu LLM extract được name/phone mới, cập nhật ngược lại vào state
        if name and not state.get("customer_name"):
            state["customer_name"] = name
            state["sales_data"]["customer_name"] = name
        if phone and not state.get("customer_phone"):
            state["customer_phone"] = phone
            state["sales_data"]["customer_phone"] = phone

        missing = []
        if not name: missing.append("họ và tên")
        if not phone: missing.append("số điện thoại hợp lệ (10 số, bắt đầu 0[3-9])")
        if not p_id or not p_name or p_name == "unknown":
            missing.append("dự án quan tâm")

        if missing:
            state["sales_data"]["booking_missing_fields"] = missing
            summary = f"Cần cung cấp thêm: {', '.join(missing)} để đăng ký tư vấn."
            return ToolResult(success=False, data={"missing": missing}, summary=summary)
        else:
            # Xóa cờ missing nếu đã đủ thông tin
            if "booking_missing_fields" in state["sales_data"]:
                del state["sales_data"]["booking_missing_fields"]

        try:
            result = await self._api.register_consultation(
                name=name,
                phoneNumber=phone,
                projectId=p_id,
                projectName=p_name,
                email=email,
                address=address
            )
            state["sales_data"]["consultation"] = {"id": result.consultation_id, "success": True}
            return ToolResult(success=True, data=result, summary=result.message)
        except SalesAPIError as e:
            return ToolResult(success=False, data=None, summary=f"Lỗi: {e.message}")



# ── Tool 5: Project List ──────────────────────────────────────────

class ProjectListTool(AgentTool):
    """Lấy danh sách dự án, có thể lọc theo trạng thái."""

    def __init__(self, api: SalesAPIPort):
        self._api = api

    @property
    def name(self) -> str:
        return "list_projects"

    @property
    def description(self) -> str:
        return (
            "Lấy danh sách các dự án bất động sản. "
            "Dùng 'dang_mo_ban' khi khách hỏi dự án đang mở bán hoặc hiện đang bán. "
            "Dùng 'sap_mo_ban' khi khách hỏi dự án sắp ra mắt. "
            "Bỏ trống (không truyền) nếu khách muốn xem tất cả dự án không phân biệt trạng thái."
        )

    @property
    def tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status_filter": {
                            "type": "string",
                            "description": (
                                "Lọc danh sách theo trạng thái dự án. "
                                "Giá trị: 'dang_mo_ban' (đang mở bán), 'sap_mo_ban' (sắp mở bán). "
                                "Bỏ trống nếu muốn xem tất cả."
                            ),
                            "enum": ["dang_mo_ban", "sap_mo_ban"]
                        }
                    },
                },
            },
        }

    async def run(self, state: AgentState, tool_kwargs: dict = None) -> ToolResult:
        kwargs = tool_kwargs or {}
        raw_filter = kwargs.get("status_filter")

        # Convert string từ LLM → enum — an toàn, không crash nếu LLM gửi sai giá trị
        status_filter: ProjectStatusFilter | None = None
        if raw_filter:
            try:
                status_filter = ProjectStatusFilter(raw_filter)
            except ValueError:
                log.warning("project_list_invalid_filter", raw=raw_filter)
                # Tiếp tục không filter — tốt hơn là crash

        try:
            projects = await self._api.list_all_projects(status_filter=status_filter)
            if not projects:
                label = "đang mở bán" if status_filter == ProjectStatusFilter.DANG_MO_BAN else (
                    "sắp mở bán" if status_filter == ProjectStatusFilter.SAP_MO_BAN else "nào"
                )
                return ToolResult(success=False, data=[], summary=f"Hiện không có dự án {label}.")

            state["sales_data"]["project_list"] = projects
            label = " đang mở bán" if status_filter == ProjectStatusFilter.DANG_MO_BAN else (
                " sắp mở bán" if status_filter == ProjectStatusFilter.SAP_MO_BAN else ""
            )
            summary = f"Hiện có {len(projects)} dự án{label}: " + ", ".join([str(p["name"]) for p in projects])
            return ToolResult(success=True, data=projects, summary=summary)
        except SalesAPIError as e:
            return ToolResult(success=False, data=None, summary=f"Lỗi: {e.message}")

