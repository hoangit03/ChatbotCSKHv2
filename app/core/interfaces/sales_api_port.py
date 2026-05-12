"""
app/core/interfaces/sales_api_port.py

Contract với hệ thống backend bán hàng (external API).

CHANGELOG v5:
  - Thêm ProjectStatus enum: nguồn sự thật duy nhất cho status project từ API.
  - Thêm ProjectStatusFilter enum: user-facing filter key (dùng trong tool schema).
  - Thêm PROJECT_STATUS_FILTER_MAP: mapping tập trung filter → list[ProjectStatus].
    Khi API thay đổi status, chỉ cần sửa tại đây.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Project Status (giá trị thực từ API backend) ──────────────────

class ProjectStatus(str, Enum):
    """
    Enum map 1-1 với giá trị trường `status` trả về từ /endpoint/project.
    Khi API thêm/đổi status, chỉ cần thêm/sửa tại đây.
    """
    ABOUT_TO_SALE = "ABOUT_TO_SALE"   # Sắp mở bán
    ON_SALE       = "ON_SALE"         # Đang mở bán
    HANDING_OVER  = "HANDING_OVER"    # Đang bàn giao
    HANDED_OVER   = "HANDED_OVER"     # Đã bàn giao


class ProjectStatusFilter(str, Enum):
    """
    User-facing filter key — dùng trong tool schema (enum) để LLM truyền vào.
    Mỗi key map tới một hoặc nhiều ProjectStatus.
    """
    DANG_MO_BAN = "dang_mo_ban"   # Đang mở bán
    SAP_MO_BAN  = "sap_mo_ban"   # Sắp mở bán


# Mapping tập trung: filter key → danh sách API status được chấp nhận.
# Đây là nơi DUY NHẤT cần sửa khi business logic thay đổi.
PROJECT_STATUS_FILTER_MAP: dict[ProjectStatusFilter, list[ProjectStatus]] = {
    ProjectStatusFilter.DANG_MO_BAN: [ProjectStatus.ON_SALE],
    ProjectStatusFilter.SAP_MO_BAN:  [ProjectStatus.ABOUT_TO_SALE],
}



# ── Response DTOs ─────────────────────────────────────────────────

@dataclass
class UnitAvailability:
    unit_code: str
    project: str
    floor: int
    area_m2: float
    bedrooms: int
    status: str           # "available" | "reserved" | "sold"
    price_vnd: float
    price_per_m2: float
    direction: Optional[str] = None
    carpet_area: Optional[float] = None
    maintenance_fee: Optional[float] = None
    total_price: Optional[float] = None
    sale_program: Optional[str] = None
    type: Optional[str] = None

    @classmethod
    def from_api_dict(cls, u: dict, project_name: str) -> "UnitAvailability":
        """Map raw API dict (product.json schema) → UnitAvailability DTO."""
        import re as _re
        raw_status = str(u.get("virtualStatus", u.get("status", ""))).lower().strip()
        
        if raw_status in ("kho", "chưa mở bán", "mở bán", "trống", "available"):
            status = "available"
        elif raw_status in ("booking", "chuyển cọc, chờ hồ sơ", "đặt cọc", "đăng kí", "thỏa thuận đảm bảo", "giữ chỗ", "reserved"):
            status = "reserved"
        elif raw_status in ("hợp đồng", "thanh lý", "chuyển nhượng", "khoà", "đã bàn giao", "bàn giao sổ hồng", "đã bán", "sold"):
            status = "sold"
        else:
            status = "unknown"

        # FIX BUG-04: Parse floor an toàn, xử lý cả dạng "ầng 5", "B1", "10F", "3"
        raw_floor = str(u.get("floor", "") or "")
        floor_match = _re.search(r"\d+", raw_floor)
        floor_int = int(floor_match.group()) if floor_match else 0

        return cls(
            unit_code=str(u.get("code", "")),
            project=project_name,
            floor=floor_int,
            area_m2=float(u.get("builtUpArea", 0) or 0),
            bedrooms=int(u.get("bedRoom", 0) or 0),
            status=status,
            price_vnd=float(u.get("priceVat", 0) or 0),
            price_per_m2=float(u.get("unitPriceVat", 0) or 0),
            direction=u.get("direction"),
            carpet_area=float(u.get("carpetArea", 0) or 0),
            maintenance_fee=float(u.get("maintenanceFeeValue", 0) or 0),
            total_price=float(u.get("totalPrice", 0) or 0),
            sale_program=u.get("saleProgramName"),
            type=u.get("type"),
        )


@dataclass
class ProjectInventory:
    project: str
    total_units: int
    available: int
    reserved: int
    sold: int


@dataclass
class PaymentPolicy:
    project: str
    name: str
    description: str
    installments: list[dict] = field(default_factory=list)


@dataclass
class ConsultationResult:
    """
    Kết quả đăng ký tư vấn bán hàng.
    """
    success: bool
    consultation_id: str
    message: str
    name: str
    phoneNumber: str
    projectId: str
    projectName: str
    email: Optional[str] = None
    address: Optional[str] = None


# ── Port ──────────────────────────────────────────────────────────

class SalesAPIPort(ABC):
    """Giao tiếp với backend hệ thống bán hàng."""

    @abstractmethod
    async def get_unit_availability(
        self,
        project: str,
        unit_code: Optional[str] = None,
    ) -> list[UnitAvailability]:
        """Kiểm tra căn hộ còn trống không."""
        ...

    @abstractmethod
    async def get_project_inventory(
        self, project: str
    ) -> ProjectInventory:
        """Tổng số căn còn lại của dự án."""
        ...

    @abstractmethod
    async def get_payment_policies(
        self, project: str
    ) -> list[PaymentPolicy]:
        """Các phương thức thanh toán/trả góp."""
        ...

    @abstractmethod
    async def search_units(
        self,
        project: str,
        bedrooms: Optional[int] = None,
        min_price_vnd: Optional[float] = None,
        max_price_vnd: Optional[float] = None,
        min_area_m2: Optional[float] = None,
        max_area_m2: Optional[float] = None,
        direction: Optional[str] = None,
        floor: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[UnitAvailability]:
        """Tìm căn hộ theo tiêu chí."""
        ...

    @abstractmethod
    async def register_consultation(
        self,
        name: str,
        phoneNumber: str,
        projectId: str,
        projectName: str,
        email: Optional[str] = None,
        address: Optional[str] = None,
    ) -> ConsultationResult:
        """
        Đăng ký yêu cầu tư vấn bán hàng.
        Endpoint backend: POST /endpoint/consultation
        Payload: {"name", "phoneNumber", "projectId", "projectName", "email", "address"}
        """
        ...

    @abstractmethod
    async def list_all_projects(
        self,
        status_filter: Optional[ProjectStatusFilter] = None,
    ) -> list[dict]:
        """
        Lấy danh sách dự án.
        status_filter: ProjectStatusFilter enum — None = trả về tất cả.
        Mapping filter → API status được định nghĩa trong PROJECT_STATUS_FILTER_MAP.
        """
        ...