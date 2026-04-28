"""
app/core/interfaces/sales_api_port.py

Contract với hệ thống backend bán hàng (external API).

Nguyên tắc bảo mật:
  - API key KHÔNG truyền qua URL — gửi qua header X-Internal-Key
  - Tất cả request qua HTTPS (verify=True, không tắt)
  - Response được validate schema trước khi dùng
  - Không log body chứa thông tin nhạy cảm

ISP: tách thành nhiều method nhỏ theo business capability,
thay vì một God method query().
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


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
class BookingResult:
    success: bool
    booking_id: str
    message: str
    unit_code: str


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
    async def trigger_booking_intent(
        self,
        project: str,
        unit_code: str,
        customer_name: str,
        customer_phone: str,
    ) -> BookingResult:
        """Đăng ký quan tâm / đặt giữ chỗ."""
        ...