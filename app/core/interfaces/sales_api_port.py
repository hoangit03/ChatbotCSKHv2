"""
app/core/interfaces/sales_api_port.py

Contract với hệ thống backend bán hàng (external API).

CHANGELOG v4:
  - Cập nhật register_consultation để khớp với payload mới:
    { "name", "phoneNumber", "projectId", "projectName", "email", "address" }
  - Giữ lại các method cơ bản về product/project.
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
    async def list_all_projects(self) -> list[dict]:
        """Lấy danh sách tất cả các dự án (bao gồm id và name)."""
        ...