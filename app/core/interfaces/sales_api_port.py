"""
app/core/interfaces/sales_api_port.py

Contract với hệ thống backend bán hàng (external API).

CHANGELOG v2:
  + AppointmentSlot DTO    — khung giờ xem nhà mẫu còn trống
  + AppointmentBookingResult DTO — kết quả đặt lịch hẹn
  + SalesAPIPort.get_available_slots() — lấy khung giờ trống
  + SalesAPIPort.book_appointment()    — đặt lịch hẹn xem nhà mẫu/sa bàn

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

@dataclass
class AppointmentSlot:
    """
    Một khung giờ xem nhà mẫu / sa bàn còn trống.
    Dùng cho flow Giai đoạn 1: chốt lịch hẹn thay vì bán nhà qua điện thoại.
    """
    slot_id: str
    date: str               # ISO format: "2026-05-10"
    time_start: str         # "09:00"
    time_end: str           # "10:00"
    location: str           # Tên Sale Gallery hoặc địa chỉ
    available_spots: int    # Số chỗ còn trống trong khung giờ này
    consultant_name: Optional[str] = None  # Chuyên viên phụ trách (nếu có)
 
 
@dataclass
class AppointmentBookingResult:
    """
    Kết quả sau khi đặt lịch hẹn xem nhà mẫu thành công.
    Trả về confirmation_code để khách hàng tra cứu.
    """
    success: bool
    appointment_id: str
    confirmation_code: str  # Mã xác nhận gửi qua SMS/Zalo
    message: str
    scheduled_date: Optional[str] = None
    scheduled_time: Optional[str] = None
    location: Optional[str] = None
    consultant_name: Optional[str] = None

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

    @abstractmethod
    async def get_available_slots(
        self,
        project: str,
        preferred_date: Optional[str] = None,   # ISO date "2026-05-10"
    ) -> list[AppointmentSlot]:
        """
        Lấy danh sách khung giờ xem nhà mẫu / sa bàn còn trống.
        Dùng cho Giai đoạn 1: mục tiêu là chốt lịch hẹn, không bán qua điện thoại.
        """
        ...
 
    @abstractmethod
    async def book_appointment(
        self,
        project: str,
        slot_id: str,
        customer_name: str,
        customer_phone: str,
        num_guests: int = 1,
        note: Optional[str] = None,
    ) -> AppointmentBookingResult:
        """
        Đặt lịch hẹn xem nhà mẫu / sa bàn.
 
        Khác với trigger_booking_intent (đặt cọc căn hộ thực).
        Flow: khách quan tâm → xem sa bàn → xem nhà mẫu → đặt cọc.
 
        Sau khi đặt thành công, hệ thống gửi SMS/Zalo xác nhận tự động.
        """
        ...

    @abstractmethod
    async def list_all_projects(self) -> list[str]:
        """Lấy danh sách tất cả tên các dự án hiện có."""
        ...