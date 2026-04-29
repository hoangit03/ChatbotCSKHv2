"""
app/infrastructure/sql_api/sales_api_adapter.py

HTTP adapter gọi backend Sales API.
Bảo mật:
  - API key qua header X-Internal-Key (không qua URL)
  - TLS enforced (verify=True)
  - Timeout + exponential retry
  - Log không bao giờ log body chứa data nhạy cảm
  - Response validated trước khi trả về
"""
from __future__ import annotations

from typing import Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.interfaces.sales_api_port import (
    BookingResult,
    PaymentPolicy,
    ProjectInventory,
    SalesAPIPort,
    UnitAvailability,
)
from app.shared.errors.exceptions import SalesAPIError
from app.shared.logging.logger import get_logger
from app.shared.security.guards import mask_value

log = get_logger(__name__)


class SalesAPIAdapter(SalesAPIPort):

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int = 10,
        max_retries: int = 3,
    ):
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "X-Internal-Key": api_key,
                "Content-Type": "application/json",
                "User-Agent": "RagAgent/1.0",
            },
            timeout=httpx.Timeout(timeout),
            verify=True,   # KHÔNG tắt TLS
        )
        self._retries = max_retries
        log.info(
            "sales_api_init",
            base_url=base_url,
            api_key_preview=mask_value(api_key),
        )

    # ── Internal helpers ──────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
    )
    async def _get(self, path: str, params: dict | None = None) -> dict | list:
        try:
            resp = await self._client.get(path, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            # Log status, KHÔNG log body
            log.error("sales_api_http_error", path=path, status=e.response.status_code)
            raise SalesAPIError(
                f"Sales API error on {path}",
                upstream_status=e.response.status_code,
            ) from e

    async def _post(self, path: str, payload: dict) -> dict:
        try:
            resp = await self._client.post(path, json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            log.error("sales_api_post_error", path=path, status=e.response.status_code)
            raise SalesAPIError(
                f"Sales API POST error on {path}",
                upstream_status=e.response.status_code,
            ) from e

    # ── Port implementation ───────────────────────────────────────

    async def get_unit_availability(
        self,
        project: str,
        unit_code: Optional[str] = None,
    ) -> list[UnitAvailability]:
        params = {"project": project}
        if unit_code:
            params["unit_code"] = unit_code

        data = await self._get("/api/v1/units/availability", params=params)
        if not isinstance(data, list):
            data = data.get("units", [])

        return [
            UnitAvailability(
                unit_code=u.get("code", u.get("unit_code", "")),
                project=u.get("projectId", project),
                floor=int(str(u.get("floor", "0")).strip()) if str(u.get("floor", "")).strip().isdigit() else 0,
                area_m2=float(u.get("builtUpArea", u.get("area_m2", 0)) or 0),
                bedrooms=int(u.get("bedRoom", u.get("bedrooms", 0)) or 0),
                status=u.get("status", "unknown"),
                price_vnd=float(u.get("priceVat", u.get("price_vnd", 0)) or 0),
                price_per_m2=float(u.get("unitPriceVat", u.get("price_per_m2", 0)) or 0),
                direction=u.get("direction"),
                carpet_area=float(u.get("carpetArea", 0) or 0),
                maintenance_fee=float(u.get("maintenanceFeeValue", 0) or 0),
                total_price=float(u.get("totalPrice", 0) or 0),
                sale_program=u.get("saleProgramName"),
                type=u.get("type")
            )
            for u in data
        ]

    async def get_project_inventory(self, project: str) -> ProjectInventory:
        data = await self._get("/api/v1/projects/inventory", params={"project": project})
        if isinstance(data, list):
            data = data[0] if data else {}
        return ProjectInventory(
            project=data.get("project", project),
            total_units=int(data.get("total_units", 0)),
            available=int(data.get("available", 0)),
            reserved=int(data.get("reserved", 0)),
            sold=int(data.get("sold", 0)),
        )

    async def get_payment_policies(self, project: str) -> list[PaymentPolicy]:
        data = await self._get("/api/v1/projects/payment-policies", params={"project": project})
        if not isinstance(data, list):
            data = data.get("policies", [])
        return [
            PaymentPolicy(
                project=project,
                name=p.get("name", ""),
                description=p.get("description", ""),
                installments=p.get("installments", []),
            )
            for p in data
        ]

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
        params: dict = {"project": project}
        if status:
            params["status"] = status
        if bedrooms is not None:
            params["bedrooms"] = bedrooms
        if min_price_vnd is not None:
            params["min_price"] = min_price_vnd
        if max_price_vnd is not None:
            params["max_price"] = max_price_vnd
        if min_area_m2 is not None:
            params["min_area"] = min_area_m2
        if max_area_m2 is not None:
            params["max_area"] = max_area_m2
        if direction:
            params["direction"] = direction
        if floor:
            params["floor"] = floor

        data = await self._get("/api/v1/units/search", params=params)
        if not isinstance(data, list):
            data = data.get("units", [])

        return [
            UnitAvailability(
                unit_code=u.get("code", u.get("unit_code", "")),
                project=u.get("projectId", project),
                floor=int(str(u.get("floor", "0")).strip()) if str(u.get("floor", "")).strip().isdigit() else 0,
                area_m2=float(u.get("builtUpArea", u.get("area_m2", 0)) or 0),
                bedrooms=int(u.get("bedRoom", u.get("bedrooms", 0)) or 0),
                status=u.get("status", "unknown"),
                price_vnd=float(u.get("priceVat", u.get("price_vnd", 0)) or 0),
                price_per_m2=float(u.get("unitPriceVat", u.get("price_per_m2", 0)) or 0),
                direction=u.get("direction"),
                carpet_area=float(u.get("carpetArea", 0) or 0),
                maintenance_fee=float(u.get("maintenanceFeeValue", 0) or 0),
                total_price=float(u.get("totalPrice", 0) or 0),
                sale_program=u.get("saleProgramName"),
                type=u.get("type")
            )
            for u in data
        ]

    async def trigger_booking_intent(
        self,
        project: str,
        unit_code: str,
        customer_name: str,
        customer_phone: str,
    ) -> BookingResult:
        payload = {
            "project": project,
            "unit_code": unit_code,
            "customer_name": customer_name,
            "customer_phone": customer_phone,
        }
        data = await self._post("/api/v1/bookings/intent", payload)
        return BookingResult(
            success=data.get("success", False),
            booking_id=data.get("booking_id", ""),
            message=data.get("message", ""),
            unit_code=unit_code,
        )

    async def close(self) -> None:
        await self._client.aclose()