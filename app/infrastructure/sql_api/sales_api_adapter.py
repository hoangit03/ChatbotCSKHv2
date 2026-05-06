"""
app/infrastructure/sql_api/sales_api_adapter.py

HTTP adapter gọi backend Sales API v2.
Cập nhật:
  - Endpoints: /endpoint/product, /endpoint/project, /endpoint/consultation
  - Consultation payload: { name, phoneNumber, projectId, projectName, email, address }
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
    ConsultationResult,
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
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "User-Agent": "RagAgent/2.0",
            },
            timeout=httpx.Timeout(timeout),
            verify=True,
        )
        self._retries = max_retries

        _retry_cfg = retry(
            retry=retry_if_exception_type(httpx.TransportError),
            stop=stop_after_attempt(self._retries),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        )
        self._get  = _retry_cfg(self._get)
        self._post = _retry_cfg(self._post)

        log.info(
            "sales_api_init",
            base_url=base_url,
            api_key_preview=mask_value(api_key),
            max_retries=max_retries,
        )

    # ── Internal helpers ──────────────────────────────────────────

    async def _get(self, path: str, params: dict | None = None) -> dict | list:
        try:
            resp = await self._client.get(path, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            log.error("sales_api_http_error", path=path, status=e.response.status_code)
            raise SalesAPIError(
                f"Sales API error on {path}",
                upstream_status=e.response.status_code,
            ) from e
        except Exception as e:
            log.error("sales_api_network_error", path=path, error=str(e))
            raise SalesAPIError(f"Network error on {path}: {str(e)}") from e

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
        except Exception as e:
            log.error("sales_api_post_network_error", path=path, error=str(e))
            raise SalesAPIError(f"Network error on POST {path}: {str(e)}") from e

    # ── Port implementation ───────────────────────────────────────

    async def get_unit_availability(
        self,
        project: str,
        unit_code: Optional[str] = None,
    ) -> list[UnitAvailability]:
        params = {"project": project}
        if unit_code:
            params["unit_code"] = unit_code

        # Chuyển sang /endpoint/product
        data = await self._get("/endpoint/product", params=params)
        if isinstance(data, dict):
            data = data.get("data", data.get("units", []))

        return [_map_unit(u, project) for u in data]

    async def get_project_inventory(self, project: str) -> ProjectInventory:
        data = await self._get("/endpoint/project", params={"name": project})
        if isinstance(data, list) and data:
            data = data[0]
        elif isinstance(data, dict):
            # Nếu trả về dict đơn lẻ
            pass
        
        return ProjectInventory(
            project=data.get("name", project),
            total_units=int(data.get("scale", 0)),
            available=int(data.get("available", 0)),
            reserved=int(data.get("reserved", 0)),
            sold=int(data.get("sold", 0)),
        )

    async def get_payment_policies(self, project: str) -> list[PaymentPolicy]:
        # Giả định policy có thể lấy từ project hoặc endpoint riêng
        # Tạm thời giữ route cũ hoặc giả định Mock API hỗ trợ
        try:
            data = await self._get("/endpoint/project/payment-policies", params={"name": project})
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
        except:
            return []

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
            params["bedRoom"] = bedrooms # Map sang bedRoom theo product.json
        if min_price_vnd is not None:
            params["minPrice"] = min_price_vnd
        if max_price_vnd is not None:
            params["maxPrice"] = max_price_vnd
        if min_area_m2 is not None:
            params["minArea"] = min_area_m2
        if direction:
            params["direction"] = direction
        if floor:
            params["floor"] = floor

        # Chuyển sang /endpoint/product
        data = await self._get("/endpoint/product", params=params)
        if isinstance(data, dict):
            data = data.get("data", data.get("units", []))

        # [NEW] Chỉ hiển thị Kho hoặc Chưa mở bán trong kết quả tìm kiếm
        valid_statuses = ("kho", "chưa mở bán")
        filtered_data = [
            u for u in data 
            if str(u.get("virtualStatus", u.get("status", ""))).lower().strip() in valid_statuses
        ]

        return [_map_unit(u, project) for u in filtered_data]

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
        Đăng ký yêu cầu tư vấn → POST /endpoint/consultation.
        Payload yêu cầu: { name, phoneNumber, projectId, projectName, email, address }
        """
        payload: dict = {
            "name":         name,
            "phoneNumber":  phoneNumber,
            "projectId":    projectId,
            "projectName":  projectName,
        }
        if email:
            payload["email"] = email
        if address:
            payload["address"] = address

        log.info("consultation_request_v2", projectId=projectId, projectName=projectName)

        data = await self._post("/endpoint/consultation", payload)
        
        return ConsultationResult(
            success=data.get("success", True),
            consultation_id=data.get("id", data.get("consultation_id", "")),
            message=data.get("message", "Đã ghi nhận yêu cầu tư vấn."),
            name=name,
            phoneNumber=phoneNumber,
            projectId=projectId,
            projectName=projectName,
            email=email,
            address=address
        )

    async def list_all_projects(self) -> list[dict]:
        data = await self._get("/endpoint/project")
        if isinstance(data, dict):
            data = data.get("data", [])
            
        results = []
        for p in data:
            results.append({
                "id": p.get("id"),
                "name": str(p.get("name", "")),
                "code": p.get("code"),
                "type": p.get("type"),
                "status": p.get("status"),
                "investor": p.get("investor"),
                "scale": p.get("scale"),
                "area": p.get("area"),
                "startPrice": p.get("startPrice"),
                "endPrice": p.get("endPrice"),
                "address": p.get("address"),
                "district": p.get("district"),
                "province": p.get("province"),
                "classification": p.get("classification")
            })
        return results

    async def close(self) -> None:
        await self._client.aclose()


# ── Helpers ───────────────────────────────────────────────────────

def _map_unit(u: dict, project_name: str) -> UnitAvailability:
    """Map raw API dict (product.json schema) → UnitAvailability DTO."""
    raw_status = str(u.get("virtualStatus", u.get("status", ""))).lower().strip()
    
    if raw_status in ("kho", "chưa mở bán", "mở bán", "trống", "available"):
        status = "available"
    elif raw_status in ("booking", "chuyển cọc, chờ hồ sơ", "đặt cọc", "đăng kí", "thỏa thuận đảm bảo", "giữ chỗ", "reserved"):
        status = "reserved"
    elif raw_status in ("hợp đồng", "thanh lý", "chuyển nhượng", "khoá", "đã bàn giao", "bàn giao sổ hồng", "đã bán", "sold"):
        status = "sold"
    else:
        status = "unknown"

    return UnitAvailability(
        unit_code=str(u.get("code", "")),
        project=project_name,
        floor=int(u.get("floor", 0)) if str(u.get("floor", "")).isdigit() else 0,
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