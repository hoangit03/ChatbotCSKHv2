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
        log.info("sales_api_request_get_start", path=path, params=params)
        try:
            resp = await self._client.get(path, params=params)
            log.info("sales_api_request_get_done", path=path, status=resp.status_code)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            log.error("sales_api_http_error", path=path, status=e.response.status_code, body=e.response.text)
            raise SalesAPIError(
                f"Sales API error on {path}",
                upstream_status=e.response.status_code,
            ) from e
        except Exception as e:
            log.error("sales_api_network_error", path=path, error=str(e))
            raise SalesAPIError(f"Network error on {path}: {str(e)}") from e

    async def _post(self, path: str, payload: dict) -> dict:
        import json
        log.info("sales_api_request_post_start", path=path, payload=json.dumps(payload, ensure_ascii=False))
        try:
            resp = await self._client.post(
                path, 
                json=payload, 
                headers={"Content-Type": "application/json"}
            )
            log.info("sales_api_request_post_done", path=path, status=resp.status_code)
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

        return [UnitAvailability.from_api_dict(u, project) for u in data]

    async def get_project_inventory(self, project: str) -> ProjectInventory:
        # 1. Lấy thông tin tổng quan của dự án
        proj_data = await self._get("/endpoint/project", params={"name": project})
        total_units = 0
        project_name = project
        if isinstance(proj_data, list) and proj_data:
            proj_data = proj_data[0]
            project_name = proj_data.get("name", project)
            total_units = int(proj_data.get("scale", 0))

        # 2. Gọi API /endpoint/product để lấy tất cả căn hộ và tự tính toán (aggregation)
        products_data = await self._get("/endpoint/product", params={"project": project})
        units = products_data.get("data", products_data.get("units", [])) if isinstance(products_data, dict) else products_data
        
        available = 0
        reserved = 0
        sold = 0
        
        for u in units:
            unit_dto = UnitAvailability.from_api_dict(u, project_name)
            if unit_dto.status == "available":
                available += 1
            elif unit_dto.status == "reserved":
                reserved += 1
            elif unit_dto.status == "sold":
                sold += 1

        return ProjectInventory(
            project=project_name,
            total_units=total_units,
            available=available,
            reserved=reserved,
            sold=sold,
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

        return [UnitAvailability.from_api_dict(u, project) for u in filtered_data]

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
        
        if isinstance(data, bool):
            return ConsultationResult(
                success=data,
                consultation_id="",
                message="Đã ghi nhận yêu cầu tư vấn." if data else "Đăng ký thất bại.",
                name=name,
                phoneNumber=phoneNumber,
                projectId=projectId,
                projectName=projectName,
                email=email,
                address=address
            )
            
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
