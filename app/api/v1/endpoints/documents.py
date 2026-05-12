from fastapi import APIRouter, UploadFile, File, Form, Depends, Request
from typing import Optional
import httpx
from app.shared.errors.exceptions import AppError

router = APIRouter(tags=["Documents"])

SHARED_ETL_URL = "http://103.186.101.200:8010"

@router.post("/documents/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    project_name: Optional[str] = Form(None),
    min_role_level: int = Form(1),
    force_overwrite: bool = Form(False)
):
    """
    Proxy upload file to shared_etl service
    """
    try:
        # Read file content
        content = await file.read()
        
        # Prepare multipart form data for httpx
        files = {
            "file": (file.filename, content, file.content_type)
        }
        
        data = {
            "min_role_level": str(min_role_level),
            "force_overwrite": str(force_overwrite).lower()
        }
        if project_name:
            data["project_name"] = project_name

        headers = {
            "X-Tenant-Id": "primer-diamond"
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{SHARED_ETL_URL}/etl/extract",
                data=data,
                files=files,
                headers=headers
            )
            
            if response.status_code != 200:
                raise AppError(
                    code="UPLOAD_ERROR", 
                    message=f"ETL Service Error: {response.text}", 
                    http_status=response.status_code
                )
                
            return response.json()
            
    except httpx.RequestError as e:
        raise AppError(
            code="ETL_CONNECTION_ERROR",
            message=f"Cannot connect to ETL service: {str(e)}",
            http_status=503
        )
    except AppError:
        raise
    except Exception as e:
        raise AppError(
            code="INTERNAL_ERROR",
            message=str(e),
            http_status=500
        )
