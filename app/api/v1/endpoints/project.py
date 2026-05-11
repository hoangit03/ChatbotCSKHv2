"""
app/api/v1/endpoints/project.py
"""
from fastapi import APIRouter, Request
from app.shared.errors.exceptions import AppError

router = APIRouter(tags=["Projects"])

@router.get("/projects")
async def get_projects(request: Request):
    """
    Lấy danh sách tên dự án hiện có trong hệ thống từ Sales API (hoặc Vector DB nếu fallback).
    """
    try:
        sales_api = getattr(request.app.state, "sales_api", None)
        if sales_api:
            try:
                data = await sales_api._get("/endpoint/project")
                projects = []
                if isinstance(data, dict) and "data" in data:
                    projects = [d.get("name") for d in data["data"] if d.get("name")]
                elif isinstance(data, list):
                    projects = [d.get("name") for d in data if d.get("name")]
                
                # Loại bỏ trùng lặp và None
                projects = list(set(projects))
                if projects:
                    return {"projects": sorted(projects)}
            except Exception as e:
                # Log lỗi và fallback xuống VDB
                import logging
                logging.getLogger(__name__).warning(f"Sales API projects error, fallback to VDB: {str(e)}")
        
        vdb = getattr(request.app.state, "vector_db", None)
        if vdb:
            projects = await vdb.list_unique_projects()
            return {"projects": projects}
            
        return {"projects": []}
    except Exception as e:
        raise AppError(code="PROJECTS_ERROR", message=f"Không thể lấy danh sách dự án: {str(e)}", http_status=500)
