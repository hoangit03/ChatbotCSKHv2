"""
app/api/v1/endpoints/project.py
"""
from fastapi import APIRouter, Request
from app.shared.errors.exceptions import AppError

router = APIRouter(tags=["Projects"])

@router.get("/projects")
async def get_projects(request: Request):
    """
    Lấy danh sách tên dự án hiện có trong hệ thống từ Vector DB.
    """
    vdb = request.app.state.vector_db
    try:
        projects = await vdb.list_unique_projects()
        return {"projects": projects}
    except Exception as e:
        raise AppError(code="VDB_ERROR", message=f"Không thể lấy danh sách dự án: {str(e)}", http_status=500)
