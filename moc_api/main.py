"""
moc_api/main.py

Mock Backend API v2 cho Sales Chatbot.
Sử dụng dữ liệu từ project.json và product.json.
Các endpoint chính:
  - GET /endpoint/project : Lấy danh sách dự án (có search theo name)
  - GET /endpoint/product : Lấy danh sách sản phẩm (có search theo project, code, floor, bedRoom, giá)
  - POST /endpoint/consultation : Đăng ký tư vấn
"""
import json
import uuid
from pydantic import BaseModel
from typing import Optional

from fastapi import FastAPI, Query, HTTPException, Header, Depends

app = FastAPI(title="Sales Mock API v2")

API_KEY = "ak_34zs6l1r7z"

async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return x_api_key

# ── Load Data ─────────────────────────────────────────────────────

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

PROJECTS_DATA = load_json("moc_api/project.json")
PRODUCTS_DATA = load_json("moc_api/product.json")["data"]

# ── Models ────────────────────────────────────────────────────────

class ConsultationRequest(BaseModel):
    name: str
    phoneNumber: str
    projectId: str
    projectName: str
    email: Optional[str] = None
    address: Optional[str] = None

class ConsultationResponse(BaseModel):
    success: bool
    id: str
    message: str

# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/endpoint/project", dependencies=[Depends(verify_api_key)])
async def get_projects(name: Optional[str] = None):
    """
    Lấy danh sách dự án.
    Nếu có query 'name', filter theo tên (case-insensitive).
    """
    if name:
        filtered = [p for p in PROJECTS_DATA if name.lower() in str(p.get("name", "")).lower()]
        return filtered
    return PROJECTS_DATA

@app.get("/endpoint/product", dependencies=[Depends(verify_api_key)])
async def get_products(
    project: Optional[str] = Query(None, description="Tên dự án"),
    unit_code: Optional[str] = Query(None, description="Mã căn"),
    bedRoom: Optional[int] = None,
    floor: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    status: Optional[str] = None
):
    """
    Lấy danh sách sản phẩm.
    Hỗ trợ filter đa dạng.
    """
    results = PRODUCTS_DATA

    # 1. Filter theo project (nếu có tên dự án, tìm projectId trước)
    if project:
        proj_matches = [p for p in PROJECTS_DATA if project.lower() in str(p.get("name", "")).lower()]
        if proj_matches:
            proj_ids = [p["id"] for p in proj_matches]
            results = [r for r in results if r["projectId"] in proj_ids]
        else:
            return []

    # 2. Filter theo mã căn
    if unit_code:
        results = [r for r in results if unit_code.lower() in str(r.get("code", "")).lower()]

    # 3. Filter theo số phòng ngủ
    if bedRoom is not None:
        results = [r for r in results if r.get("bedRoom") == bedRoom]

    # 4. Filter theo tầng
    if floor:
        results = [r for r in results if str(r.get("floor")) == str(floor)]

    # 5. Filter theo giá
    if min_price is not None:
        results = [r for r in results if r.get("priceVat", 0) >= min_price]
    if max_price is not None:
        results = [r for r in results if r.get("priceVat", 0) <= max_price]

    # 6. Filter theo trạng thái
    if status:
        results = [r for r in results if status.lower() in str(r.get("status", "")).lower()]

    return {"data": results}

@app.post("/endpoint/consultation", response_model=ConsultationResponse, dependencies=[Depends(verify_api_key)])
async def post_consultation(req: ConsultationRequest):
    """
    Đăng ký tư vấn.
    """
    print(f"[*] Nhận yêu cầu tư vấn: {req.name} - {req.phoneNumber} cho {req.projectName} ({req.projectId})")
    
    # Giả lập thành công
    return ConsultationResponse(
        success=True,
        id=str(uuid.uuid4()),
        message=f"Chào {req.name}, yêu cầu tư vấn cho dự án {req.projectName} đã được ghi nhận. Chuyên viên sẽ gọi cho bạn qua số {req.phoneNumber} sớm nhất."
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)