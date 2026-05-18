"""
app/agent/sales/usp_registry.py

DEPRECATED — Không còn sử dụng kể từ v2.

USP injection đã bị loại bỏ hoàn toàn. Sale có Graph riêng với
toàn quyền truy cập dữ liệu — không cần inject USP vào prompt.

File stub này giúp tránh ImportError. Không dùng trong code mới.
"""


class USPRegistry:
    """DEPRECATED stub."""
    def register_project(self, *a, **kw): pass
    def get_usps_for_stage(self, *a, **kw): return []
    def format_usps_for_context(self, *a, **kw): return ""
    def get_all_projects(self): return []


# Singleton stub — các import cũ không bị lỗi
usp_registry = USPRegistry()