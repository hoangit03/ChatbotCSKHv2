"""
app/agent/sales/usp_registry.py

[NEW FILE] USP Registry — Quản lý Unique Selling Points theo từng dự án và giai đoạn.

Kịch bản sales định nghĩa 11 USP chia theo 3 giai đoạn tâm lý.
File này implement cơ chế:
  1. Lưu trữ USP per-project (mỗi dự án có bộ USP riêng)
  2. Truy vấn USP phù hợp theo CustomerStage
  3. Tránh inject USP đã dùng trong cùng session
  4. Export USP content để inject vào Synthesizer context

Cách dùng:
    registry = USPRegistry()
    registry.register_project("prime_diamond", PRIME_DIAMOND_USPS)
    usps = registry.get_usps_for_stage("prime_diamond", CustomerStage.AWARENESS, used_ids=[])
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.agent.state.agent_state import CustomerStage, USPItem
from app.shared.logging.logger import get_logger

log = get_logger(__name__)


# ── USP Definitions — Prime Diamond (mẫu) ────────────────────────
# Đây là bộ USP mẫu dựa trên kịch bản đã cung cấp.
# Trong production, load từ DB/config file theo project_name.

PRIME_DIAMOND_USPS: list[USPItem] = [
    # ── Giai đoạn 1: AWARENESS — Tạo tò mò, không báo giá ────────
    USPItem(
        usp_id="USP_3",
        title="5 năm 0% lãi suất",
        description=(
            "Chủ đầu tư hỗ trợ chính sách 5 năm không lo lãi suất. "
            "Khách hàng mua bây giờ gần như không chịu áp lực tài chính "
            "trong 5 năm tới — đây là điểm cực kỳ hiếm trên thị trường hiện tại."
        ),
        stage=CustomerStage.AWARENESS,
        priority=1,
    ),
    USPItem(
        usp_id="USP_4",
        title="Cầu vượt nối thẳng ga Metro vào tầng 2",
        description=(
            "Dự án duy nhất tại khu vực có cầu vượt nối thẳng ga Metro vào tầng 2 tòa nhà. "
            "Từ nhà đi quận 1 làm việc hay đi chơi không cần bước xuống mặt đường — "
            "tiện nghi tuyệt đối cho người đi làm hàng ngày."
        ),
        stage=CustomerStage.AWARENESS,
        priority=2,
    ),
    USPItem(
        usp_id="USP_5",
        title="Vị trí kim cương — mặt tiền Võ Nguyên Giáp",
        description=(
            "Vị trí kim cương nằm ngay mặt tiền Võ Nguyên Giáp, "
            "trực diện công viên lớn và đối diện trung tâm hành chính. "
            "Đây là vị trí độc nhất vô nhị trong toàn khu vực."
        ),
        stage=CustomerStage.AWARENESS,
        priority=3,
    ),

    # ── Giai đoạn 2: CONSIDERATION — Đẩy cảm xúc, trải nghiệm ───
    USPItem(
        usp_id="USP_6",
        title="Đối diện CBD Trường Thọ — lõi trung tâm mới",
        description=(
            "Sở hữu nhà tại Prime Diamond là sở hữu bất động sản tại lõi trung tâm mới "
            "của thành phố — CBD Trường Thọ. Giá trị bất động sản khu vực này "
            "được dự báo tăng mạnh theo tiến độ quy hoạch."
        ),
        stage=CustomerStage.CONSIDERATION,
        priority=1,
    ),
    USPItem(
        usp_id="USP_7",
        title="Kiến trúc Singapore độc đáo",
        description=(
            "Dự án mang ngôn ngữ kiến trúc Singapore — phong cách hiện đại, "
            "tinh tế và đẳng cấp quốc tế. Mỗi góc nhìn của tòa nhà đều "
            "thể hiện đẳng cấp của chủ nhân."
        ),
        stage=CustomerStage.CONSIDERATION,
        priority=2,
    ),
    USPItem(
        usp_id="USP_8",
        title="Thiết kế 101 giác cắt — 100% căn không góc khuất",
        description=(
            "Thiết kế đặc biệt với 101 giác cắt của viên kim cương tuyệt đỉnh. "
            "Ánh sáng tự nhiên tràn ngập khắp căn hộ, 100% căn không có góc khuất, "
            "tối ưu phong thủy cho gia chủ."
        ),
        stage=CustomerStage.CONSIDERATION,
        priority=3,
    ),
    USPItem(
        usp_id="USP_9",
        title="2 tầng hầm 31.000m² — rộng nhất khu vực",
        description=(
            "Tầng hầm 31.000m² rộng nhất toàn khu vực — giải quyết triệt để "
            "bài toán đỗ xe của chung cư cao cấp. Đảm bảo chỗ đỗ thoải mái "
            "cho xe cá nhân của toàn bộ cư dân."
        ),
        stage=CustomerStage.CONSIDERATION,
        priority=4,
    ),
    USPItem(
        usp_id="USP_10",
        title="Lễ hội ánh sáng quốc tế",
        description=(
            "Về đêm, dự án là tâm điểm với Lễ hội ánh sáng của thương hiệu quốc tế "
            "trình chiếu ngay trên mặt ngoài tòa nhà. "
            "Căn hộ không chỉ là nơi ở — đó là một biểu tượng tự hào thực sự."
        ),
        stage=CustomerStage.CONSIDERATION,
        priority=5,
    ),

    # ── Giai đoạn 3: DECISION — Xóa rủi ro, chốt deal ───────────
    USPItem(
        usp_id="USP_1",
        title="Pháp lý đầy đủ — Giấy phép mở bán 2026",
        description=(
            "Pháp lý tuyệt đối an toàn: đã có Giấy phép mở bán 2026 "
            "và Giấy phép bán cho người nước ngoài. "
            "Đây là dự án hiếm hoi trên thị trường có đủ hai loại giấy phép này."
        ),
        stage=CustomerStage.DECISION,
        priority=1,
    ),
    USPItem(
        usp_id="USP_2",
        title="Ngân hàng phát triển nhà bảo lãnh bàn giao",
        description=(
            "Tiền của khách hàng được Ngân hàng Phát triển Nhà quản lý "
            "và cam kết tiến độ xây dựng. Độ an toàn 100% — "
            "đây là bảo chứng uy tín cao nhất mà một dự án có thể cung cấp."
        ),
        stage=CustomerStage.DECISION,
        priority=2,
    ),
    USPItem(
        usp_id="USP_11",
        title="Bảo chứng tăng giá — Kim cương trên tuyến Metro",
        description=(
            "Sở hữu cầu vượt nối ga Metro vào tầng 2, kết hợp chính sách 5 năm 0% lãi suất — "
            "đây là bảo chứng tăng giá chắc chắn. Trong 5 năm tới, "
            "tuyến Metro chạy ổn định, giá nhà tăng vọt mà không tốn một đồng trả lãi ngân hàng."
        ),
        stage=CustomerStage.DECISION,
        priority=3,
    ),
]


# ── USPRegistry ───────────────────────────────────────────────────

class USPRegistry:
    """
    [NEW] Quản lý toàn bộ USP theo project.

    Thiết kế:
      - Singleton pattern — dùng chung trong toàn app
      - Register USPs per-project lúc startup
      - Thread-safe read (dict lookup)
      - get_usps_for_stage() là hot path — O(n) nhưng n < 20 per project
    """

    def __init__(self) -> None:
        # project_name (lowercase) → list[USPItem]
        self._store: dict[str, list[USPItem]] = {}

    def register_project(self, project_name: str, usps: list[USPItem]) -> None:
        """
        Đăng ký bộ USP cho một dự án.
        Gọi lúc app startup hoặc khi onboard dự án mới.
        """
        key = project_name.lower().strip()
        self._store[key] = sorted(usps, key=lambda u: u.priority)
        log.info(
            "usp_registry_registered",
            project=key,
            count=len(usps),
            stages={s.value: sum(1 for u in usps if u.stage == s) for s in CustomerStage},
        )

    def get_usps_for_stage(
        self,
        project_name: str,
        stage: CustomerStage,
        used_ids: list[str],
        max_usps: int = 3,
    ) -> list[USPItem]:
        """
        Trả về danh sách USP phù hợp với stage hiện tại.

        Args:
            project_name : Tên dự án
            stage        : Giai đoạn tâm lý khách hàng hiện tại
            used_ids     : Danh sách USP IDs đã dùng trong session (để tránh lặp)
            max_usps     : Số USP tối đa trả về (default 3 — tránh overwhelm context)

        Returns:
            list[USPItem] đã lọc và sort theo priority
        """
        key = project_name.lower().strip()
        all_usps = self._store.get(key, [])

        if not all_usps:
            log.warning("usp_registry_project_not_found", project=key)
            return []

        # Lọc theo stage và loại bỏ USP đã dùng
        candidates = [
            u for u in all_usps
            if u.stage == stage and u.usp_id not in used_ids
        ]

        # Nếu không còn USP mới cho stage hiện tại, cho phép dùng lại (tránh context trống)
        if not candidates:
            candidates = [u for u in all_usps if u.stage == stage]
            log.info("usp_registry_all_used_recycling", stage=stage.value, project=key)

        return candidates[:max_usps]

    def format_usps_for_context(
        self,
        usps: list[USPItem],
    ) -> str:
        """
        Format danh sách USP thành chuỗi để inject vào Synthesizer context.
        """
        if not usps:
            return ""

        lines = ["=== ĐIỂM BÁN HÀNG NỔI BẬT (USP) ==="]
        for usp in usps:
            lines.append(f"\n✦ {usp.title}")
            lines.append(f"  {usp.description}")

        return "\n".join(lines)

    def get_all_projects(self) -> list[str]:
        """Trả về danh sách project đã đăng ký."""
        return list(self._store.keys())


# ── Singleton instance ─────────────────────────────────────────────
# Import và dùng trực tiếp trong toàn app:
#   from app.agent.sales.usp_registry import usp_registry
#   usp_registry.register_project(...)

usp_registry = USPRegistry()

# Đăng ký Prime Diamond mặc định
# Trong production: load từ DB/config file và gọi register_project() lúc startup
usp_registry.register_project("prime_diamond", PRIME_DIAMOND_USPS)