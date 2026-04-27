"""
app/application/usecases/import_qa.py

Use Case: Import bộ Q&A chuẩn từ file Excel vào QAStore.

Excel format (tối thiểu 2 cột):
  | Question        | Answer       | Keywords (opt)      | DocGroup (opt) |
  |-----------------|--------------|---------------------|----------------|
  | Giá căn hộ 2PN? | Từ 3 tỷ VNĐ  | giá, 2pn, price     | price_list     |

SRP: chỉ lo đọc Excel + normalize + add vào QAStore.
OCP: format Excel thay đổi → chỉ sửa _parse_rows(), không sửa gì khác.
"""
from __future__ import annotations

import io
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from app.agent.tools.qa_tool import QAItem, QAStore
from app.shared.logging.logger import get_logger

log = get_logger(__name__)


# ── Request / Response DTOs ───────────────────────────────────────

@dataclass
class ImportQARequest:
    file_bytes: bytes
    file_name: str
    project_name: str
    sheet_name: Optional[str] = None     # None → sheet đầu tiên


@dataclass
class ImportQAResult:
    total_rows: int
    imported: int
    skipped: int
    errors: list[str] = field(default_factory=list)


# ── Use Case ──────────────────────────────────────────────────────

class ImportQAUseCase:
    """
    Đọc Excel → chuẩn hoá → thêm vào QAStore.
    QAStore được inject qua constructor (DIP).
    """

    def __init__(self, qa_store: QAStore) -> None:
        self._store = qa_store

    def execute(self, req: ImportQARequest) -> ImportQAResult:
        """
        Đồng bộ (không async) vì openpyxl không cần async I/O.
        File bytes đã được đọc trước khi gọi use case này.
        """
        log.info(
            "qa_import_start",
            project=req.project_name,
            file=req.file_name,
            sheet=req.sheet_name,
        )

        try:
            import openpyxl
        except ImportError:
            raise ImportError("pip install openpyxl")

        wb = openpyxl.load_workbook(io.BytesIO(req.file_bytes), data_only=True)

        # Chọn sheet
        if req.sheet_name and req.sheet_name in wb.sheetnames:
            ws = wb[req.sheet_name]
        else:
            ws = wb.active

        rows = list(ws.iter_rows(values_only=True))
        if len(rows) < 2:
            return ImportQAResult(total_rows=0, imported=0, skipped=0,
                                  errors=["File Excel không có dữ liệu (cần ít nhất 1 dòng header + 1 dòng data)"])

        header = [str(c or "").strip().lower() for c in rows[0]]
        col = _map_columns(header)

        if "question" not in col or "answer" not in col:
            return ImportQAResult(
                total_rows=0, imported=0, skipped=0,
                errors=["Excel thiếu cột 'Question' hoặc 'Answer'. "
                        "Kiểm tra tên cột ở dòng đầu tiên."],
            )

        imported = 0
        skipped = 0
        errors: list[str] = []
        data_rows = rows[1:]

        for row_idx, row in enumerate(data_rows, start=2):
            try:
                question = _cell(row, col["question"])
                answer   = _cell(row, col["answer"])

                if not question or not answer:
                    skipped += 1
                    continue

                # Bỏ qua dòng tiêu đề nhóm (vd: "1. Nhóm câu hỏi về pháp lý...")
                # Dấu hiệu: không có dấu ? và bắt đầu bằng số + dấu chấm
                if re.match(r"^\d+[\.\)]\s+", question) and "?" not in question:
                    skipped += 1
                    continue

                # Bỏ qua dòng trùng câu hỏi cùng project
                if self._store.search(question, project=req.project_name, threshold=0.95):
                    skipped += 1
                    continue

                keywords: list[str] = []
                if "keywords" in col:
                    raw_kw = _cell(row, col["keywords"])
                    keywords = [k.strip() for k in raw_kw.split(",") if k.strip()]

                doc_group: Optional[str] = None
                if "docgroup" in col:
                    doc_group = _cell(row, col["docgroup"]) or None

                item = QAItem(
                    id=str(uuid.uuid4()),
                    project_name=req.project_name,
                    question=question,
                    answer=answer,
                    keywords=keywords,
                    doc_group=doc_group,
                )
                self._store.add(item)
                imported += 1

            except Exception as e:
                errors.append(f"Dòng {row_idx}: {e}")

        log.info(
            "qa_import_done",
            project=req.project_name,
            imported=imported,
            skipped=skipped,
            errors=len(errors),
        )

        return ImportQAResult(
            total_rows=len(data_rows),
            imported=imported,
            skipped=skipped,
            errors=errors,
        )


# ── Helpers ───────────────────────────────────────────────────────

def _cell(row: tuple, idx: int) -> str:
    try:
        return str(row[idx] or "").strip()
    except IndexError:
        return ""


# Alias linh hoạt cho tên cột — hỗ trợ cả tiếng Việt lẫn tiếng Anh
_COLUMN_ALIASES: dict[str, list[str]] = {
    "question": ["question", "câu hỏi", "cau hoi", "q", "hỏi"],
    "answer":   ["answer", "câu trả lời", "cau tra loi", "a", "trả lời", "tra loi"],
    "keywords": ["keywords", "từ khoá", "tu khoa", "tags", "từ khóa"],
    "docgroup": ["docgroup", "doc_group", "nhóm tài liệu", "group", "nhom"],
}


def _map_columns(header: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for idx, col_name in enumerate(header):
            normalized = col_name.lower().strip().replace(" ", "")
            if col_name in aliases or normalized in [a.replace(" ", "") for a in aliases]:
                result[canonical] = idx
                break
    return result