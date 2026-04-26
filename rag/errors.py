"""Error types for the RAG MVP."""
from __future__ import annotations

from typing import Any


class IngestionError(Exception):
    """Raised when ingestion of a single file fails.

    Carries the stage where it broke so callers can surface a clear reason.
    Stages: "validate", "extract", "chunk", "embed", "store".
    """

    def __init__(
        self,
        reason: str,
        *,
        file_path: str,
        stage: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.file_path = file_path
        self.stage = stage
        self.__cause__ = cause

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "error",
            "reason": self.reason,
            "filePath": self.file_path,
            "stage": self.stage,
        }

    def __str__(self) -> str:
        return f"[{self.stage}] {self.reason} (file: {self.file_path})"
