"""Compressor interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class CompressionInput:
    text: str
    query: str
    must_have_terms: list[str]


@dataclass
class CompressionResult:
    text: str
    used: str               # provider name
    fell_back: bool = False
    error: str | None = None


class Compressor(Protocol):
    name: str

    def compress(self, item: CompressionInput) -> CompressionResult: ...


__all__ = ["Compressor", "CompressionInput", "CompressionResult"]
