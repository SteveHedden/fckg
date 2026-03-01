"""Shared text-normalization utilities for feature extraction."""

from __future__ import annotations

import re


def _clean_space(text: str) -> str:
    return " ".join(str(text).split())


def normalize_person_name(name: str) -> str:
    """Normalize person names for entity matching — strip citations and noise."""
    cleaned = _clean_space(name)
    if not cleaned:
        return ""
    cleaned = re.sub(r"\[[^\]]+\]", "", cleaned)
    cleaned = cleaned.replace("†", "").replace("‡", "")
    cleaned = re.sub(r"\s*\((?:TIE)\)\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+won Academy Award.*$", "", cleaned, flags=re.IGNORECASE)
    return _clean_space(cleaned).strip(" ,;")
