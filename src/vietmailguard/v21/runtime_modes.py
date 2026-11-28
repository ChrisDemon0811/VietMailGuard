"""Explicit V2/V2.1 mode boundary for the existing app and mail service."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vietmailguard.inference import analyze_email as analyze_v2_email
from .runtime_inference import V21ModelUnavailable, analyze_v21_email, model_availability


V2_MODE = "v2"
V21_ML_MODE = "v21_ml"
V21_HYBRID_MODE = "v21_hybrid"
V21_MULTILINGUAL_MODE = "v21_multilingual"


def analyzer_for_mode(mode: str, *, hybrid_base: str = V21_ML_MODE) -> Callable[..., dict[str, Any]]:
    """Fail closed for an unavailable V2.1 artifact; never silently relabel."""
    if mode == V2_MODE:
        return analyze_v2_email
    if mode == V21_HYBRID_MODE:
        if hybrid_base not in {V21_ML_MODE, V21_MULTILINGUAL_MODE, "v23_ml", "v23_large_scale"}:
            raise ValueError("Unknown Hybrid base classifier")
        return analyzer_for_mode(hybrid_base)
    if mode in {"v23_ml", "v23_large_scale"}:
        from vietmailguard.v23.runtime_inference import (
            model_availability as v23_availability, analyze_v23_email, analyze_v23_large_email,
        )
        available, reason = v23_availability(mode)
        if not available:
            raise V21ModelUnavailable(reason)
        return analyze_v23_email if mode == "v23_ml" else analyze_v23_large_email
    if mode == V21_MULTILINGUAL_MODE:
        from .multilingual_runtime import analyze_multilingual_email, multilingual_availability
        available, reason = multilingual_availability()
        if not available:
            raise V21ModelUnavailable(reason)
        return analyze_multilingual_email
    if mode == V21_ML_MODE:
        available, reason = model_availability()
        if not available:
            raise V21ModelUnavailable(reason)
        return analyze_v21_email
    raise ValueError("Unknown analysis mode")
