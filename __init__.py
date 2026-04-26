"""ComfyUI-Omnivoice-Triton: OmniVoice TTS via omnivoice-triton."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _add_vendor_dir_if_present() -> None:
    vendor_dir = Path(__file__).resolve().parent / "vendor"
    if vendor_dir.is_dir():
        vendor = str(vendor_dir)
        if vendor in sys.path:
            sys.path.remove(vendor)
        sys.path.insert(0, vendor)
        logger.info("[ComfyUI-Omnivoice-Triton] added vendor dir %s", vendor)


def _resolve_sideload_dir(raw_path: str) -> str:
    candidate = Path(raw_path).expanduser()
    direct_pkg = candidate / "omnivoice_triton"
    src_pkg = candidate / "src" / "omnivoice_triton"
    if direct_pkg.is_dir():
        return str(candidate)
    if src_pkg.is_dir():
        return str(candidate / "src")
    return ""


def _resolve_local_src_dir() -> str:
    src_dir = Path(__file__).resolve().parent / "src"
    if (src_dir / "omnivoice_triton").is_dir():
        return str(src_dir)
    return ""


def _sideload_source_if_needed() -> None:
    """Allow pre-PyPI development against a local omnivoice-triton checkout."""
    try:
        import omnivoice_triton  # noqa: F401

        return
    except ImportError:
        pass

    raw_path = os.environ.get("COMFYUI_OMNIVOICE_TRITON_SRC", "")
    src_dir = _resolve_sideload_dir(raw_path) if raw_path else _resolve_local_src_dir()
    if src_dir and src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        logger.info(
            "[ComfyUI-Omnivoice-Triton] sideloaded omnivoice_triton from %s",
            src_dir,
        )


_add_vendor_dir_if_present()
_sideload_source_if_needed()

try:
    from .nodes import (
        OmnivoiceTTSAuto,
        OmnivoiceTTSVoiceClone,
        OmnivoiceTTSVoiceDesign,
    )

    _import_ok = True
except ImportError as exc:
    logger.warning(
        "[ComfyUI-Omnivoice-Triton] failed to import nodes: %s. "
        "Install omnivoice-triton or set COMFYUI_OMNIVOICE_TRITON_SRC.",
        exc,
    )
    OmnivoiceTTSAuto = None  # type: ignore[assignment, misc]
    OmnivoiceTTSVoiceClone = None  # type: ignore[assignment, misc]
    OmnivoiceTTSVoiceDesign = None  # type: ignore[assignment, misc]
    _import_ok = False

if _import_ok:
    NODE_CLASS_MAPPINGS = {
        "OmnivoiceTTSAuto": OmnivoiceTTSAuto,
        "OmnivoiceTTSVoiceClone": OmnivoiceTTSVoiceClone,
        "OmnivoiceTTSVoiceDesign": OmnivoiceTTSVoiceDesign,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "OmnivoiceTTSAuto": "Omnivoice TTS (Auto)",
        "OmnivoiceTTSVoiceClone": "Omnivoice TTS (Voice Clone)",
        "OmnivoiceTTSVoiceDesign": "Omnivoice TTS (Voice Design)",
    }
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
