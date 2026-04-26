"""ComfyUI node for OmniVoice TTS via the `omnivoice-triton` package."""

from __future__ import annotations

import inspect
import logging
import os
import sys as _sys
import threading
from typing import Any

# Match the Qwen3 Triton node's ComfyUI hardening: Python 3.11 installs can
# hit numba compilation failures when librosa lazily imports @guvectorize code.
if _sys.version_info < (3, 12):
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

# Disable transformers>=5.x parallel shard materialization. Its
# ThreadPoolExecutor races against ComfyUI's default cudaMallocAsync
# backend (PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync, set by
# ComfyUI/cuda_malloc.py) and surfaces as a fake `torch.OutOfMemoryError`
# on small allocations. This env var is honored at `from_pretrained()`
# call time (see transformers/core_model_loading.py), so setting it at
# module import is sufficient. `setdefault` so users who deliberately
# set it remain in control.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

logger = logging.getLogger(__name__)

try:
    from omnivoice_triton import ALL_RUNNER_NAMES, create_runner

    _IMPORT_ERROR: str | None = None
except ImportError as exc:
    _IMPORT_ERROR = str(exc)
    create_runner = None  # type: ignore[assignment]
    ALL_RUNNER_NAMES = [
        "base",
        "triton",
        "faster",
        "hybrid",
    ]

_DTYPE_CHOICES = ["fp16", "bf16", "fp32"]
_DEVICE_CHOICES = ["cuda", "cpu"]
_LANGUAGE_CHOICES = ["auto", "ko", "en", "zh"]
_DEFAULT_MODEL_ID = "k2-fsa/OmniVoice"
_OMNIVOICE_SAMPLE_RATE = 24000

# Module-level warm-runner cache, shared across workflow executions.
_runner_cache: dict[tuple[str, str, str, str], Any] = {}
_cache_lock = threading.Lock()


def _get_or_create_runner(
    runner_mode: str,
    model_id: str,
    dtype: str,
    device: str,
) -> Any:
    """Return a warm runner, creating & loading a new one if needed.

    Thread-safe. When the (mode, model_id, dtype, device) key changes,
    the previously cached runner is ``unload_model()``-ed first so we
    don't keep two copies of the model in VRAM.
    """
    if create_runner is None:
        raise ImportError(
            "omnivoice-triton is not installed. "
            "Run: pip install omnivoice-triton\n"
            "Or set COMFYUI_OMNIVOICE_TRITON_SRC=/path/to/omnivoice-triton[/src]\n"
            f"Original error: {_IMPORT_ERROR}"
        )

    key = (runner_mode, model_id, dtype, device)
    with _cache_lock:
        if key in _runner_cache:
            logger.info("[Omnivoice-Triton] cache hit: %s", key)
            return _runner_cache[key]

        for old_key, old_runner in list(_runner_cache.items()):
            logger.info("[Omnivoice-Triton] evicting cached runner: %s", old_key)
            try:
                old_runner.unload_model()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[Omnivoice-Triton] unload failed: %s", exc)
            del _runner_cache[old_key]

        logger.info("[Omnivoice-Triton] creating runner: %s", key)
        # Ask ComfyUI to release its VRAM reservation so the runner's
        # `load_model()` can allocate through HF/transformers without
        # hitting ComfyUI's allocator cap. Safe no-op outside ComfyUI.
        try:
            import comfy.model_management as _mm  # type: ignore[import-not-found]

            _mm.unload_all_models()
            _mm.soft_empty_cache()
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Omnivoice-Triton] unload_all_models failed: %s", exc)

        runner = create_runner(
            runner_mode,
            device=device,
            model_id=model_id,
            dtype=dtype,
        )
        runner.load_model()
        _runner_cache[key] = runner
        return runner


def _to_list(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_to_list(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float))


def _ensure_waveform_3d(audio: Any) -> list[list[list[float]]]:
    data = _to_list(audio)
    if _is_number(data):
        return [[[float(data)]]]
    if not isinstance(data, (list, tuple)) or not data:
        raise ValueError("Omnivoice-Triton: audio output is empty or invalid")

    first = data[0]
    if _is_number(first):
        return [[[float(item) for item in data]]]

    if isinstance(first, (list, tuple)) and first and _is_number(first[0]):
        return [[[float(item) for item in channel] for channel in data]]

    if (
        isinstance(first, (list, tuple))
        and first
        and isinstance(first[0], (list, tuple))
    ):
        return [
            [
                [float(sample) for sample in channel]
                for channel in batch
            ]
            for batch in data
        ]

    raise ValueError("Omnivoice-Triton: unsupported audio tensor shape")


def _comfy_audio(audio: Any, sample_rate: int) -> dict[str, Any]:
    import torch

    waveform = torch.tensor(_ensure_waveform_3d(audio), dtype=torch.float32)
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def _normalize_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if isinstance(result, tuple) and len(result) == 2:
        audio, sample_rate = result
        return {
            "audio": audio,
            "sample_rate": sample_rate,
            "time_s": 0.0,
            "peak_vram_gb": 0.0,
        }
    raise ValueError("Omnivoice-Triton: unsupported upstream result format")


def _call_with_supported_kwargs(method: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(method)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return method(**kwargs)
    supported = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters
    }
    return method(**supported)


def _mixdown_to_mono(audio: dict[str, Any]) -> list[float]:
    waveform = _to_list(audio["waveform"])
    if not isinstance(waveform, list) or not waveform:
        raise ValueError("Omnivoice-Triton: ref_audio waveform is empty")

    if isinstance(waveform[0], list) and waveform[0] and isinstance(waveform[0][0], list):
        channels = waveform[0]
    elif isinstance(waveform[0], list):
        channels = waveform
    else:
        return [float(sample) for sample in waveform]

    channel_lists = [[float(sample) for sample in channel] for channel in channels]
    if not channel_lists or not channel_lists[0]:
        raise ValueError("Omnivoice-Triton: ref_audio waveform is empty")

    length = min(len(channel) for channel in channel_lists)
    return [
        sum(channel[index] for channel in channel_lists) / len(channel_lists)
        for index in range(length)
    ]


def _clone_ref_audio_input(audio: dict[str, Any]) -> tuple[Any, int]:
    sample_rate = int(audio["sample_rate"])
    mono = _mixdown_to_mono(audio)

    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "OmnivoiceTTSVoiceClone requires numpy to prepare reference AUDIO. "
            "Run install.py to install audio dependencies."
        ) from exc

    ref_audio = np.asarray(mono, dtype=np.float32)
    if sample_rate == _OMNIVOICE_SAMPLE_RATE:
        return ref_audio, sample_rate

    try:
        import soxr
    except ImportError as exc:
        raise ImportError(
            "OmnivoiceTTSVoiceClone requires soxr to resample reference AUDIO "
            "to 24 kHz. Run install.py to install audio dependencies."
        ) from exc

    ref_audio = soxr.resample(
        ref_audio,
        sample_rate,
        _OMNIVOICE_SAMPLE_RATE,
    ).astype(np.float32, copy=False)
    return ref_audio, _OMNIVOICE_SAMPLE_RATE


def _run_auto_tts(
    runner: Any,
    *,
    text: str,
    language: str,
    num_step: int,
    guidance_scale: float,
    class_temperature: float,
) -> dict[str, Any]:
    lang: str | None = None if language == "auto" else language
    return _normalize_result(
        _call_with_supported_kwargs(
            runner.generate,
            text=text,
            language=lang,
            num_step=num_step,
            guidance_scale=guidance_scale,
            class_temperature=class_temperature,
        )
    )


def _run_voice_clone(
    runner: Any,
    *,
    text: str,
    language: str,
    ref_audio: dict[str, Any],
    ref_text: str,
    num_step: int,
    guidance_scale: float,
    class_temperature: float,
) -> dict[str, Any]:
    if not hasattr(runner, "generate_voice_clone"):
        raise AttributeError(
            "The selected omnivoice-triton runner does not expose "
            "generate_voice_clone(). Update omnivoice-triton or use Auto TTS."
        )

    clone_ref_audio = _clone_ref_audio_input(ref_audio)
    lang: str | None = None if language == "auto" else language
    return _normalize_result(
        _call_with_supported_kwargs(
            runner.generate_voice_clone,
            text=text,
            ref_audio=clone_ref_audio,
            ref_text=ref_text,
            language=lang,
            num_step=num_step,
            guidance_scale=guidance_scale,
            class_temperature=class_temperature,
        )
    )


def _run_voice_design(
    runner: Any,
    *,
    text: str,
    instruct: str,
    language: str,
    num_step: int,
    guidance_scale: float,
    class_temperature: float,
) -> dict[str, Any]:
    if not hasattr(runner, "generate_voice_design"):
        raise AttributeError(
            "The selected omnivoice-triton runner does not expose "
            "generate_voice_design(). Update omnivoice-triton or use Auto TTS."
        )

    lang: str | None = None if language == "auto" else language
    return _normalize_result(
        _call_with_supported_kwargs(
            runner.generate_voice_design,
            text=text,
            instruct=instruct,
            language=lang,
            num_step=num_step,
            guidance_scale=guidance_scale,
            class_temperature=class_temperature,
        )
    )


_OPTIONAL_GENERATION_PARAMS = {
    "num_step": (
        "INT",
        {"default": 32, "min": 4, "max": 64, "step": 1},
    ),
    "guidance_scale": (
        "FLOAT",
        {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1},
    ),
    "class_temperature": (
        "FLOAT",
        {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05},
    ),
    "language": (
        _LANGUAGE_CHOICES,
        {"default": "auto"},
    ),
    "model_id": (
        "STRING",
        {"default": _DEFAULT_MODEL_ID},
    ),
    "dtype": (
        _DTYPE_CHOICES,
        {"default": "fp16"},
    ),
    "device": (
        _DEVICE_CHOICES,
        {"default": "cuda"},
    ),
}


class OmnivoiceTTSAuto:
    """Standard OmniVoice text-to-speech with Triton-accelerated runner.

    The underlying ``omnivoice-triton`` runner is cached at module scope
    and reused across queue runs. Changing the mode, model id, dtype or
    device on the node triggers a fresh runner (the old one is unloaded
    first).
    """

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "안녕하세요, OmniVoice Triton TTS 테스트입니다.",
                    },
                ),
                "runner_mode": (
                    ALL_RUNNER_NAMES,
                    {"default": "hybrid"},
                ),
            },
            "optional": {
                **_OPTIONAL_GENERATION_PARAMS,
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Omnivoice-Triton"
    DESCRIPTION = (
        "Standard OmniVoice text-to-speech via omnivoice-triton. "
        "Use Voice Clone or Voice Design nodes for conditioned voices."
    )

    def generate(
        self,
        text: str,
        runner_mode: str,
        num_step: int = 32,
        guidance_scale: float = 2.0,
        class_temperature: float = 0.0,
        language: str = "auto",
        model_id: str = _DEFAULT_MODEL_ID,
        dtype: str = "fp16",
        device: str = "cuda",
    ) -> tuple[dict[str, Any]]:
        if not text or not text.strip():
            raise ValueError("OmnivoiceTTSAuto: `text` must be non-empty")

        result = _run_auto_tts(
            _get_or_create_runner(runner_mode, model_id, dtype, device),
            text=text,
            language=language,
            num_step=num_step,
            guidance_scale=guidance_scale,
            class_temperature=class_temperature,
        )

        logger.info(
            "[Omnivoice-Triton] TTS done: %.2fs, peak VRAM %.2f GB",
            result.get("time_s", 0.0),
            result.get("peak_vram_gb", 0.0),
        )

        return (_comfy_audio(result["audio"], result["sample_rate"]),)


class OmnivoiceTTSVoiceClone:
    """OmniVoice zero-shot voice cloning from ComfyUI reference AUDIO."""

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Hello, this is an OmniVoice cloned voice test.",
                    },
                ),
                "ref_audio": ("AUDIO",),
                "runner_mode": (ALL_RUNNER_NAMES, {"default": "hybrid"}),
            },
            "optional": {
                "ref_text": ("STRING", {"multiline": True, "default": ""}),
                **_OPTIONAL_GENERATION_PARAMS,
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Omnivoice-Triton"
    DESCRIPTION = (
        "OmniVoice voice cloning via omnivoice-triton. Provide a reference "
        "AUDIO clip and optional transcript."
    )

    def generate(
        self,
        text: str,
        ref_audio: dict[str, Any],
        runner_mode: str,
        ref_text: str = "",
        num_step: int = 32,
        guidance_scale: float = 2.0,
        class_temperature: float = 0.0,
        language: str = "auto",
        model_id: str = _DEFAULT_MODEL_ID,
        dtype: str = "fp16",
        device: str = "cuda",
    ) -> tuple[dict[str, Any]]:
        if not text or not text.strip():
            raise ValueError("OmnivoiceTTSVoiceClone: `text` must be non-empty")

        result = _run_voice_clone(
            _get_or_create_runner(runner_mode, model_id, dtype, device),
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            num_step=num_step,
            guidance_scale=guidance_scale,
            class_temperature=class_temperature,
        )

        logger.info(
            "[Omnivoice-Triton] task=voice_clone runner=%s done: %.2fs, peak VRAM %.2f GB",
            runner_mode,
            result.get("time_s", 0.0),
            result.get("peak_vram_gb", 0.0),
        )

        return (_comfy_audio(result["audio"], result["sample_rate"]),)


class OmnivoiceTTSVoiceDesign:
    """OmniVoice voice design from a natural language voice instruction."""

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Hello, this is an OmniVoice designed voice test.",
                    },
                ),
                "instruct": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "female, young adult, high pitch",
                    },
                ),
                "runner_mode": (ALL_RUNNER_NAMES, {"default": "hybrid"}),
            },
            "optional": {
                **_OPTIONAL_GENERATION_PARAMS,
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Omnivoice-Triton"
    DESCRIPTION = (
        "OmniVoice voice design via omnivoice-triton. Describe the target "
        "voice instead of providing reference audio."
    )

    def generate(
        self,
        text: str,
        instruct: str,
        runner_mode: str,
        num_step: int = 32,
        guidance_scale: float = 2.0,
        class_temperature: float = 0.0,
        language: str = "auto",
        model_id: str = _DEFAULT_MODEL_ID,
        dtype: str = "fp16",
        device: str = "cuda",
    ) -> tuple[dict[str, Any]]:
        if not text or not text.strip():
            raise ValueError("OmnivoiceTTSVoiceDesign: `text` must be non-empty")
        if not instruct or not instruct.strip():
            raise ValueError("OmnivoiceTTSVoiceDesign: `instruct` must be non-empty")

        result = _run_voice_design(
            _get_or_create_runner(runner_mode, model_id, dtype, device),
            text=text,
            instruct=instruct,
            language=language,
            num_step=num_step,
            guidance_scale=guidance_scale,
            class_temperature=class_temperature,
        )

        logger.info(
            "[Omnivoice-Triton] task=voice_design runner=%s done: %.2fs, peak VRAM %.2f GB",
            runner_mode,
            result.get("time_s", 0.0),
            result.get("peak_vram_gb", 0.0),
        )

        return (_comfy_audio(result["audio"], result["sample_rate"]),)
