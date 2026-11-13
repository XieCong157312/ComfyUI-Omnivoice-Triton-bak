#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OmniVoice-Triton 独立 API 服务
支持：普通TTS / 声音克隆 / 声音设计
"""

import io
import os
import time
import threading
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ==================== 配置 ====================
DEFAULT_MODEL_ID = "k2-fsa/OmniVoice"
DEFAULT_RUNNER_MODE = "hybrid"          # base / triton / faster / hybrid
DEFAULT_DTYPE = "fp16"
DEFAULT_DEVICE = "cuda"
SAMPLE_RATE = 24000

# ==================== Runner 缓存 ====================
_runner_cache = {}
_cache_lock = threading.Lock()


def get_runner(
    runner_mode: str = DEFAULT_RUNNER_MODE,
    model_id: str = DEFAULT_MODEL_ID,
    dtype: str = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
):
    from omnivoice_triton import create_runner

    key = (runner_mode, model_id, dtype, device)
    with _cache_lock:
        if key in _runner_cache:
            return _runner_cache[key]

        # 卸载旧 runner，防止显存翻倍
        for old_key, old_runner in list(_runner_cache.items()):
            try:
                old_runner.unload_model()
            except Exception:
                pass
            del _runner_cache[old_key]

        print(f"[OmniVoice] Loading runner: {key}")
        runner = create_runner(
            runner_mode,
            device=device,
            model_id=model_id,
            dtype=dtype,
        )
        runner.load_model()
        _runner_cache[key] = runner
        print(f"[OmniVoice] Runner ready.")
        return runner


# ==================== 工具函数 ====================
def audio_to_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """把 numpy 音频转成 wav bytes"""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()


def load_ref_audio(file: UploadFile) -> tuple[np.ndarray, int]:
    """读取上传的参考音频，并重采样到 24kHz"""
    data, sr = sf.read(file.file, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # 转单声道

    if sr != SAMPLE_RATE:
        try:
            import soxr
            data = soxr.resample(data, sr, SAMPLE_RATE).astype(np.float32)
            sr = SAMPLE_RATE
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="需要 soxr 库来重采样参考音频，请执行: pip install soxr"
            )
    return data, sr


# ==================== FastAPI ====================
app = FastAPI(
    title="OmniVoice-Triton API",
    description="Fast OmniVoice TTS API (Auto / Voice Clone / Voice Design)",
    version="1.0.0",
)


class TTSRequest(BaseModel):
    text: str = Field(..., description="要合成的文本")
    language: str = Field("auto", description="auto / ko / en / zh")
    runner_mode: str = Field(DEFAULT_RUNNER_MODE, description="base / triton / faster / hybrid")
    num_step: int = Field(32, ge=4, le=64)
    guidance_scale: float = Field(2.0, ge=0.0, le=10.0)
    class_temperature: float = Field(0.0, ge=0.0, le=2.0)
    model_id: str = Field(DEFAULT_MODEL_ID)
    dtype: str = Field(DEFAULT_DTYPE)
    device: str = Field(DEFAULT_DEVICE)


@app.get("/")
def root():
    return {
        "message": "OmniVoice-Triton API is running",
        "endpoints": {
            "POST /tts": "普通文本转语音",
            "POST /tts/clone": "声音克隆（需要上传参考音频）",
            "POST /tts/design": "声音设计（自然语言描述音色）",
            "GET /health": "健康检查",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "cached_runners": list(_runner_cache.keys())}


@app.post("/tts")
def tts(req: TTSRequest):
    """普通 TTS"""
    try:
        runner = get_runner(req.runner_mode, req.model_id, req.dtype, req.device)
        lang = None if req.language == "auto" else req.language

        t0 = time.time()
        result = runner.generate(
            text=req.text,
            language=lang,
            num_step=req.num_step,
            guidance_scale=req.guidance_scale,
            class_temperature=req.class_temperature,
        )
        cost = time.time() - t0

        # 兼容不同返回格式
        if isinstance(result, dict):
            audio = result["audio"]
            sr = result.get("sample_rate", SAMPLE_RATE)
        else:
            audio, sr = result

        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().cpu().float().numpy()

        wav_bytes = audio_to_bytes(audio, sr)
        headers = {
            "X-Generation-Time": f"{cost:.3f}",
            "X-Sample-Rate": str(sr),
        }
        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers=headers,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/clone")
async def tts_clone(
    text: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(""),
    language: str = Form("auto"),
    runner_mode: str = Form(DEFAULT_RUNNER_MODE),
    num_step: int = Form(32),
    guidance_scale: float = Form(2.0),
    class_temperature: float = Form(0.0),
    model_id: str = Form(DEFAULT_MODEL_ID),
    dtype: str = Form(DEFAULT_DTYPE),
    device: str = Form(DEFAULT_DEVICE),
):
    """声音克隆"""
    try:
        runner = get_runner(runner_mode, model_id, dtype, device)
        if not hasattr(runner, "generate_voice_clone"):
            raise HTTPException(status_code=400, detail="当前 runner 不支持 voice clone")

        ref_data, ref_sr = load_ref_audio(ref_audio)
        lang = None if language == "auto" else language

        t0 = time.time()
        result = runner.generate_voice_clone(
            text=text,
            ref_audio=(ref_data, ref_sr),
            ref_text=ref_text,
            language=lang,
            num_step=num_step,
            guidance_scale=guidance_scale,
            class_temperature=class_temperature,
        )
        cost = time.time() - t0

        if isinstance(result, dict):
            audio = result["audio"]
            sr = result.get("sample_rate", SAMPLE_RATE)
        else:
            audio, sr = result

        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().cpu().float().numpy()

        wav_bytes = audio_to_bytes(audio, sr)
        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"X-Generation-Time": f"{cost:.3f}"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/design")
def tts_design(
    text: str = Form(...),
    instruct: str = Form(..., description="例如: female, young adult, high pitch"),
    language: str = Form("auto"),
    runner_mode: str = Form(DEFAULT_RUNNER_MODE),
    num_step: int = Form(32),
    guidance_scale: float = Form(2.0),
    class_temperature: float = Form(0.0),
    model_id: str = Form(DEFAULT_MODEL_ID),
    dtype: str = Form(DEFAULT_DTYPE),
    device: str = Form(DEFAULT_DEVICE),
):
    """声音设计"""
    try:
        runner = get_runner(runner_mode, model_id, dtype, device)
        if not hasattr(runner, "generate_voice_design"):
            raise HTTPException(status_code=400, detail="当前 runner 不支持 voice design")

        lang = None if language == "auto" else language

        t0 = time.time()
        result = runner.generate_voice_design(
            text=text,
            instruct=instruct,
            language=lang,
            num_step=num_step,
            guidance_scale=guidance_scale,
            class_temperature=class_temperature,
        )
        cost = time.time() - t0

        if isinstance(result, dict):
            audio = result["audio"]
            sr = result.get("sample_rate", SAMPLE_RATE)
        else:
            audio, sr = result

        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().cpu().float().numpy()

        wav_bytes = audio_to_bytes(audio, sr)
        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"X-Generation-Time": f"{cost:.3f}"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # 建议加上 --disable-cuda-malloc 类似的环境（如果遇到显存问题）
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=10086,
        reload=False,
        workers=1,          # 模型较大，建议单进程
    )
