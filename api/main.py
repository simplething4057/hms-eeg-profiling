"""HMS EEG 판독 보조 API — FastAPI 진입점"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.predict import ModelServer
from api.schemas import (
    CLASSES,
    ClassProb,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("모델 서버 초기화 중…")
    ModelServer.instance()   # 싱글턴 초기화 (DL + P1 로딩)
    logger.info(
        "모델 로드 완료 — DL=%s P1=%s",
        ModelServer.instance().dl_loaded,
        ModelServer.instance().p1_loaded,
    )
    yield


app = FastAPI(
    title="HMS EEG 판독 보조 API",
    version="1.0.0",
    description="E8 앙상블 (E4+E5 딥러닝 × 0.85 + P1 LightGBM × 0.15) + R1b 추론 플래그",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── 헬스체크 ──────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
def health() -> dict:
    srv = ModelServer.instance()
    return {
        "status": "ok",
        "dl_loaded": srv.dl_loaded,
        "p1_loaded": srv.p1_loaded,
    }


# ── 모델 정보 ──────────────────────────────────────────────────────

@app.get("/model/info", response_model=ModelInfoResponse, tags=["system"])
def model_info() -> ModelInfoResponse:
    srv = ModelServer.instance()
    return ModelInfoResponse(
        version="1.0.0",
        recipe=srv.recipe,
        dl_loaded=srv.dl_loaded,
        p1_loaded=srv.p1_loaded,
        n_folds=5,
        classes=CLASSES,
    )


# ── 예측 ──────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest) -> PredictResponse:
    srv = ModelServer.instance()
    if not srv.p1_loaded:
        raise HTTPException(status_code=503, detail="모델 로딩 중입니다. 잠시 후 재시도하세요.")

    import numpy as np  # noqa: PLC0415

    try:
        result = srv.predict(
            qeeg=np.array(req.qeeg_features, dtype=np.float32),
            spec_b64=req.spectrogram_b64,
        )
    except Exception as e:
        logger.exception("추론 오류")
        raise HTTPException(status_code=500, detail=str(e)) from e

    prob = result["prob"]
    pred_idx = int(max(range(len(prob)), key=lambda i: prob[i]))

    return PredictResponse(
        segment_id=req.segment_id,
        probabilities=ClassProb.from_array(prob),
        predicted_class=CLASSES[pred_idx],
        confidence=round(prob[pred_idx], 4),
        r1b_flag=result["flag"],
        r1b_flag_reason=result["reason"],
        model_used=result["model_used"],
        processing_ms=round(result["ms"], 1),
    )


# ── 배치 예측 ──────────────────────────────────────────────────────

@app.post("/predict/batch", response_model=list[PredictResponse], tags=["inference"])
def predict_batch(requests: list[PredictRequest]) -> list[PredictResponse]:
    if len(requests) > 64:
        raise HTTPException(status_code=400, detail="배치 최대 64개")
    return [predict(r) for r in requests]


# ── 파일 업로드 추론 ────────────────────────────────────────────────

@app.post("/predict/eeg", tags=["inference"])
async def predict_eeg_upload(file: UploadFile = File(...)) -> dict:
    """EEG parquet 파일 업로드 → E8 앙상블 추론.

    전처리 파이프라인: 더블 바나나 몽타주 → QC → 보간/필터 → qEEG 67차원 + 스펙트로그램 → 예측
    """
    if not (file.filename or "").endswith(".parquet"):
        raise HTTPException(status_code=400, detail="parquet 파일만 지원합니다 (.parquet)")

    data = await file.read()
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="파일이 너무 큽니다 (최대 50 MB)")

    import numpy as np  # noqa: PLC0415

    try:
        from api.infer_from_eeg import preprocess_eeg_file  # noqa: PLC0415
        qeeg_vec, spec_b64, file_meta = preprocess_eeg_file(data)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"EEG 전극 컬럼 없음: {e}") from e
    except Exception as e:
        logger.exception("EEG 전처리 오류")
        raise HTTPException(status_code=500, detail=str(e)) from e

    srv = ModelServer.instance()
    if not srv.p1_loaded:
        raise HTTPException(status_code=503, detail="모델 로딩 중입니다.")

    try:
        result = srv.predict(qeeg=qeeg_vec, spec_b64=spec_b64)
    except Exception as e:
        logger.exception("추론 오류")
        raise HTTPException(status_code=500, detail=str(e)) from e

    prob = result["prob"]
    pred_idx = int(np.argmax(prob))

    return {
        "segment_id": file.filename,
        "probabilities": dict(zip(CLASSES, [round(float(p), 4) for p in prob])),
        "predicted_class": CLASSES[pred_idx],
        "confidence": round(float(prob[pred_idx]), 4),
        "r1b_flag": result["flag"],
        "r1b_flag_reason": result["reason"],
        "model_used": result["model_used"],
        "processing_ms": round(result["ms"], 1),
        "file_meta": file_meta,
        "spectrogram_b64": spec_b64,
    }


# ── 대시보드 UI (StaticFiles — 마지막에 마운트) ───────────────────
if os.path.isdir("static"):
    app.mount("/ui", StaticFiles(directory="static", html=True), name="static")
