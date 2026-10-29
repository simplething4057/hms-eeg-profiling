"""HMS EEG 판독 보조 API — FastAPI 진입점"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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
