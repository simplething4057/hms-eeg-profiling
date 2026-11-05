"""HMS EEG 판독 보조 API — FastAPI 진입점"""
from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.alerts import check_alert, log_alert
from api.auth import (
    DEMO_USERS,
    Token,
    TokenData,
    authenticate_user,
    create_access_token,
    get_current_user,
    require_permission,
)
from api.database import (
    ack_alert,
    get_alerts,
    get_history,
    init_db,
    save_alert,
    save_analysis,
    save_audit,
)
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
    ModelServer.instance()
    logger.info(
        "모델 로드 완료 — DL=%s P1=%s",
        ModelServer.instance().dl_loaded,
        ModelServer.instance().p1_loaded,
    )
    await init_db()
    logger.info("데이터베이스 초기화 완료")
    yield


app = FastAPI(
    title="HMS EEG 판독 보조 API",
    version="1.1.0",
    description=(
        "E8 앙상블 (E4+E5 딥러닝 × 0.85 + P1 LightGBM × 0.15) + R1b 추론 플래그\n\n"
        "**인증**: `/auth/login`에서 JWT 발급 후 Authorize 버튼에 토큰 입력\n\n"
        "**시연 계정**: eeg_tech / tech1234 · resident / res1234 · neurologist / neuro1234"
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── 인증 ──────────────────────────────────────────────────────────

@app.post("/auth/login", response_model=Token, tags=["auth"])
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    request: Request,
) -> Token:
    """JWT 토큰 발급 (Swagger Authorize 버튼 연동)"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="아이디 또는 비밀번호가 올바르지 않습니다",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(user["username"], user["role"])
    await save_audit(
        user["username"], user["role"], "LOGIN",
        ip=request.client.host if request.client else "",
    )
    return Token(
        access_token=token,
        role=user["role"],
        full_name=user["full_name"],
    )


@app.get("/auth/me", tags=["auth"])
async def me(user: Annotated[TokenData, Depends(get_current_user)]) -> dict:
    """현재 로그인 사용자 정보"""
    info = DEMO_USERS.get(user.username, {})
    return {
        "username": user.username,
        "role": user.role,
        "full_name": info.get("full_name", user.username),
    }


# ── 헬스체크 ──────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
def health() -> dict:
    srv = ModelServer.instance()
    return {"status": "ok", "dl_loaded": srv.dl_loaded, "p1_loaded": srv.p1_loaded}


# ── 모델 정보 ──────────────────────────────────────────────────────

@app.get("/model/info", response_model=ModelInfoResponse, tags=["system"])
def model_info() -> ModelInfoResponse:
    srv = ModelServer.instance()
    return ModelInfoResponse(
        version="1.1.0",
        recipe=srv.recipe,
        dl_loaded=srv.dl_loaded,
        p1_loaded=srv.p1_loaded,
        n_folds=5,
        classes=CLASSES,
    )


# ── 예측 (JSON 입력) ───────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(
    req: PredictRequest,
    request: Request,
    user: Annotated[TokenData, Depends(require_permission("upload"))],
) -> PredictResponse:
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
    predicted_class = CLASSES[pred_idx]
    confidence = round(prob[pred_idx], 4)

    await save_audit(
        user.username, user.role, "PREDICT_JSON",
        detail=f"seg={req.segment_id} class={predicted_class}",
        ip=request.client.host if request.client else "",
    )

    return PredictResponse(
        segment_id=req.segment_id,
        probabilities=ClassProb.from_array(prob),
        predicted_class=predicted_class,
        confidence=confidence,
        r1b_flag=result["flag"],
        r1b_flag_reason=result["reason"],
        model_used=result["model_used"],
        processing_ms=round(result["ms"], 1),
    )


@app.post("/predict/batch", response_model=list[PredictResponse], tags=["inference"])
async def predict_batch(
    requests_body: list[PredictRequest],
    request: Request,
    user: Annotated[TokenData, Depends(require_permission("upload"))],
) -> list[PredictResponse]:
    if len(requests_body) > 64:
        raise HTTPException(status_code=400, detail="배치 최대 64개")
    await save_audit(
        user.username, user.role, "PREDICT_BATCH",
        detail=f"n={len(requests_body)}",
        ip=request.client.host if request.client else "",
    )
    return [await predict(r, request, user) for r in requests_body]


# ── 파일 업로드 추론 ────────────────────────────────────────────────

@app.post("/predict/eeg", tags=["inference"])
async def predict_eeg_upload(
    request: Request,
    file: UploadFile = File(...),
    user: Annotated[TokenData, Depends(require_permission("upload"))] = None,
) -> dict:
    """EEG parquet 파일 업로드 → 슬라이딩 윈도우 E8 앙상블 추론.

    파이프라인: 더블 바나나 몽타주 → QC → 보간/필터 → 50초 창 슬라이딩 스캔 → 최선 창 E8 추론
    """
    if not (file.filename or "").endswith(".parquet"):
        raise HTTPException(status_code=400, detail="parquet 파일만 지원합니다 (.parquet)")

    data = await file.read()
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="파일이 너무 큽니다 (최대 50 MB)")

    import numpy as np  # noqa: PLC0415

    try:
        from api.infer_from_eeg import _make_dl_spec_b64, preprocess_eeg_windows  # noqa: PLC0415
        windows_data, file_meta = preprocess_eeg_windows(data)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"EEG 전극 컬럼 없음: {e}") from e
    except Exception as e:
        logger.exception("EEG 전처리 오류")
        raise HTTPException(status_code=500, detail=str(e)) from e

    srv = ModelServer.instance()
    if not srv.p1_loaded:
        raise HTTPException(status_code=503, detail="모델 로딩 중입니다.")

    # ── 1단계: 모든 창 빠른 스캔 (DL spec 없이) ─────────────────────
    window_results: list[dict] = []
    try:
        for w in windows_data:
            result = srv.predict(qeeg=w["qeeg"], spec_b64=None)
            prob = result["prob"]
            pred_idx = int(np.argmax(prob))
            window_results.append({
                "start_sec": w["start_sec"],
                "end_sec": w["end_sec"],
                "predicted_class": CLASSES[pred_idx],
                "probabilities": dict(zip(CLASSES, [round(float(p), 4) for p in prob])),
                "confidence": round(float(prob[pred_idx]), 4),
                "r1b_flag": result["flag"],
                "r1b_flag_reason": result["reason"],
                "model_used": result["model_used"],
                "processing_ms": round(result["ms"], 1),
            })
    except Exception as e:
        logger.exception("추론 오류")
        raise HTTPException(status_code=500, detail=str(e)) from e

    # ── 2단계: 가장 이상한 창 선택 (non-Other 최고 확신도) ───────────
    def _abnormality_score(wr: dict) -> float:
        if wr["predicted_class"] != "other":
            return wr["confidence"]
        return max(wr["probabilities"][c] for c in CLASSES if c != "other")

    best_idx = max(range(len(window_results)), key=lambda i: _abnormality_score(window_results[i]))
    best_win = windows_data[best_idx]

    # ── 3단계: 최선 창에 DL spec 추가 → E8 최종 추론 ─────────────────
    spec_b64: str | None = None
    try:
        if srv.dl_loaded:
            spec_b64 = _make_dl_spec_b64(best_win["_eeg_50s"])
    except Exception:  # noqa: BLE001
        pass

    try:
        final_result = srv.predict(qeeg=best_win["qeeg"], spec_b64=spec_b64)
        final_prob = final_result["prob"]
        final_pred_idx = int(np.argmax(final_prob))
        window_results[best_idx].update({
            "predicted_class": CLASSES[final_pred_idx],
            "probabilities": dict(zip(CLASSES, [round(float(p), 4) for p in final_prob])),
            "confidence": round(float(final_prob[final_pred_idx]), 4),
            "r1b_flag": final_result["flag"],
            "r1b_flag_reason": final_result["reason"],
            "model_used": final_result["model_used"],
            "processing_ms": round(final_result["ms"], 1),
        })
    except Exception as e:
        logger.exception("최선 창 E8 추론 오류")

    best_wr = window_results[best_idx]
    predicted_class = best_wr["predicted_class"]
    confidence = best_wr["confidence"]

    # ── 4단계: 분석 결과 저장 & 감사 로그 ───────────────────────────
    analysis_id = await save_analysis({
        "username": user.username,
        "filename": file.filename,
        "duration_sec": file_meta.get("duration_sec"),
        "n_windows": file_meta.get("n_windows", 1),
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": best_wr["probabilities"],
        "r1b_flag": best_wr["r1b_flag"],
        "model_used": best_wr["model_used"],
        "processing_ms": best_wr["processing_ms"],
        "scan_mode": file_meta["n_windows"] > 1,
    })

    await save_audit(
        user.username, user.role, "PREDICT_EEG",
        detail=f"file={file.filename} class={predicted_class} conf={confidence:.3f} id={analysis_id}",
        ip=request.client.host if request.client else "",
    )

    # ── 5단계: 알람 판정 & 저장 ──────────────────────────────────────
    alert_info = check_alert(predicted_class, confidence)
    if alert_info:
        log_alert(alert_info, file.filename or "", user.username)
        await save_alert(analysis_id, {
            "username": user.username,
            "filename": file.filename,
            "alert_level": alert_info["alert_level"],
            "predicted_class": predicted_class,
            "confidence": confidence,
        })

    return {
        # 기존 단일 결과 필드 (최선 창 기준, backward compat)
        "segment_id": file.filename,
        "predicted_class": predicted_class,
        "probabilities": best_wr["probabilities"],
        "confidence": confidence,
        "r1b_flag": best_wr["r1b_flag"],
        "r1b_flag_reason": best_wr["r1b_flag_reason"],
        "model_used": best_wr["model_used"],
        "processing_ms": best_wr["processing_ms"],
        "file_meta": file_meta,
        "spectrogram_b64": spec_b64,
        # 슬라이딩 윈도우 필드
        "windows": window_results,
        "best_window_idx": best_idx,
        "scan_mode": file_meta["n_windows"] > 1,
        # 신규: 분석 ID & 알람
        "analysis_id": analysis_id,
        "alert": alert_info,
    }


# ── 알람 조회 & 확인 ──────────────────────────────────────────────

@app.get("/alerts", tags=["alerts"])
async def list_alerts(
    unread_only: bool = False,
    limit: int = 100,
    user: Annotated[TokenData, Depends(require_permission("view_alerts"))] = None,
) -> dict:
    """미확인 / 전체 알람 목록 조회 (전문의 권한 필요)"""
    alerts = await get_alerts(unread_only=unread_only, limit=limit)
    return {"alerts": alerts, "total": len(alerts)}


@app.post("/alerts/{alert_id}/ack", tags=["alerts"])
async def acknowledge_alert(
    alert_id: int,
    request: Request,
    user: Annotated[TokenData, Depends(require_permission("ack_alert"))] = None,
) -> dict:
    """알람 확인 처리 (전문의 권한 필요)"""
    ok = await ack_alert(alert_id, user.username)
    if not ok:
        raise HTTPException(status_code=404, detail="알람을 찾을 수 없거나 이미 확인된 알람입니다")
    await save_audit(
        user.username, user.role, "ACK_ALERT",
        detail=f"alert_id={alert_id}",
        ip=request.client.host if request.client else "",
    )
    return {"ok": True, "alert_id": alert_id, "ack_by": user.username}


# ── 분석 이력 조회 ────────────────────────────────────────────────

@app.get("/history", tags=["history"])
async def analysis_history(
    limit: int = 50,
    user: Annotated[TokenData, Depends(require_permission("view_history"))] = None,
) -> dict:
    """분석 이력 조회 (레지던트 이상 권한 필요)"""
    records = await get_history(limit=limit)
    return {"records": records, "total": len(records)}


@app.get("/history/me", tags=["history"])
async def my_history(
    limit: int = 50,
    user: Annotated[TokenData, Depends(require_permission("view_own"))] = None,
) -> dict:
    """내 분석 이력 조회"""
    records = await get_history(username=user.username, limit=limit)
    return {"records": records, "total": len(records)}


# ── 루트 리다이렉트 ───────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


# ── 대시보드 UI (StaticFiles — 마지막에 마운트) ───────────────────
if os.path.isdir("static"):
    app.mount("/ui", StaticFiles(directory="static", html=True), name="static")
