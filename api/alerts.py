"""알람 설정 — 클래스별 긴급도 및 임계값
포트폴리오 시연: DB 저장 + /alerts 엔드포인트
실제 임상: SMS / 앱 푸시 / 페이저 연동으로 교체
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# 클래스별 알람 설정 (임계값 초과 시 알람 발생)
ALERT_CONFIG: dict[str, dict] = {
    "seizure": {
        "level": "CRITICAL",
        "threshold": 0.65,
        "label": "발작 (Seizure)",
        "emoji": "🔴",
        "description": "즉시 전문의 확인 필요",
    },
    "lpd": {
        "level": "HIGH",
        "threshold": 0.60,
        "label": "편측 주기 방전 (LPD)",
        "emoji": "🟠",
        "description": "편측 주기성 패턴 감지 — 전문의 우선 판독",
    },
    "gpd": {
        "level": "HIGH",
        "threshold": 0.60,
        "label": "전반 주기 방전 (GPD)",
        "emoji": "🟠",
        "description": "전반 주기성 패턴 감지 — 전문의 우선 판독",
    },
    "lrda": {
        "level": "MEDIUM",
        "threshold": 0.65,
        "label": "편측 율동 방전 (LRDA)",
        "emoji": "🟡",
        "description": "편측 율동성 패턴 감지 — 추가 확인 권고",
    },
    "grda": {
        "level": "MEDIUM",
        "threshold": 0.65,
        "label": "전반 율동 방전 (GRDA)",
        "emoji": "🟡",
        "description": "전반 율동성 패턴 감지 — 추가 확인 권고",
    },
    "other": {
        "level": "NONE",
        "threshold": 1.1,
        "label": "기타 (Other)",
        "emoji": "⚪",
        "description": "",
    },
}

# 긴급도 순서 (UI 정렬용)
LEVEL_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "NONE": 3}


def check_alert(predicted_class: str, confidence: float) -> dict | None:
    """알람 발생 여부 판단. 임계값 미만이면 None 반환."""
    cfg = ALERT_CONFIG.get(predicted_class.lower(), ALERT_CONFIG["other"])
    if cfg["level"] == "NONE" or confidence < cfg["threshold"]:
        return None
    return {
        "alert_level": cfg["level"],
        "label": cfg["label"],
        "emoji": cfg["emoji"],
        "description": cfg["description"],
        "threshold": cfg["threshold"],
        "confidence": round(confidence, 4),
    }


def log_alert(alert_info: dict, filename: str, username: str) -> None:
    """알람 로그 출력 (실제 임상: SMS / 앱 푸시 / 페이저로 교체)"""
    logger.warning(
        "🚨 [ALERT %s] %s %s — 파일: %s / 업로드: %s / 확신도: %.1f%%",
        alert_info["alert_level"],
        alert_info["emoji"],
        alert_info["label"],
        filename,
        username,
        alert_info["confidence"] * 100,
    )
