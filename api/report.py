"""임상 EEG 판독 보고서 생성 — 지식 베이스 조회 기반"""
from __future__ import annotations

from datetime import datetime

from api.knowledge_base import KNOWLEDGE_BASE

CLASSES = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
CONF_HIGH = 0.70
CONF_MED = 0.50


def _confidence_label(p_max: float) -> str:
    if p_max >= CONF_HIGH:
        return f"높음 ({p_max*100:.1f}%)"
    if p_max >= CONF_MED:
        return f"보통 ({p_max*100:.1f}%)"
    return f"낮음 ({p_max*100:.1f}%) — 전문의 확인 필요"


def generate_report(
    segment_id: str,
    prob: list[float],
    predicted_class: str,
    flag: bool,
    flag_reason: str,
    model_used: str,
    recorded_at: datetime | None = None,
) -> dict:
    """분류 결과 → 임상 보고서 딕셔너리.

    Returns:
        dict with keys: header, findings, clinical, differential,
                        recommendations, urgency, references
    """
    kb = KNOWLEDGE_BASE.get(predicted_class, KNOWLEDGE_BASE["other"])
    p_max = max(prob)
    p_dict = {cls: round(p * 100, 1) for cls, p in zip(CLASSES, prob)}
    ts = (recorded_at or datetime.now()).strftime("%Y-%m-%d %H:%M")

    # 확률 순위 정렬
    ranked = sorted(p_dict.items(), key=lambda x: x[1], reverse=True)
    prob_lines = [
        f"{KNOWLEDGE_BASE[cls]['label_ko'].split('(')[0].strip()} {pct}%"
        for cls, pct in ranked
    ]

    header = {
        "title": "EEG 자동 판독 보고서",
        "subtitle": "HMS 뇌파 판독 보조 시스템 (E8 앙상블)",
        "segment_id": segment_id,
        "analyzed_at": ts,
        "model": "E8 앙상블 (ONNX)" if model_used == "e8_ensemble" else "P1 LightGBM",
        "disclaimer": (
            "본 보고서는 임상 의사결정 보조 목적으로 생성된 자동 분석 결과이며, "
            "최종 판독은 반드시 신경과 전문의가 확인해야 합니다."
        ),
    }

    findings = {
        "primary_finding": kb["label_ko"],
        "acns_code": kb["acns_code"],
        "confidence": _confidence_label(p_max),
        "review_flag": flag,
        "review_reason": flag_reason if flag else "없음",
        "probability_distribution": prob_lines,
        "description": kb["description"],
    }

    clinical = {
        "acns_criteria": kb["acns_criteria"],
        "clinical_significance": kb["clinical_significance"],
    }

    differential = kb["differential"]

    recommendations = kb["recommendations"]

    urgency = {
        "label": kb["urgency"],
        "level": kb["urgency_level"],
    }

    return {
        "header": header,
        "findings": findings,
        "clinical": clinical,
        "differential": differential,
        "recommendations": recommendations,
        "urgency": urgency,
        "reference": kb["reference"],
    }


def format_text_report(report: dict) -> str:
    """보고서 딕셔너리 → 텍스트 형식 (로그·디버그용)"""
    h = report["header"]
    f = report["findings"]
    c = report["clinical"]
    lines = [
        f"{'━'*50}",
        f"  {h['title']}",
        f"  {h['subtitle']}",
        f"{'━'*50}",
        f"구간 ID : {h['segment_id']}",
        f"분석 일시: {h['analyzed_at']}",
        f"분석 모델: {h['model']}",
        "",
        "[ 판독 소견 ]",
        f"주요 소견  : {f['primary_finding']} ({f['acns_code']})",
        f"판독 확신도: {f['confidence']}",
        f"검토 권고  : {'⚠ ' + f['review_reason'] if f['review_flag'] else '✓ 정상 판독'}",
        "",
        "확률 분포:",
    ] + [f"  {p}" for p in f["probability_distribution"]] + [
        "",
        "[ 뇌파 소견 ]",
        f["description"],
        "",
        "[ ACNS 2021 진단 기준 ]",
        c["acns_criteria"],
        "",
        "[ 임상적 의의 ]",
        c["clinical_significance"],
        "",
        "[ 감별 진단 ]",
    ] + [f"  • {d}" for d in report["differential"]] + [
        "",
        "[ 권고 사항 ]",
    ] + [f"  {i+1}. {r}" for i, r in enumerate(report["recommendations"])] + [
        "",
        f"긴급도: {report['urgency']['label']}",
        f"참고 기준: {report['reference']}",
        f"{'━'*50}",
        f"※ {h['disclaimer']}",
    ]
    return "\n".join(lines)
