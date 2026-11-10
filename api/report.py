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


def _background_activity(prob: list[float], all_windows: list[dict] | None) -> str:
    """확률 분포 및 창 스캔 결과에서 배경파 활동 기술 생성."""
    p_other = prob[5]  # other 클래스 인덱스
    if all_windows and len(all_windows) > 1:
        n_total = len(all_windows)
        n_normal = sum(1 for w in all_windows if w.get("predicted_class") == "other")
        pct = n_normal / n_total * 100
        if pct >= 70:
            return (
                f"연속 {n_total}개 구간 중 {n_normal}개({pct:.0f}%)에서 정상 범위 — "
                "비이상 구간의 배경파 비교적 보존됨"
            )
        elif pct >= 30:
            return (
                f"연속 {n_total}개 구간 중 {n_normal}개({pct:.0f}%)에서 정상 범위 — "
                "간헐적 이상 소견, 배경파 부분 저하"
            )
        else:
            return (
                f"연속 {n_total}개 구간 중 {n_normal}개({pct:.0f}%)에서 정상 범위 — "
                "지속적 이상 소견, 배경파 현저 저하"
            )
    # 단일 창
    if p_other >= 0.40:
        return f"현재 분석 구간 배경파 비교적 보존됨 (정상 범위 확률 {p_other*100:.1f}%)"
    elif p_other >= 0.20:
        return f"현재 분석 구간 배경파 부분적 저하 (정상 범위 확률 {p_other*100:.1f}%)"
    return f"현재 분석 구간 배경파 현저 저하 (정상 범위 확률 {p_other*100:.1f}%)"


def generate_report(
    segment_id: str,
    prob: list[float],
    predicted_class: str,
    flag: bool,
    flag_reason: str,
    model_used: str,
    recorded_at: datetime | None = None,
    patient_info: dict | None = None,
    recording_params: dict | None = None,
    all_windows: list[dict] | None = None,
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

    pi = patient_info or {}
    rp = recording_params or {}
    dur = rp.get("duration_sec")
    n_win = rp.get("n_windows", 1)
    dur_str = f"{dur}초" if dur else "알 수 없음"
    if n_win > 1:
        dur_str += f" ({n_win}개 구간 슬라이딩 스캔)"

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
        "patient_id": pi.get("patient_id", ""),
        "patient_dob": pi.get("patient_dob", ""),
        "referring_physician": pi.get("referring_physician", ""),
        "montage": rp.get("montage", "더블 바나나 (16채널 차동 참조)"),
        "recording_duration": dur_str,
    }

    findings = {
        "primary_finding": kb["label_ko"],
        "acns_code": kb["acns_code"],
        "confidence": _confidence_label(p_max),
        "review_flag": flag,
        "review_reason": flag_reason if flag else "없음",
        "probability_distribution": prob_lines,
        "description": kb["description"],
        "eeg_features": kb.get("eeg_features", ""),
        "iic_spectrum": kb.get("iic_spectrum", ""),
    }

    clinical = {
        "acns_criteria": kb["acns_criteria"],
        "clinical_significance": kb["clinical_significance"],
        "background_activity": _background_activity(prob, all_windows),
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
