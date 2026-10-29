"""R1 판독 보조 리포트 문구 : '소견 + 판정 + 근거 + 신뢰도' (연구·학습 목적, 임상 진단 아님)

10/26 R1 결론 반영 (코드북 22차 113~114)
- 플래그(전문의 우선 확인)의 주 조건 = 저확신 (최고 확률 < p_low, 학습 fold 고득표 하위 20%)
- 근거 불일치는 오류 탐지가 아니라 '어느 판단을 재확인할지' 알려주는 보조 문구 (편측 2.60배, 율동 2.05배)
- 시간 변화(Seizure) 근거는 판별력이 확인되지 않아 참고 정보로만 표시
"""
import numpy as np

from src.reasoning.rules import AXES, CLAIMS, CLASSES, PERIODIC

NAME = {"seizure": "발작(Seizure)", "lpd": "편측 주기성 방전(LPD)", "gpd": "전반 주기성 방전(GPD)",
        "lrda": "편측 율동성 델타(LRDA)", "grda": "전반 율동성 델타(GRDA)", "other": "기타(Other)"}
REGION = ["좌측 외측(LL)", "우측 외측(RL)", "좌측 부정중(LP)", "우측 부정중(RP)"]
FEAT_KO = {"abs_asym_total": "좌우 파워 비대칭", "abs_asym_delta": "좌우 델타 비대칭",
           "delta_peak_sharp_max": "델타 피크 선명도", "rel_delta": "델타 상대 파워",
           "center_ctx_power_absmax": "판정 구간 파워 변화(앞뒤 2분 대비)",
           "center_ctx_centroid_absmax": "판정 구간 주파수 변화(앞뒤 2분 대비)"}
RECHECK = {"lateral": "편측·전반 판단 재확인", "rhythmic": "율동성(델타) 판단 재확인"}
INFO_ONLY = {"seizure"}                     # 플래그·재확인에 쓰지 않는 참고 축


def confidence(p_max, recheck, p_low=0.4):
    if p_max < p_low:
        return "낮음"
    if p_max >= 0.6 and not recheck:
        return "높음"
    return "중간"


def make_report(feat, prob, thr, status=None, p_low=0.4):
    """feat : 한 구간의 지표(dict 또는 Series), prob : 6클래스 확률, thr : 임계값,
    p_low : 저확신 기준 (nb06 R1b : 학습 fold 고득표 최고 확률 하위 20%, 약 0.47). status는 이전 호출 호환용(사용 안 함)."""
    prob = np.asarray(prob, float)
    order = np.argsort(prob)[::-1]
    c, c2 = CLASSES[order[0]], CLASSES[order[1]]
    p_max = float(prob[order[0]])

    side = "좌측" if feat["asym_total"] > 0 else "우측"
    finding = NAME[c]
    if c in ("lpd", "lrda"):
        finding = f"{side} 우세 {finding}, 최대 이상 부위 {REGION[int(feat['focal_region'])]}"
    elif c == "seizure":
        finding = f"{finding} 의심, 최대 이상 부위 {REGION[int(feat['focal_region'])]}"

    evidence, recheck = [], []
    for ax_key, direction in CLAIMS[c]:
        ax = AXES[ax_key]
        parts, hits = [], 0
        for f in ax["features"]:
            ok = feat[f] > thr[f] if direction == "high" else feat[f] <= thr[f]
            hits += int(ok)
            parts.append(f"{FEAT_KO[f]} {feat[f]:.3g} ({'충족' if ok else '미충족'}, 기준 {'>' if direction == 'high' else '≤'} {thr[f]:.3g})")
        label = ax["name"] + (" 낮음(전반 패턴)" if direction == "low" else "")
        tag = " [참고 : 판별력 미확인]" if ax_key in INFO_ONLY else ""
        evidence.append(f"{label}{tag} : " + ", ".join(parts))
        if hits == 0 and ax_key not in INFO_ONLY:
            recheck.append(RECHECK[ax_key])
    if c in PERIODIC:
        evidence.append("주기성 : 정량 근거 미확정 (H5 미지지, 딥러닝 판단만 제시)")
    if not evidence:
        evidence.append("해당 판정의 확정 근거 지표 없음 (배경·기타 패턴)")

    low = p_max < p_low
    conf = confidence(p_max, recheck, p_low)
    return {
        "소견": finding,
        "판정": f"{NAME[c]} {p_max:.0%} (2순위 {NAME[c2]} {prob[order[1]]:.0%})",
        "근거": evidence,
        "재확인": recheck,
        "신뢰도": f"{conf} (최고 확률 {p_max:.2f}{', 저확신 기준 ' + format(p_low, '.2f') + ' 미만' if low else ''}"
                 f"{', 근거 불일치 : ' + ' · '.join(recheck) if recheck else ''})",
        "플래그": "전문의 우선 확인 (저확신)" if low else "",
    }


def format_report(r):
    lines = [f"[소견] {r['소견']}", f"[판정] {r['판정']}"] + [f"[근거] {e}" for e in r["근거"]]
    if r.get("재확인"):
        lines.append(f"[재확인] {' · '.join(r['재확인'])}")
    lines.append(f"[신뢰도] {r['신뢰도']}")
    if r["플래그"]:
        lines.append(f"[플래그] {r['플래그']}")
    return "\n".join(lines)
