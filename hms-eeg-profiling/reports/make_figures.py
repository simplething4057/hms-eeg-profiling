"""결과 그림 초안 (README·분석 리포트용). 실행 : python reports/make_figures.py

- 그림 1 : experiments/results.csv (제출 실험의 고득표 CV·Private)
- 그림 2·3 : nb07 v3·nb06 출력값 (10/26 Kaggle 실행 결과를 옮겨 적음, 출처는 각 상수 주석)
그림 글자는 영문 (실행 환경 한글 글꼴 의존 제거)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # 참조 팔레트 1~3번 (검증된 순서)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb"})

# 그림 1 : 제출 실험 점수 추이
res = pd.read_csv(ROOT / "experiments" / "results.csv").set_index("exp")
sub = res.loc[["E1", "E2", "E4", "E8"], ["cv_kl_high", "private_kl"]].astype(float)
labels = ["E1\nbaseline", "E2\n+ high-vote\nfine-tune", "E4\n+ 10 min & 50 s\ninputs", "E8\n+ ensemble\n(E4, E5, P1)"]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
x = range(len(sub))
ax.plot(x, sub.private_kl, color=BLUE, lw=2, marker="o", ms=8, label="Private LB (hidden test)")
ax.plot(x, sub.cv_kl_high, color=ORANGE, lw=2, marker="o", ms=8, label="High-vote CV (OOF)")
for i, (cv, pv) in enumerate(zip(sub.cv_kl_high, sub.private_kl)):
    ax.annotate(f"{pv:.3f}", (i, pv), textcoords="offset points", xytext=(0, 9), ha="center", color=INK, fontsize=9)
    ax.annotate(f"{cv:.3f}", (i, cv), textcoords="offset points", xytext=(0, -16), ha="center", color=MUTED, fontsize=9)
ax.axhline(0.40, color=MUTED, lw=1, ls="--"); ax.text(-0.25, 0.403, "target 0.40", va="bottom", fontsize=8, color=MUTED)
ax.axhline(0.35, color=MUTED, lw=1, ls=":"); ax.text(-0.25, 0.353, "stretch 0.35", va="bottom", fontsize=8, color=MUTED)
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("KL divergence (lower is better)"); ax.set_ylim(0.25, 0.58); ax.set_xlim(-0.35, 3.35)
ax.set_title("Submission history: Private 0.532 -> 0.374 (-30%)", loc="left", fontsize=11, color=INK)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "fig1_score_history.png", dpi=150); plt.close(fig)

# 그림 2 : 클래스별 고득표 KL (nb07 v3 섹션 2, 10/26)
per_class = pd.DataFrame({
    "P1v2 (profile + LightGBM)": [0.3927, 0.6957, 0.4887, 0.7017, 0.4585, 0.3965],
    "E2 (10-min spectrogram)": [0.5044, 0.5103, 0.3829, 0.4921, 0.4335, 0.3094],
    "E4 (10-min + 50-s EEG)": [0.4138, 0.4147, 0.3262, 0.4459, 0.3885, 0.2577],
}, index=["Seizure\n(n=258)", "LPD\n(1162)", "GPD\n(747)", "LRDA\n(305)", "GRDA\n(425)", "Other\n(3042)"])
fig, ax = plt.subplots(figsize=(8.5, 4.2))
w = 0.26
for j, (name, col) in enumerate(zip(per_class.columns, [AQUA, ORANGE, BLUE])):
    ax.bar([i + (j - 1) * (w + 0.02) for i in range(6)], per_class[name], width=w, color=col, label=name,
           edgecolor="#fcfcfb", linewidth=1)
ax.set_xticks(range(6)); ax.set_xticklabels(per_class.index, fontsize=8.5)
ax.set_ylabel("mean KL, high-vote segments"); ax.grid(axis="x", visible=False)
ax.set_title("Per-class error: interpretable profile model is best on Seizure, weakest on lateralized patterns",
             loc="left", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")
ax.set_ylim(0, 0.85)
fig.tight_layout(); fig.savefig(OUT / "fig2_per_class_kl.png", dpi=150); plt.close(fig)

# 그림 3 : R1 진단 (nb06 3-2·3-3, 10/26) : 오답률 비 (플래그 구간 / 나머지)
methods = pd.Series({"R1: evidence only\n(pre-specified)": 1.52, "R1b: evidence OR\nlow confidence": 2.72,
                     "Low confidence only\n(control)": 2.96})
axes_ = pd.Series({"Lateralized vs\ngeneralized": 2.60, "Rhythmic (delta)": 2.05, "Temporal evolution\n(seizure)": 1.00})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0))
fig.subplots_adjust(left=0.2, right=0.97, wspace=0.75, top=0.80, bottom=0.17)
for ax, ser, col, ttl in [(a1, methods, BLUE, "Overall flag (all high-vote segments)"),
                          (a2, axes_, ORANGE, "Axis level (axis judgement wrong)")]:
    ax.barh(range(len(ser))[::-1], ser.values, color=col, height=0.5)
    ax.set_yticks(range(len(ser))[::-1]); ax.set_yticklabels(ser.index, fontsize=8.5)
    for i, v in zip(range(len(ser))[::-1], ser.values):
        ax.text(v + 0.05, i, f"{v:.2f}x", va="center", fontsize=9, color=INK)
    ax.axvline(2.0, color=MUTED, lw=1, ls="--"); ax.axvline(1.0, color=MUTED, lw=0.8)
    ax.set_xlim(0, 3.5); ax.set_xlabel("error-rate ratio (flagged / not flagged)"); ax.grid(axis="y", visible=False)
    ax.set_title(ttl, loc="left", fontsize=10, color=INK)
fig.suptitle("R1 reasoning check: evidence explains axis-level doubt, confidence drives error detection",
             x=0.02, y=0.97, ha="left", fontsize=10.5, color=INK)
fig.savefig(OUT / "fig3_r1_diagnostics.png", dpi=150); plt.close(fig)
print("saved :", sorted(p.name for p in OUT.iterdir()))
