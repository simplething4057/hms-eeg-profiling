# HMS EEG Profiling : 설명 가능한 뇌파 판독 보조 시스템

> 50초 EEG 구간을 **딥러닝 6개 클래스 확률 + 판독 관점 정량 지표(프로파일)** 로 함께 분석해
> **'소견 + 판정 + 근거 + 신뢰도'** 형태의 판독 보조 리포트를 출력한다.

## 문제 정의
- 중환자실 EEG 판독은 전문의 의존도가 높고 판독 지연이 발생한다.
- FDA 허가 발작 탐지 제품은 '진단 대체'가 아닌 '판독 보조'로 허가된다.
- 따라서 분류 점수뿐 아니라 **'왜 그 판정인지'를 위치·패턴 단위로 설명**하는 구조가 필요하다.

## 데이터
- [Kaggle HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)
- 원시 EEG(200Hz, 19전극 + EKG), 10분 스펙트로그램, 전문가 투표(6개 클래스)
- **데이터는 이 저장소에 포함하지 않는다.** 모든 실행은 Kaggle 노트북에서 `/kaggle/input` 경로로 진행한다.

## 시스템 구조 (3단계)
1. **신호 품질 판정** : 결측, 평탄 신호, 과도 진폭 검출
2. **뇌 기능 프로파일링** : 주파수 구성, 편측성, 국소성, 주기성, 율동성, 시간 변화, 배경 활동
3. **소견 통합 추론** : 딥러닝 확률 + 축별 확률(위치·패턴) + 프로파일 근거의 일관성 검증, 불일치 시 '전문의 우선 확인' 플래그

## 평가 지표
| 구분 | 지표 | 목표 |
|---|---|---|
| 분류 | KL Divergence (전체 CV / 고득표 CV / Private) | Private 0.35 이하 |
| 축 분해 | 편측 대 전반 AUC, 주기성 대 율동성 AUC | 각 0.85 이상 |
| 사업 지표 | Seizure 민감도 @ 특이도 0.90 | 0.80 이상 |
| 추론 로직 | 근거 일치율 / 불일치 플래그 샘플 오답 비율 | 80% 이상 / 2배 이상 |

## 실행 순서 (Kaggle 노트북)
| 순서 | 노트북 | 가속기 | 출력 |
|---|---|---|---|
| 1 | `nb01_data_check` | CPU | EDA 그래프, `data_check_summary.json` |
| 2 | `nb02_preprocess` | CPU | Dataset `hms-processed` |
| 3 | `nb03_profile_lgbm` | CPU | 프로파일 지표, P1 OOF |
| 4 | `nb04_train` | GPU | Dataset `hms-models` |
| 5 | `nb05_submit` | GPU, 인터넷 OFF | `submission.csv` (PRESET : 단일 실험 / E4E5 / E8 앙상블) |
| 6 | `nb06_reasoning_report` | CPU | R1 근거 일치율·플래그 실효성, 판독 보조 리포트, 사례 후보 |
| 7 | `nb07_blend` | CPU, 인터넷 ON | OOF만으로 앙상블·보정·결합 방식·클래스별 비교 (수 분) |
| 8 | `nb08_pack` | CPU, 인터넷 ON | 제출용 모델 묶음 (E4·E5 가중치 + P1 fold 모델 + e8_recipe.json) → nb05 입력 |

- 노트북은 Kaggle에서 실행하고, Kaggle의 GitHub 저장 기능으로 `kaggle_notebooks/`에 사본을 저장한다.
- `src/`는 이 저장소가 원본이며, Kaggle Dataset `hms-src`로 올려 노트북에서 import한다.

```python
# Kaggle 노트북에서 src 사용
import sys; sys.path.append("/kaggle/input/hms-src")
from src.data.targets import build_targets
from src.eval.metrics import kl_divergence
```

## 저장소 구조
```
hms-eeg-profiling/
├── README.md
├── requirements.txt          # 로컬 테스트용 (Kaggle 환경에는 기본 설치)
├── configs/base.yaml         # 경로, 시드, fold 수, 필터 대역, 고득표 기준
├── kaggle_notebooks/         # Kaggle 실행 노트북 사본 (nb01~nb06)
├── kaggle/dataset-metadata.json  # src/ → Kaggle Dataset 'hms-src' 업로드용
├── src/
│   ├── config.py             # 설정 로드
│   ├── data/
│   │   ├── load.py           # Kaggle 경로 자동 탐색, 50초 EEG·10분 스펙트로그램 절단
│   │   ├── filters.py        # 결측 보간, 0.5~20Hz 대역 통과 필터
│   │   ├── targets.py        # 투표→확률, 축 보조 정답, 마스크, Seizure 이진 정답, 대표 구간 선택
│   │   ├── montage.py        # 더블 바나나 16채널, 좌우 반전 (EEG·스펙트로그램)
│   │   ├── folds.py          # patient_id 기준 GroupKFold, 누수 검사
│   │   └── build.py          # hms-processed 배열 생성 (병렬, 단일 npy 저장), eeg_segment (학습·제출 공통 EEG 전처리)
│   ├── quality/
│   │   └── signal_qc.py      # 1단계 신호 품질 지표
│   ├── profile/
│   │   └── qeeg.py           # 2단계 프로파일 지표 7종 (64개, 10/21 수정 지표 포함)
│   ├── models/
│   │   └── eeg_spec.py       # 50초 EEG → 스펙트로그램(GPU STFT), 입력 방식별 분류기 HMSNet (E3·E4)
│   ├── reasoning/
│   │   ├── rules.py          # R1 : H5 지지 지표 근거, 학습 fold 임계값(Youden), 판정-근거 일치 판정
│   │   └── report.py         # 소견·판정·근거·신뢰도 리포트 문구
│   └── eval/
│       └── metrics.py        # KL(전체/고득표), 축별 AUC, 민감도@특이도
├── tests/                    # 핵심 함수 단위 테스트 (pytest)
├── experiments/results.csv   # 실험 결과표
└── reports/                  # make_figures.py, figures/(결과 그림), 분석 리포트, 사례 리포트
```

## 실험 결과 (10/26 초안)

![score history](reports/figures/fig1_score_history.png)

| 실험 | 내용 | 고득표 CV | Private |
|---|---|---|---|
| E1 | 10분 스펙트로그램 + EfficientNet-B0 (기준선) | 0.501 | 0.532 |
| E2 | + 고득표(≥10표) 2단계 미세조정 | 0.385 | 0.464 |
| E4 | + 10분 스펙트로그램과 50초 EEG 스펙트로그램 결합 | 0.323 | 0.391 |
| **E8** | **0.85·mean(E4, E5) + 0.15·P1v2 (프로파일 지표 모델)** | **0.296** | **0.374** |

- 편측 AUC 0.945 · 주기성 AUC 0.936 · Seizure 민감도(특이도 0.90) 0.956 (E8 OOF)
- 고득표 CV↔Private 차이는 +0.07~0.08로 일정 : 실험 간 순위·개선폭은 CV로 판단, 절대값은 Private으로 보고

### 가설 판정

| 가설 | 판정 | 근거 |
|---|---|---|
| H1 고득표 미세조정 | 지지 | CV +0.091, Private +0.068, E3~E6에서 재현 |
| H2 10분 + 50초 결합 입력 | 지지 | 단독 입력 최고 대비 −0.049 (초기·나중 결합 차이 없음) |
| H3 좌우 반전 증강 | 미지지 | LPD 개선, LRDA 악화, 고득표 KL +0.012 |
| H4 축 분해 멀티태스크 | 기각 | 편측↔전반 혼동률 변화 없음 (이미 3~5%로 낮음) |
| H5 프로파일 지표 방향성 | 6/12 지지 | 편측성·율동성·Seizure(수정 지표) 지지, 주기성 확정 실패 |

![per-class KL](reports/figures/fig2_per_class_kl.png)

### 추론 로직(R1) : 판정 근거 검증
- 사전 규칙(편측·율동·시간 변화 근거) : 근거 일치율 79.6%, 불일치 플래그 오답률 비 1.36배 → **목표(80%, 2배) 미달**
- 진단 : 고득표 오답의 77%가 근거 지표가 없는 Other 경계. 근거가 적용되는 축에서는 편측 2.60배, 율동 2.05배로 작동, 시간 변화 근거는 판별력 없음(1.00배)
- 보완(사후 1회, R1b : 편측·율동 근거 또는 저확신) : 80.1%, 2.72배로 수치 달성. 단, 저확신 조건만으로 2.96배 → **오류 탐지는 딥러닝 확신도가 담당하고, 프로파일 근거의 역할은 판정 설명과 '어느 판단을 재확인할지' 표시**

![R1 diagnostics](reports/figures/fig3_r1_diagnostics.png)

### 대표 사례 : 근거가 의심 축을 알려준 오답 (전문가 판정 GPD 85%)
```
[소견] 우측 우세 편측 주기성 방전(LPD), 최대 이상 부위 우측 외측(RL)
[판정] 편측 주기성 방전(LPD) 83% (2순위 전반 주기성 방전(GPD) 11%)
[근거] 편측성 : 좌우 파워 비대칭 0.164 (미충족, 기준 > 0.203), 좌우 델타 비대칭 0.144 (미충족, 기준 > 0.245)
[근거] 주기성 : 정량 근거 미확정 (H5 미지지, 딥러닝 판단만 제시)
[재확인] 편측·전반 판단 재확인
[신뢰도] 중간 (최고 확률 0.83, 근거 불일치 : 편측·전반 판단 재확인)
```
- 사례 8건 해설 : [reports/case_reports/commentary.md](reports/case_reports/commentary.md) (클래스별 정답 5, 저확신으로 걸러진 오답, 근거가 의심 축을 알려준 오답, 놓친 오답)

## 한계
- HMS는 연속 기록이 아닌 구간 단위 데이터 : 시간당 오경보율 산출 불가, 특이도로 대체
- 투표수 편차(1~28표)로 저득표 샘플의 라벨 신뢰도가 낮음
- 연구·학습 목적 프로젝트이며 임상 진단 도구가 아님
