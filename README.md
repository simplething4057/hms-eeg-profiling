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
| 5 | `nb05_submit` | GPU, 인터넷 OFF | `submission.csv` |
| 6 | `nb06_reasoning_report` | CPU | 추론 검증, 사례 리포트 |

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
│   │   ├── targets.py        # 투표→확률, 축 보조 정답, 마스크, Seizure 이진 정답, eeg_id 집계
│   │   ├── montage.py        # 더블 바나나 16채널
│   │   └── folds.py          # patient_id 기준 GroupKFold
│   └── eval/
│       └── metrics.py        # KL(전체/고득표), 축별 AUC, 민감도@특이도
├── tests/                    # 핵심 함수 단위 테스트 (pytest)
├── experiments/results.csv   # 실험 결과표
└── reports/                  # 그래프, 분석 리포트, 사례 리포트
```

## 실험 결과
(실험 완료 후 기입 : 결과표, 핵심 그래프, 사례 리포트 1건)

## 한계
- HMS는 연속 기록이 아닌 구간 단위 데이터 : 시간당 오경보율 산출 불가, 특이도로 대체
- 투표수 편차(1~28표)로 저득표 샘플의 라벨 신뢰도가 낮음
- 연구·학습 목적 프로젝트이며 임상 진단 도구가 아님
