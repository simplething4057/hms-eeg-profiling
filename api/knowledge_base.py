"""임상 EEG 지식 베이스 — ACNS 표준화 중환자 EEG 용어 (2021) 기반

참고:
  Hirsch LJ et al. "American Clinical Neurophysiology Society's Standardized
  Critical Care EEG Terminology: 2021 Version."
  J Clin Neurophysiol. 2021;38(1):1-29.
"""
from __future__ import annotations

KNOWLEDGE_BASE: dict[str, dict] = {
    "seizure": {
        "label_ko": "발작 (Seizure)",
        "label_en": "Electrographic Seizure",
        "acns_code": "SZ",
        "description": (
            "명확한 발작파 패턴이 관찰됩니다. 뇌파에서 갑작스러운 리듬 변화와 "
            "발작적 진화(ictal evolution)가 확인되며, 임상적 발작 증상 동반 여부와 "
            "무관하게 뇌전증 발작에 준하여 처치가 필요합니다."
        ),
        "eeg_features": (
            "시작 : 국소 또는 전반성 고진폭(>50 µV) 리듬성 방전 (주로 5–25 Hz).\n"
            "경과 : 주파수 감소·진폭 증가·공간 확산으로 진화 필수 — 진화 없으면 발작으로 분류 불가.\n"
            "종료 : 갑작스러운 소멸 후 국소/전반성 서파 억제(postictal suppression) 및 PLEDS 동반 가능."
        ),
        "iic_spectrum": "IIC 연속체 최고단 (Ictal) — 명확한 발작파 활동. 즉각 개입 필요.",
        "acns_criteria": (
            "≥10초 지속되는 반복 방전으로, 주파수·형태·진폭·공간 분포 중 "
            "≥1개 항목이 명확히 진화(evolve)하거나, >2.5Hz의 주기성 방전이 "
            "10초 이상 지속되는 경우. 비경련성 뇌전증지속증(NCSE) 포함."
        ),
        "clinical_significance": (
            "급성 뇌손상(뇌졸중·외상·뇌염)에서의 이차적 발작, "
            "대사성 원인(저혈당·전해질 이상·간성 뇌증), "
            "기존 뇌전증의 악화를 의미할 수 있습니다. "
            "비경련성 발작지속증(NCSE)은 의식 저하만으로 나타날 수 있어 "
            "EEG 없이는 진단이 불가능합니다."
        ),
        "differential": [
            "비경련성 발작지속증 (NCSE)",
            "전신 강직-간대 발작 (GTCS)",
            "국소 발작 (Focal onset seizure)",
            "미세 발작 (Subtle seizure) — 운동 증상 미약",
        ],
        "recommendations": [
            "즉각적인 신경과 전문의 긴급 협진",
            "항경련제 1차 투여 (벤조디아제핀 IV) 고려",
            "지속 EEG 모니터링 (cEEG) 유지",
            "NCSE 프로토콜 적용 여부 판단",
            "원인 감별: 혈당·전해질·간·신 기능 검사",
            "뇌 CT/MRI 긴급 시행 고려",
        ],
        "urgency": "긴급 (STAT)",
        "urgency_level": 3,
        "reference": "ACNS 2021, SZ 정의; Brophy GM et al. Neurocrit Care. 2012",
    },

    "lpd": {
        "label_ko": "편측 주기성 방전 (LPD)",
        "label_en": "Lateralized Periodic Discharges",
        "acns_code": "LPD",
        "description": (
            "한쪽 반구에 국한된 주기성 방전이 관찰됩니다. "
            "국소 뇌 병변과 연관성이 높으며, 발작으로의 진행 위험이 있습니다."
        ),
        "eeg_features": (
            "위상 역전(phase reversal)으로 국소 발생원 확인.\n"
            "극파 또는 극-서파 복합체 반복 (주기 일정, 50% 이상 출현율).\n"
            "중첩된 빠른 활동(superimposed fast activity) 있을 시 'LPD+' — 발작에 더 가까움.\n"
            "진폭 : 통상 50–200 µV, 병변 반대 반구보다 2배 이상 우세."
        ),
        "iic_spectrum": "IIC 연속체 중간 (Interictal-Ictal 경계) — '+F'/'+R' 수식어 동반 시 발작에 근접.",
        "acns_criteria": (
            "편측(또는 편측 우세) 반구에서 0.5–3Hz 주기로 반복되는 방전. "
            "방전 간 배경파가 억제되거나 불연속적일 수 있음. "
            "'+F'(fast 성분), '+R'(rhythmic 성분) 수식어로 발작 위험도 세분화."
        ),
        "clinical_significance": (
            "급성 국소 뇌 병변(허혈성 뇌졸중, 뇌출혈, 뇌염, 뇌종양)에서 흔히 관찰됩니다. "
            "LPD는 '아이코탈-인터아이코탈 연속체(IIC)'에 위치하며, "
            "약 20–30%에서 임상 발작으로 진행합니다. "
            "'+F' 수식어(fast 성분 추가) 시 발작 위험도가 더 높습니다."
        ),
        "differential": [
            "뇌졸중 후 LPD (Post-stroke LPD)",
            "단순 포진 뇌염 (HSV encephalitis)",
            "국소 종양 또는 경막하 혈종",
            "PRES (가역적 후두부 뇌병증)",
        ],
        "recommendations": [
            "연속 EEG 모니터링 지속 (발작 진행 여부 감시)",
            "항경련제 투여 고려 (levetiracetam 또는 lacosamide)",
            "MRI/MRA 뇌 영상 확인 (국소 병변 감별)",
            "신경과 전문의 협진",
            "CSF 검사 고려 (뇌염 의심 시)",
        ],
        "urgency": "준긴급 (Urgent)",
        "urgency_level": 2,
        "reference": "ACNS 2021, LPD 정의; Rodriguez Ruiz A et al. Neurology. 2017",
    },

    "gpd": {
        "label_ko": "전반 주기성 방전 (GPD)",
        "label_en": "Generalized Periodic Discharges",
        "acns_code": "GPD",
        "description": (
            "양측 반구에 걸친 전반 주기성 방전이 관찰됩니다. "
            "미만성 뇌 기능 저하를 시사하며, 원인 감별이 필요합니다."
        ),
        "eeg_features": (
            "전반성 스파이크, 폴리스파이크 또는 샤프-서파 복합체 (0.5–3 Hz).\n"
            "삼상파 : 전두엽 우세, 앞→뒤 위상 지연(anterior-to-posterior lag), 양성 1상-음성 2상-양성 3상.\n"
            "Burst-suppression : 고진폭 방전군(burst)과 전압 억제기(suppression)가 교대 — 마취·저산소 뇌증 감별 필수."
        ),
        "iic_spectrum": "IIC 연속체 중간 (Interictal 범주) — 발작 동반 여부 감별 필요. Burst-suppression 동반 시 중증 뇌손상 시사.",
        "acns_criteria": (
            "양측 반구에 대칭적 또는 거의 대칭적으로 나타나는 주기성 방전. "
            "0.5–3Hz 반복. 삼상파(triphasic waves)는 GPD의 형태 변형으로 분류. "
            "배경 억제(suppression) 동반 시 뇌 손상 중증도 증가."
        ),
        "clinical_significance": (
            "주요 원인: 대사성 뇌병증(간성 뇌증·요독성 뇌증), "
            "저산소성-허혈성 뇌손상(심정지 후), 패혈성 뇌병증, "
            "크로이츠펠트-야콥병(CJD). "
            "삼상파 형태는 간성 뇌증에서 전형적으로 나타나며, "
            "배경파 억제를 동반한 GPD는 불량한 신경학적 예후와 연관됩니다."
        ),
        "differential": [
            "간성 뇌증 (Hepatic encephalopathy) — 삼상파",
            "저산소성-허혈성 뇌병증 (HIE, 심정지 후)",
            "크로이츠펠트-야콥병 (CJD) — 1Hz GSED",
            "패혈성 뇌병증 (Septic encephalopathy)",
            "전신 강직-간대 발작 후 발작간기 (Postictal)",
        ],
        "recommendations": [
            "원인 감별: 간기능·신기능·전해질·혈당 검사",
            "대사 이상 적극 교정",
            "심정지 후 목표체온요법 환자: 뇌 예후 평가 프로토콜 적용",
            "CJD 의심 시: CSF RT-QuIC 검사, MRI DWI 확인",
            "지속 EEG 모니터링 유지",
            "항경련제는 발작 동반 여부 확인 후 결정",
        ],
        "urgency": "준긴급 (Urgent)",
        "urgency_level": 2,
        "reference": "ACNS 2021, GPD 정의; Young GB. J Clin Neurophysiol. 2009",
    },

    "lrda": {
        "label_ko": "편측 율동성 델타 활동 (LRDA)",
        "label_en": "Lateralized Rhythmic Delta Activity",
        "acns_code": "LRDA",
        "description": (
            "한쪽 반구 우세의 율동성 델타파 활동이 관찰됩니다. "
            "국소 뇌 기능 저하를 반영하며, 발작 연속체에 포함될 수 있습니다."
        ),
        "eeg_features": (
            "국소 0.5–4 Hz 율동성 서파, 편측 우세 (주로 측두·두정 부위).\n"
            "위상 역전으로 병변 반구 확인 — 주기성 방전과 달리 방전 간 배경파 억제 없음.\n"
            "LRDA+ (중첩 빠른 활동) : 발작에 더 근접 — 연속 모니터링 강화 필요.\n"
            "진폭 변화 없이 단조로운 반복 패턴이 특징."
        ),
        "iic_spectrum": "IIC 연속체 하단 (주로 Interictal) — '+F' 수식어 동반 시 ictal 가능성 증가. LPD보다 발작 위험 낮음.",
        "acns_criteria": (
            "편측 반구 우세의 비주기성(nonperiodic) 율동성 델타파(1–4Hz). "
            "LPD와의 차이: 방전 간 배경파 억제 없음. "
            "'+F' 수식어(fast 성분) 동반 시 아이코탈 연속체 가능성 증가."
        ),
        "clinical_significance": (
            "국소 뇌 병변(뇌졸중·종양·염증)으로 인한 편측 뇌 기능 저하를 반영합니다. "
            "LRDA는 발작 발생 위험도가 LPD보다 낮으나, "
            "'+F' 수식어 동반 시 아이코탈 활동으로 간주할 수 있습니다. "
            "ACNS IIC(아이코탈-인터아이코탈 연속체)의 중간 단계에 해당합니다."
        ),
        "differential": [
            "편측 뇌 기능 저하 (Focal cerebral dysfunction)",
            "편측 뇌졸중 또는 뇌출혈",
            "국소 뇌염 또는 뇌농양",
            "발작 후 Todd 마비 동반 국소 서파",
        ],
        "recommendations": [
            "추적 EEG 시행 (12–24시간 내)",
            "뇌 CT 또는 MRI로 구조적 병변 확인",
            "'+F' 수식어 동반 시 항경련제 투여 고려",
            "신경과 협진",
        ],
        "urgency": "주의 (Watch)",
        "urgency_level": 1,
        "reference": "ACNS 2021, LRDA 정의; Gaspard N et al. Epilepsia. 2014",
    },

    "grda": {
        "label_ko": "전반 율동성 델타 활동 (GRDA)",
        "label_en": "Generalized Rhythmic Delta Activity",
        "acns_code": "GRDA",
        "description": (
            "양측 반구에 걸친 전반 율동성 델타파 활동이 관찰됩니다. "
            "미만성 뇌 기능 저하를 시사합니다."
        ),
        "eeg_features": (
            "전반성 1–3 Hz 고진폭 서파, 양측 대칭 (전두부 > 후두부 우세).\n"
            "수면 또는 과호흡 유발 시 정상 변이 여부 감별 필요.\n"
            "전압 : 일반적으로 75–200 µV, 배경 활동 억제 동반 시 중증 시사.\n"
            "GRDA+ (중첩 빠른 활동) : 발작성 변환 가능성 — FIRDA/OIRDA 형태 확인."
        ),
        "iic_spectrum": "IIC 연속체 하단 (대부분 Interictal) — 성인에서 대부분 비간질성. 소아 OIRDA는 결신 발작 감별 필요.",
        "acns_criteria": (
            "양측 반구에 대칭적으로 나타나는 비주기성 율동성 델타파(1–4Hz). "
            "GRDA의 대표적 형태: FIRDA(전두부 간헐 율동성 델타 활동), "
            "OIRDA(후두부 간헐 율동성 델타 활동). "
            "수면·각성 상태, 자극에 따른 반응성 평가 중요."
        ),
        "clinical_significance": (
            "미만성 뇌 기능 저하의 비특이적 지표입니다. "
            "FIRDA: 대사성·중독성 뇌병증, 수두증, 심부 반구 병변. "
            "OIRDA: 소아에서 결신 발작과 연관 가능. "
            "성인에서 GRDA는 대부분 비간질성이나 대사성 원인 교정이 필요합니다."
        ),
        "differential": [
            "대사성 뇌병증 (전해질 이상·약물·독소)",
            "수두증 (Hydrocephalus)",
            "미만성 뇌 위축 또는 허혈",
            "깊은 수면 상태 (N2–N3 수면)",
            "소아: 결신 발작 (OIRDA)",
        ],
        "recommendations": [
            "대사 원인 교정 (전해질·혈당·약물 독성 확인)",
            "각성 수준 및 반응성 평가",
            "뇌 영상(CT/MRI) 고려",
            "소아 환자: 항경련제 투여 여부 전문의 판단",
            "원인 미상 시 추적 EEG 시행",
        ],
        "urgency": "주의 (Watch)",
        "urgency_level": 1,
        "reference": "ACNS 2021, GRDA 정의; Accolla EA et al. Epilepsia. 2011",
    },

    "other": {
        "label_ko": "정상 범위 / 기타",
        "label_en": "Normal / Other",
        "acns_code": "N/A",
        "description": (
            "분석된 EEG 구간에서 위 5개 주요 패턴(발작·LPD·GPD·LRDA·GRDA)에 "
            "해당하는 명확한 이상 소견이 관찰되지 않습니다."
        ),
        "eeg_features": (
            "정상 : 8–13 Hz 후두 우세 알파파, 적절한 베타 활동, 수면 방추(sleep spindle)·K-complex 정상 출현.\n"
            "경증 이상 : 미만성 4–7 Hz 세타파화, 알파 기저 활동 감소, 경미한 좌우 비대칭.\n"
            "저진폭 배경 : 전반성 억제 — 진정제 효과·저산소 뇌증 감별 필요.\n"
            "아티팩트 : 근전도(EMG)·체동·전극 잡음에 의한 허위 패턴 여부 확인."
        ),
        "iic_spectrum": "N/A — 상기 IIC 연속체 패턴에 해당하지 않음.",
        "acns_criteria": (
            "배경파가 연령·각성 수준에 적절하거나, "
            "상기 5개 패턴 기준에 해당하지 않는 비특이적 소견."
        ),
        "clinical_significance": (
            "현재 분석 구간에서 중증 뇌전증 패턴은 관찰되지 않습니다. "
            "다만 EEG는 구간 단위 분석이므로, 임상 증상이 지속된다면 "
            "연속 EEG 모니터링을 통해 간헐적 발작 여부를 확인해야 합니다."
        ),
        "differential": [
            "정상 각성 EEG",
            "경미한 비특이적 서파화 (Mild nonspecific slowing)",
            "수면 EEG",
            "근전도(EMG) 또는 체동 아티팩트",
        ],
        "recommendations": [
            "임상 증상 지속 시 연속 EEG 모니터링 고려",
            "정기적 임상 평가 유지",
        ],
        "urgency": "정상 (Routine)",
        "urgency_level": 0,
        "reference": "ACNS 2021 Background Activity 분류 기준",
    },
}
