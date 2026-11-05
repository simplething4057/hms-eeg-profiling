"""API 입출력 스키마"""
from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field

CLASSES = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]


class PredictRequest(BaseModel):
    segment_id: str = Field(..., description="구간 식별자")
    qeeg_features: list[float] = Field(
        ..., min_length=67, max_length=67,
        description="67차원 qEEG 피처 벡터 (src.profile.qeeg로 사전 계산)"
    )
    # base64 encoded float32 bytes — shape (400, 556), row-major
    # None이면 P1 단독 추론
    spectrogram_b64: Optional[str] = Field(
        None, description="400×556 float32 스펙트로그램 (base64). None이면 P1 단독 추론"
    )
    include_report: bool = Field(
        False, description="True이면 임상 보고서(clinical_report)를 응답에 포함"
    )


class ClassProb(BaseModel):
    seizure: float
    lpd: float
    gpd: float
    lrda: float
    grda: float
    other: float

    @classmethod
    def from_array(cls, arr: list[float]) -> "ClassProb":
        return cls(**dict(zip(CLASSES, arr)))


class ReportHeader(BaseModel):
    title: str
    subtitle: str
    segment_id: str
    analyzed_at: str
    model: str
    disclaimer: str


class ReportFindings(BaseModel):
    primary_finding: str
    acns_code: str
    confidence: str
    review_flag: bool
    review_reason: str
    probability_distribution: list[str]
    description: str


class ReportClinical(BaseModel):
    acns_criteria: str
    clinical_significance: str


class ReportUrgency(BaseModel):
    label: str
    level: int


class ClinicalReport(BaseModel):
    header: ReportHeader
    findings: ReportFindings
    clinical: ReportClinical
    differential: list[str]
    recommendations: list[str]
    urgency: ReportUrgency
    reference: str


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    segment_id: str
    probabilities: ClassProb
    predicted_class: str
    confidence: float = Field(..., description="최고 클래스 확률")
    r1b_flag: bool = Field(..., description="전문의 우선 확인 플래그")
    r1b_flag_reason: str = Field(..., description="플래그 사유 (저확신 / 근거불일치 / 없음)")
    model_used: str = Field(..., description="e8_ensemble | p1_only")
    processing_ms: float
    clinical_report: Optional[ClinicalReport] = Field(
        None, description="임상 보고서 (include_report=True 시 포함)"
    )


class ModelInfoResponse(BaseModel):
    version: str
    recipe: dict
    dl_loaded: bool
    p1_loaded: bool
    n_folds: int
    classes: list[str]
