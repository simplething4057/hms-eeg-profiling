"""설정 로드 : configs/base.yaml을 읽어 dict로 반환한다.

Kaggle에서는 hms-src Dataset 안의 configs/base.yaml을 기본값으로 사용한다.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs" / "base.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
