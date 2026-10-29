FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 (lightgbm 빌드에 필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 먼저 설치 (레이어 캐시 활용)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# CPU 전용 PyTorch (DL 추론 활성화 시 주석 해제)
# RUN pip install --no-cache-dir torch==2.3.0+cpu timm==0.9.16 \
#     --index-url https://download.pytorch.org/whl/cpu

# 소스 코드 + 모델 가중치 복사
COPY src/ src/
COPY api/ api/
COPY configs/ configs/
COPY e8_recipe.json .
COPY static/ static/
# models/ 는 ECR 이미지에 포함하거나 EFS 마운트 — 아래는 로컬 빌드용
COPY models/ models/

EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
