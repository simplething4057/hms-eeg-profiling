#!/bin/bash
# EC2 배포 스크립트 — release.yml 이 base64 인코딩해서 SSM 으로 전달
# IMAGE_URI 자리 표시자는 release.yml 에서 실제 값으로 치환됩니다.
set -e

ECR_URI="641150854945.dkr.ecr.ap-northeast-2.amazonaws.com"
IMAGE="HMS_IMAGE_URI_PLACEHOLDER"

# 롤백용 이전 컨테이너 이미지 저장
PREV=$(docker inspect hms-eeg-api --format '{{.Image}}' 2>/dev/null || echo "")
echo "Previous image: ${PREV:-none}"

# ECR 로그인
aws ecr get-login-password --region ap-northeast-2 \
  | docker login --username AWS --password-stdin "$ECR_URI"

# 새 이미지 풀
docker pull "$IMAGE"

# 컨테이너 교체
docker stop hms-eeg-api 2>/dev/null || true
docker rm   hms-eeg-api 2>/dev/null || true
docker run -d --name hms-eeg-api \
  --restart always -p 8000:8000 -e PYTHONUNBUFFERED=1 "$IMAGE"

# 헬스체크 (30s, 5s 간격 x 6회)
echo "Waiting for service..."
HEALTHY=0
for i in 1 2 3 4 5 6; do
  sleep 5
  HTTP=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null || echo 000)
  echo "[${i}0s] Health: HTTP $HTTP"
  if [ "$HTTP" = "200" ]; then HEALTHY=1; break; fi
done

if [ "$HEALTHY" -eq 1 ]; then
  echo "=== Deploy OK ==="
  docker ps --filter name=hms-eeg-api --format 'Status: {{.Status}}'
else
  echo "=== HEALTH CHECK FAILED ==="
  echo "--- Container logs (last 30 lines) ---"
  docker logs hms-eeg-api --tail 30 2>&1 || true
  echo "--------------------------------------"
  if [ -n "$PREV" ]; then
    echo "Rolling back to: $PREV"
    docker stop hms-eeg-api || true
    docker rm   hms-eeg-api || true
    docker run -d --name hms-eeg-api \
      --restart always -p 8000:8000 -e PYTHONUNBUFFERED=1 "$PREV" \
      && echo "Rollback complete" \
      || echo "Rollback FAILED — manual intervention required"
  else
    echo "No previous image to roll back to"
  fi
  exit 1
fi
