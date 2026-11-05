#!/bin/bash
# 시연용 자체 서명 인증서 생성 (실제 임상: Let's Encrypt 또는 병원 내부 CA로 교체)
mkdir -p certs
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout certs/server.key \
  -out    certs/server.crt \
  -subj "/C=KR/ST=Seoul/O=HMS-EEG-Demo/CN=localhost"
echo "✅ 인증서 생성 완료: nginx/certs/server.crt"
