# AWS 배포 설정 가이드 (D3)

## 1. AWS CLI 설치 (Windows)

```powershell
# PowerShell 관리자 모드에서 실행
msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi /quiet
# 설치 후 터미널 재시작
aws --version
```

## 2. IAM 사용자 생성 (CI/CD용)

1. AWS 콘솔 → IAM → Users → **Create user**
2. 이름: `hms-eeg-deploy`
3. 권한 정책 직접 연결:
   - `AmazonEC2FullAccess`
   - `AmazonEC2ContainerRegistryFullAccess`
4. Security credentials 탭 → **Create access key** → CLI 선택
5. **Access Key ID** 와 **Secret Access Key** 저장

## 3. IAM Role 생성 (EC2 → ECR pull용)

EC2 인스턴스가 ECR에서 이미지를 pull하려면 Instance Profile이 필요합니다.

1. IAM → Roles → **Create role**
2. Trusted entity: **AWS service** → EC2
3. 정책 연결:
   - `AmazonEC2ContainerRegistryReadOnly`
4. Role name: **`hms-eeg-ec2-role`** (deploy.ps1에서 이 이름 사용)
5. Create role

## 4. AWS CLI 자격증명 설정

```powershell
aws configure
# AWS Access Key ID: <2번에서 저장한 키>
# AWS Secret Access Key: <2번에서 저장한 키>
# Default region name: ap-northeast-2
# Default output format: json
```

확인:
```powershell
aws sts get-caller-identity
```

## 5. 전체 배포 실행

```powershell
cd C:\Users\USER\Documents\Claude\Projects\hms-eeg-portfolio
.\scripts\deploy.ps1
```

ECR push만 (EC2 없이):
```powershell
.\scripts\deploy.ps1 -EcrOnly
```

## 6. GitHub Actions secrets 설정

[설정 페이지](https://github.com/simplething4057/hms-eeg-profiling/settings/secrets/actions) 에서 추가:

| Secret 이름 | 값 |
|---|---|
| `AWS_ACCESS_KEY_ID` | `aws configure get aws_access_key_id` 출력값 |
| `AWS_SECRET_ACCESS_KEY` | `aws configure get aws_secret_access_key` 출력값 |

설정 후 main 브랜치에 push하면 CI가 자동으로 ECR에 이미지를 올립니다.

## 7. 비용 추정 (88달러 크레딧 기준)

| 항목 | 사양 | 월 비용 |
|---|---|---|
| EC2 t3.small | 2 vCPU / 2GB RAM | ~$15 |
| ECR 스토리지 | ~500MB | ~$0.05 |
| 데이터 전송 | 포트폴리오 수준 | ~$1 |
| **합계** | | **~$16/월** |

→ **약 5개월** 운영 가능 (포트폴리오 기간 충분)
