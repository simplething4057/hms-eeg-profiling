# HMS EEG API — ECR push + EC2 배포 스크립트
# 실행 전: aws configure 완료 필요
# 사용법: .\scripts\deploy.ps1 [-Region ap-northeast-2] [-InstanceType t3.small]

param(
    [string]$Region = "ap-northeast-2",
    [string]$InstanceType = "t3.small",
    [string]$RepoName = "hms-eeg-api",
    [switch]$EcrOnly   # ECR push만 (EC2 건너뜀)
)

$ErrorActionPreference = "Stop"

# ── 유틸 ──────────────────────────────────────────────
function Log($msg) { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $msg" -ForegroundColor Cyan }
function Ok($msg)  { Write-Host "  OK  $msg" -ForegroundColor Green }
function Err($msg) { Write-Host "  ERR $msg" -ForegroundColor Red; exit 1 }

# ── 1. 계정 ID 확인 ───────────────────────────────────
Log "AWS 자격증명 확인..."
$AccountId = (aws sts get-caller-identity --query Account --output text 2>&1)
if ($LASTEXITCODE -ne 0) { Err "aws configure 가 완료되지 않았습니다. scripts/setup_aws.md 를 먼저 읽어주세요." }
Ok "Account: $AccountId  Region: $Region"

$EcrUri = "$AccountId.dkr.ecr.$Region.amazonaws.com"
$ImageUri = "$EcrUri/${RepoName}:latest"

# ── 2. ECR 레포 생성 (이미 있으면 건너뜀) ──────────────
Log "ECR 레포 확인/생성..."
$existing = aws ecr describe-repositories --repository-names $RepoName --region $Region 2>&1
if ($LASTEXITCODE -ne 0) {
    aws ecr create-repository `
        --repository-name $RepoName `
        --region $Region `
        --image-scanning-configuration scanOnPush=true `
        --encryption-configuration encryptionType=AES256 | Out-Null
    Ok "ECR 레포 생성: $RepoName"
} else {
    Ok "ECR 레포 기존 사용: $RepoName"
}

# ── 3. Docker 로그인 + 빌드 + push ────────────────────
Log "ECR Docker 로그인..."
aws ecr get-login-password --region $Region |
    docker login --username AWS --password-stdin $EcrUri
if ($LASTEXITCODE -ne 0) { Err "ECR 로그인 실패" }

Log "Docker 이미지 빌드 (models/ 포함)..."
$ProjectRoot = Split-Path -Parent $PSScriptRoot
docker build -t "${RepoName}:latest" $ProjectRoot
if ($LASTEXITCODE -ne 0) { Err "Docker 빌드 실패" }

docker tag "${RepoName}:latest" $ImageUri
docker push $ImageUri
if ($LASTEXITCODE -ne 0) { Err "ECR push 실패" }
Ok "ECR push 완료: $ImageUri"

if ($EcrOnly) { Log "ECR-only 모드 — 완료"; exit 0 }

# ── 4. EC2 보안 그룹 ──────────────────────────────────
Log "보안 그룹 설정..."
$SgName = "hms-eeg-api-sg"
$SgId = (aws ec2 describe-security-groups `
    --filters "Name=group-name,Values=$SgName" `
    --query "SecurityGroups[0].GroupId" `
    --output text --region $Region 2>&1)

if ($SgId -eq "None" -or $SgId -eq "") {
    $SgId = (aws ec2 create-security-group `
        --group-name $SgName `
        --description "HMS EEG API server" `
        --region $Region `
        --query GroupId --output text)
    # SSH (22), HTTP (80), API (8000)
    foreach ($port in @(22, 80, 8000)) {
        aws ec2 authorize-security-group-ingress `
            --group-id $SgId `
            --protocol tcp --port $port `
            --cidr 0.0.0.0/0 `
            --region $Region | Out-Null
    }
    Ok "보안 그룹 생성: $SgId"
} else {
    Ok "보안 그룹 기존 사용: $SgId"
}

# ── 5. EC2 User Data (Docker + 컨테이너 자동 기동) ───────
$UserData = @"
#!/bin/bash
set -e
yum update -y
yum install -y docker
systemctl enable docker
systemctl start docker

# ECR 로그인 (인스턴스 프로파일 필요 — ecr:GetAuthorizationToken 권한)
aws ecr get-login-password --region $Region | \
    docker login --username AWS --password-stdin $EcrUri

docker pull $ImageUri

docker run -d \
    --name hms-eeg-api \
    --restart always \
    -p 8000:8000 \
    -e PYTHONUNBUFFERED=1 \
    $ImageUri
"@

$UserDataB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($UserData))

# ── 6. 최신 Amazon Linux 2023 AMI ───────────────────
Log "AMI 조회..."
$AmiId = (aws ec2 describe-images `
    --owners amazon `
    --filters "Name=name,Values=al2023-ami-2023*-x86_64" `
              "Name=state,Values=available" `
    --query "sort_by(Images,&CreationDate)[-1].ImageId" `
    --output text --region $Region)
Ok "AMI: $AmiId"

# ── 7. EC2 인스턴스 시작 ─────────────────────────────
Log "EC2 인스턴스 시작 ($InstanceType)..."
$InstanceId = (aws ec2 run-instances `
    --image-id $AmiId `
    --instance-type $InstanceType `
    --security-group-ids $SgId `
    --user-data $UserDataB64 `
    --iam-instance-profile Name=hms-eeg-ec2-role `
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=hms-eeg-api}]" `
    --region $Region `
    --query "Instances[0].InstanceId" `
    --output text 2>&1)

if ($LASTEXITCODE -ne 0) {
    Write-Host $InstanceId -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  IAM 인스턴스 프로파일이 없어서 EC2 시작 실패." -ForegroundColor Yellow
    Write-Host "  scripts/setup_aws.md 의 IAM 섹션을 참고해 hms-eeg-ec2-role 을 만들어 주세요." -ForegroundColor Yellow
    Write-Host "  또는 --EcrOnly 플래그로 ECR push만 실행할 수 있습니다." -ForegroundColor Yellow
    exit 1
}

Ok "인스턴스 시작: $InstanceId"

# ── 8. 퍼블릭 IP 대기 ────────────────────────────────
Log "퍼블릭 IP 할당 대기 (최대 60초)..."
$Ip = ""
for ($i = 0; $i -lt 12; $i++) {
    Start-Sleep -Seconds 5
    $Ip = (aws ec2 describe-instances `
        --instance-ids $InstanceId `
        --query "Reservations[0].Instances[0].PublicIpAddress" `
        --output text --region $Region)
    if ($Ip -and $Ip -ne "None") { break }
}

if (-not $Ip -or $Ip -eq "None") {
    Ok "인스턴스 시작됨 (IP 아직 미할당). 나중에 확인:"
    Write-Host "  aws ec2 describe-instances --instance-ids $InstanceId --query 'Reservations[0].Instances[0].PublicIpAddress' --output text --region $Region"
} else {
    Ok "배포 완료!"
    Write-Host ""
    Write-Host "  API 엔드포인트  : http://${Ip}:8000" -ForegroundColor Green
    Write-Host "  헬스체크        : http://${Ip}:8000/health" -ForegroundColor Green
    Write-Host "  API 문서        : http://${Ip}:8000/docs" -ForegroundColor Green
    Write-Host ""
    Write-Host "  ※ Docker 기동까지 약 2-3분 소요됩니다." -ForegroundColor Yellow
}

# ── 9. GitHub Actions secrets 안내 ───────────────────
Write-Host ""
Write-Host "GitHub Actions secrets 설정 (github.com/simplething4057/hms-eeg-profiling/settings/secrets/actions):" -ForegroundColor Cyan
Write-Host "  AWS_ACCESS_KEY_ID     = (aws configure get aws_access_key_id)" -ForegroundColor DarkGray
Write-Host "  AWS_SECRET_ACCESS_KEY = (aws configure get aws_secret_access_key)" -ForegroundColor DarkGray
