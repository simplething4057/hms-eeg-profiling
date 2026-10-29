param(
    [string]$Region      = "ap-northeast-2",
    [string]$InstanceId  = "i-05475380d8e492d67",
    [string]$AccountId   = "641150854945",
    [string]$RepoName    = "hms-eeg-api",
    [string]$ImageTag    = "latest",
    [string]$AwsExe      = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
)

$EcrUri   = "$AccountId.dkr.ecr.$Region.amazonaws.com"
$ImageUri = "$EcrUri/${RepoName}:${ImageTag}"

function Log($msg) { Write-Host "[$(Get-Date -Format HH:mm:ss)] $msg" -ForegroundColor Cyan }
function Ok($msg)  { Write-Host "  OK  $msg" -ForegroundColor Green }
function Err($msg) { Write-Host "  ERR $msg" -ForegroundColor Red }

$AwsCli = if (Get-Command aws -ErrorAction SilentlyContinue) { "aws" } else { $AwsExe }

# ── EC2 에서 실행할 deploy bash 스크립트 ─────────────────────────────────────
# - 헬스체크 통과 시 정상 완료
# - 헬스체크 실패 시 이전 이미지로 자동 롤백 후 exit 1 (SSM Status = Failed)
# @'...'@ 단일인용 here-string : PowerShell 이 $ 를 확장하지 않음
$DeployScript = @'
#!/bin/bash
set -e
ECR_URI="__ECR_URI__"
IMAGE="__IMAGE_URI__"
PREV=$(docker inspect hms-eeg-api --format '{{.Image}}' 2>/dev/null || echo "")
echo "Previous image: ${PREV:-none}"
aws ecr get-login-password --region __REGION__ | docker login --username AWS --password-stdin "$ECR_URI"
docker pull "$IMAGE"
docker stop hms-eeg-api 2>/dev/null || true
docker rm hms-eeg-api 2>/dev/null || true
docker run -d --name hms-eeg-api --restart always -p 8000:8000 -e PYTHONUNBUFFERED=1 "$IMAGE"
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
  docker logs hms-eeg-api --tail 30 2>&1 || true
  if [ -n "$PREV" ]; then
    echo "Rolling back to: $PREV"
    docker stop hms-eeg-api || true
    docker rm hms-eeg-api || true
    docker run -d --name hms-eeg-api --restart always -p 8000:8000 -e PYTHONUNBUFFERED=1 "$PREV" && echo "Rollback complete" || echo "Rollback FAILED"
  else
    echo "No previous image to roll back to"
  fi
  exit 1
fi
'@

# PowerShell 변수 치환 (bash $는 here-string에서 이미 리터럴)
$DeployScript = $DeployScript `
    -replace '__ECR_URI__',   $EcrUri `
    -replace '__IMAGE_URI__', $ImageUri `
    -replace '__REGION__',    $Region

# CRLF → LF (Windows here-string이 CRLF로 생성되어 EC2 bash 줄 계속이 깨지는 현상 방지)
$DeployScript = $DeployScript -replace "`r`n", "`n"

# Base64 인코딩 ($ · 특수문자 이스케이프 없이 SSM 에 전달)
$Bytes = [System.Text.Encoding]::UTF8.GetBytes($DeployScript)
$B64   = [Convert]::ToBase64String($Bytes)

$DeployCmd = "echo '$B64' | base64 -d | bash"

# SSM 파라미터 JSON (UTF-8 BOM 없이 저장)
$paramsObj  = [ordered]@{ commands = @($DeployCmd) }
$paramsJson = $paramsObj | ConvertTo-Json -Depth 3

$TmpJson = "$env:TEMP\ssm_params.json"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($TmpJson, $paramsJson, $utf8NoBom)
$TmpJsonFwd = $TmpJson -replace '\\', '/'

Log "Sending SSM send-command..."
$CmdId = & $AwsCli ssm send-command `
    --instance-ids $InstanceId `
    --document-name "AWS-RunShellScript" `
    --parameters "file://$TmpJsonFwd" `
    --region $Region `
    --query "Command.CommandId" `
    --output text 2>&1

if ($LASTEXITCODE -ne 0) {
    Err "send-command failed: $CmdId"
    Write-Host ""
    Write-Host "  Manual fallback — AWS Console Session Manager:" -ForegroundColor Yellow
    Write-Host "  https://console.aws.amazon.com/ec2/v2/home?region=$Region#Instances"
    exit 1
}

Ok "CommandId: $CmdId"
Log "Waiting for completion (max 120s, includes health-check)..."

# SSM 상태 폴링 (최대 24회 × 5s = 120s)
for ($i = 0; $i -lt 24; $i++) {
    Start-Sleep -Seconds 5
    $Status = & $AwsCli ssm get-command-invocation `
        --command-id $CmdId `
        --instance-id $InstanceId `
        --region $Region `
        --query "Status" `
        --output text 2>$null
    Write-Host "  [$([int]($i+1)*5)s] SSM: $Status"

    if ($Status -eq "Success") {
        $Out = & $AwsCli ssm get-command-invocation `
            --command-id $CmdId `
            --instance-id $InstanceId `
            --region $Region `
            --query "StandardOutputContent" `
            --output text
        Write-Host ""
        Ok "Redeployment complete!"
        Write-Host $Out
        Write-Host ""
        Write-Host "  Dashboard : http://3.38.165.173:8000/ui" -ForegroundColor Green
        Write-Host "  API docs  : http://3.38.165.173:8000/docs" -ForegroundColor Green
        exit 0
    }

    if ($Status -eq "Failed" -or $Status -eq "Cancelled") {
        Write-Host ""
        $Out = & $AwsCli ssm get-command-invocation `
            --command-id $CmdId `
            --instance-id $InstanceId `
            --region $Region `
            --query "StandardOutputContent" `
            --output text
        $ErrOut = & $AwsCli ssm get-command-invocation `
            --command-id $CmdId `
            --instance-id $InstanceId `
            --region $Region `
            --query "StandardErrorContent" `
            --output text
        Err "Deploy failed (health check 실패 or 롤백):"
        Write-Host $Out
        if ($ErrOut) { Write-Host "STDERR: $ErrOut" -ForegroundColor DarkRed }
        Write-Host ""
        Write-Host "  롤백이 실행됐습니다. EC2 상태를 확인하세요:" -ForegroundColor Yellow
        Write-Host "  http://3.38.165.173:8000/health" -ForegroundColor White
        exit 1
    }
}

Write-Host ""
Write-Host "  Timeout — check manually: http://3.38.165.173:8000/ui" -ForegroundColor Yellow
