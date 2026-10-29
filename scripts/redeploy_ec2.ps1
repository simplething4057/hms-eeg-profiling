param(
    [string]$Region      = "ap-northeast-2",
    [string]$InstanceId  = "i-05475380d8e492d67",
    [string]$AccountId   = "641150854945",
    [string]$RepoName    = "hms-eeg-api",
    [string]$AwsExe      = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
)

$EcrUri   = "$AccountId.dkr.ecr.$Region.amazonaws.com"
$ImageUri = "$EcrUri/${RepoName}:latest"

function Log($msg) { Write-Host "[$(Get-Date -Format HH:mm:ss)] $msg" -ForegroundColor Cyan }
function Ok($msg)  { Write-Host "  OK  $msg" -ForegroundColor Green }
function Err($msg) { Write-Host "  ERR $msg" -ForegroundColor Red }

$cmd1 = "aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $EcrUri"
$cmd2 = "docker pull $ImageUri"
$cmd3 = "docker stop hms-eeg-api 2>/dev/null; docker rm hms-eeg-api 2>/dev/null; true"
$cmd4 = "docker run -d --name hms-eeg-api --restart always -p 8000:8000 -e PYTHONUNBUFFERED=1 $ImageUri"
$cmd5 = "sleep 3 && docker ps --filter name=hms-eeg-api --format 'Status: {{.Status}}'"

$paramsObj  = [ordered]@{ commands = @($cmd1, $cmd2, $cmd3, $cmd4, $cmd5) }
$paramsJson = $paramsObj | ConvertTo-Json -Depth 3

$TmpJson = "$env:TEMP\ssm_params.json"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($TmpJson, $paramsJson, $utf8NoBom)
$TmpJsonFwd = $TmpJson -replace '\\', '/'

$AwsCli = if (Get-Command aws -ErrorAction SilentlyContinue) { "aws" } else { $AwsExe }
Log "Sending SSM send-command (params via file)..."
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
    Write-Host "  Manual fallback via AWS Console Session Manager:" -ForegroundColor Yellow
    Write-Host "  https://console.aws.amazon.com/ec2/v2/home?region=$Region#Instances" -ForegroundColor White
    Write-Host "  Select i-05475380d8e492d67 > Connect > Session Manager > Connect" -ForegroundColor White
    Write-Host ""
    Write-Host "  Run these commands:" -ForegroundColor White
    Write-Host "     $cmd1" -ForegroundColor Gray
    Write-Host "     $cmd2" -ForegroundColor Gray
    Write-Host "     docker stop hms-eeg-api; docker rm hms-eeg-api" -ForegroundColor Gray
    Write-Host "     $cmd4" -ForegroundColor Gray
    exit 1
}

Ok "CommandId: $CmdId"
Log "Waiting for completion (max 90s)..."

for ($i = 0; $i -lt 18; $i++) {
    Start-Sleep -Seconds 5
    $Status = & $AwsCli ssm get-command-invocation `
        --command-id $CmdId `
        --instance-id $InstanceId `
        --region $Region `
        --query "Status" `
        --output text 2>$null
    Write-Host "  [$([int]($i+1)*5)s] Status: $Status"
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
        $ErrOut = & $AwsCli ssm get-command-invocation `
            --command-id $CmdId `
            --instance-id $InstanceId `
            --region $Region `
            --query "StandardErrorContent" `
            --output text
        Err "Command failed:`n$ErrOut"
        exit 1
    }
}
Write-Host ""
Write-Host "  Timeout - check http://3.38.165.173:8000/ui manually" -ForegroundColor Yellow
