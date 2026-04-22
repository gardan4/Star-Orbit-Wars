# Poll a Kaggle kernel status and emit on state change. Exits when the
# run reaches a terminal state (COMPLETE / ERROR / CANCEL / INVALID /
# FAIL).
#
# Usage:
#   powershell -NoProfile -File tools\wait_kaggle_kernel.ps1 -Slug gardan4/orbit-wars-mcts-v2

param(
    [Parameter(Mandatory=$true)] [string]$Slug,
    [int]$PollSeconds = 20
)

$env:KAGGLE_API_TOKEN = (Get-Content .env | Where-Object {$_ -match 'KAGGLE_API_TOKEN='}) -replace 'KAGGLE_API_TOKEN=',''
$last = ''
while ($true) {
    $now = .venv\Scripts\kaggle.exe kernels status $Slug 2>&1 | Out-String
    $now = $now.Trim()
    if ($now -ne $last) {
        Write-Output $now
        $last = $now
    }
    if ($now -match 'COMPLETE|ERROR|CANCEL|INVALID|FAIL') { break }
    Start-Sleep -Seconds $PollSeconds
}
