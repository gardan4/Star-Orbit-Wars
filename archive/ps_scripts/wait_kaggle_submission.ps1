# Poll a Kaggle competition submission status and emit on state change.
# Exits when the most recent submission transitions out of PENDING.
#
# Usage:
#   powershell -NoProfile -File tools/wait_kaggle_submission.ps1 -Comp orbit-wars

param(
    [Parameter(Mandatory=$true)] [string]$Comp,
    [int]$PollSeconds = 60
)

$env:KAGGLE_API_TOKEN = (Get-Content .env | Where-Object {$_ -match 'KAGGLE_API_TOKEN='}) -replace 'KAGGLE_API_TOKEN=',''
$last = ''
while ($true) {
    $raw = .venv\Scripts\kaggle.exe competitions submissions -c $Comp 2>&1 | Out-String
    # Trap only the first data row (most recent submission) plus its header
    # for a stable one-line state summary.
    $lines = $raw -split "`r?`n" | Where-Object { $_ -match '\w' }
    # Line 2 (0-indexed 1 after header + separator) is first data row; or
    # match any row with a SubmissionStatus token.
    $row = $lines | Where-Object { $_ -match 'SubmissionStatus\.' } | Select-Object -First 1
    if (-not $row) { $row = $raw.Trim() }
    $row = $row.Trim()
    if ($row -ne $last) {
        Write-Output $row
        $last = $row
    }
    if ($row -match 'COMPLETE|ERROR|FAIL|INVALID|CANCEL') { break }
    Start-Sleep -Seconds $PollSeconds
}
