$Chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
$Profile = Join-Path $PSScriptRoot "chrome-local-profile"
$Extension = Join-Path $PSScriptRoot "extension"

if (-not (Test-Path $Chrome)) {
  $Chrome = "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
}
if (-not (Test-Path $Chrome)) {
  throw "Chrome 실행 파일을 찾을 수 없습니다."
}

New-Item -ItemType Directory -Path $Profile -Force | Out-Null
Start-Process -FilePath $Chrome -ArgumentList @(
  "--user-data-dir=$Profile",
  "--load-extension=$Extension",
  "--disable-extensions-except=$Extension",
  "--no-first-run",
  "--new-window",
  "http://localhost:8000/demo/browse"
)
