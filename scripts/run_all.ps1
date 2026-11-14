# 一鍵跑完整流程：訓練 + 評估 + 非監督分析
# 用法（在專案根目錄）：powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
# 加 -Quick 先小規模試跑：powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Quick
param([switch]$Quick)
$env:PYTHONIOENCODING = "utf-8"
$py = ".venv\Scripts\python.exe"
if ($Quick) {
    & $py -m src.tune --quick
    if ($?) { & $py -m src.analysis --results results\quick --k-min 4 --k-max 12 --k-step 4 }
} else {
    & $py -m src.tune
    if ($?) { & $py -m src.analysis }
}
