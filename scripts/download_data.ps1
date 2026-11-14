# 下載 FB15k-237（標準切分）、實體名稱、TKRL 實體類型
# 用法（在專案根目錄）：powershell -ExecutionPolicy Bypass -File scripts\download_data.ps1
$ErrorActionPreference = "Stop"
$fb = "data\FB15k-237"
New-Item -ItemType Directory -Force $fb | Out-Null
$base = "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237"
foreach ($f in "train.txt", "valid.txt", "test.txt") {
    Invoke-WebRequest "$base/$f" -OutFile "$fb\$f"
}
# 實體名稱（MID -> 名稱），來自 KG-BERT repo
Invoke-WebRequest "https://raw.githubusercontent.com/yao8839836/kg-bert/master/data/FB15k-237/entity2text.txt" -OutFile "$fb\entity2text.txt"

# TKRL (Xie et al., 2016) 的實體類型
$tk = "data\tkrl_raw"
New-Item -ItemType Directory -Force "$tk\v2" | Out-Null
Invoke-WebRequest "https://github.com/thunlp/TKRL/raw/master/data.zip" -OutFile "$tk\data.zip"
Expand-Archive "$tk\data.zip" -DestinationPath $tk -Force
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead((Resolve-Path "$tk\data\TKRL-all-input-v2.zip"))
foreach ($e in $zip.Entries) {
    if ($e.Name -in "entity2type.txt", "type2domain.txt", "domain2id.txt") {
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($e, "$tk\v2\$($e.Name)", $true)
    }
}
$zip.Dispose()
Write-Host "完成。data\FB15k-237 應有 train 272115 / valid 17535 / test 20466 行。"
