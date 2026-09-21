# 快速启动：选一张图 -> 跑集成流水线 -> 打开结果
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "[!] 找不到虚拟环境 .venv，请先安装依赖（见 README）。" -ForegroundColor Red
    Read-Host "按回车退出"
    exit
}

Add-Type -AssemblyName System.Windows.Forms

Write-Host "请选择一张线稿图（默认目录 data\real）..." -ForegroundColor Cyan
$dlg = New-Object System.Windows.Forms.OpenFileDialog
$dlg.Title = "选择线稿图"
$dlg.Filter = "图片|*.jpg;*.jpeg;*.png;*.bmp;*.webp"
$dlg.InitialDirectory = Join-Path $root "data\real"
if ($dlg.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
    Write-Host "未选择文件，使用默认示例 real01.jpg" -ForegroundColor Yellow
    $img = Join-Path $root "data\real\real01.jpg"
} else {
    $img = $dlg.FileName
}

$out = Join-Path $root "demo_out"
Write-Host ""
Write-Host "处理中：$img" -ForegroundColor Green
& $py (Join-Path $root "demo.py") $img --out $out

$stem = [System.IO.Path]::GetFileNameWithoutExtension($img)
$compare = Join-Path $out "$stem`_compare.png"
$dxf = Join-Path $out "$stem`_best.dxf"

Write-Host ""
Write-Host "打开结果..." -ForegroundColor Green
if (Test-Path $compare) { Start-Process $compare }
Start-Process explorer.exe $out

Write-Host ""
Write-Host "完成！产出："
Write-Host "  DXF   : $dxf"
Write-Host "  对比图: $compare"
Read-Host "按回车关闭本窗口"
