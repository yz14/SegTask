# ============================================================================
# Plan A 升维收益对照实验 — 一键启动（Windows PowerShell）
#
# 用法（无需 conda activate；直接调用 env 内 python.exe，避免 conda 启动开销）：
#   cd d:\codes\work-projects\SegTask
#   .\experiments\lift_a\run_all.ps1
#
# 三组顺序串行执行（避免 GPU 争抢；同卡显存复用更可比）：
#   1) baseline_2_5d  — 折叠 2.5D + basic（控制组）
#   2) planA_lift     — 升维 3D + R(2+1)D（实验组）
#   3) ref_3d_zaxis   — 原生 z_axis 3D（参照组）
#
# 每组的 stdout/stderr 都重定向到 train.log（trainer 自带的 FileHandler 也会
# 同步写入 outputs/.../train.log，本脚本的 tee 是冗余备份，便于异常时定位）。
#
# 跑完后用：
#   python experiments/lift_a/aggregate_results.py
# 生成对比表。
# ============================================================================

$ErrorActionPreference = "Stop"

# 让 trainer 的 amp / cuDNN benchmark 跑出可比时间
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUNBUFFERED = "1"

# 直接锁定到 torch27_env 的 python.exe；避免 conda activate 启动开销。
$PYTHON = "D:\miniconda\envs\torch27_env\python.exe"
if (-not (Test-Path $PYTHON)) {
    Write-Error "Expected interpreter not found: $PYTHON"
    exit 1
}

$configs = @(
    "configs/experiments/lift_a_baseline_2_5d.yaml",
    "configs/experiments/lift_a_planA.yaml",
    "configs/experiments/lift_a_ref_3d.yaml"
)

$names = @("baseline_2_5d", "planA_lift_r2plus1d", "ref_3d_zaxis")

for ($i = 0; $i -lt $configs.Length; $i++) {
    $cfg  = $configs[$i]
    $name = $names[$i]
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[$($i+1)/$($configs.Length)] Launching: $name"
    Write-Host "  config: $cfg"
    Write-Host "============================================================"

    $start = Get-Date
    & $PYTHON -m segtask_v1.train --config $cfg
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Run '$name' failed with exit code $LASTEXITCODE; aborting sweep."
        exit $LASTEXITCODE
    }
    $elapsed = (Get-Date) - $start
    Write-Host "[$name] done in $($elapsed.ToString())"
}

Write-Host ""
Write-Host "All 3 runs complete. Running aggregator..."
& $PYTHON experiments\lift_a\aggregate_results.py
