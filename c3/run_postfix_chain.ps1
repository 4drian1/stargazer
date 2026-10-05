# run_postfix_chain.ps1 — 等 RELABEL_V2(PID 传入)结束后，自动跑下游步骤
param([int]$V2Pid = 11072)

$ErrorActionPreference = "Continue"
$root = "c:\Users\10757\trading\stargazer\c3_pack"
Set-Location $root

Write-Output "[chain] 等待 RELABEL_V2 PID=$V2Pid 结束 ..."
while (Get-Process -Id $V2Pid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 15
Write-Output "[chain] RELABEL_V2 已退出，校验 events_SUPERSEDED_V2_labeled.parquet 是否为 V2 ..."

$ver = & .c3venv\Scripts\python.exe -c "import pandas as pd; df=pd.read_parquet('out/events_SUPERSEDED_V2_labeled.parquet'); print(df['label_version'].iloc[0] if 'label_version' in df.columns else 'NONE')"
Write-Output "[chain] events_SUPERSEDED_V2_labeled.parquet label_version = $ver"
if ($ver -notlike "*CORVUS_LABEL_V2*") {
    Write-Output "[chain] ERROR: V2 未就绪(label_version 不是 V2)，中止下游步骤"
    exit 1
}

# ① 补 available_at_precision QA 列（已知全局属性：历史公告仅日期粒度）
Write-Output "[chain] 补 available_at_precision 列 ..."
& .c3venv\Scripts\python.exe -c "import pandas as pd; p='out/events_SUPERSEDED_V2_labeled.parquet'; df=pd.read_parquet(p); (df['available_at_precision']='DATE_ONLY') if 'available_at_precision' not in df.columns else None; df.to_parquet(p, index=False); print('available_at_precision rows=', len(df))"

# ② corp fix 影响报告
Write-Output "[chain] 运行 corp_fix_impact.py ..."
& .c3venv\Scripts\python.exe corp_fix_impact.py >> out\impact.log 2>> out\impact.err
if ($LASTEXITCODE -ne 0) { Write-Output "[chain] WARN: corp_fix_impact 退出码 $LASTEXITCODE" }

# ③ POST_FIX walk-forward（读 V2）
Write-Output "[chain] 运行 walk_forward --tag post_fix ..."
& .c3venv\Scripts\python.exe walk_forward.py --events out\event_cluster.parquet --tag post_fix --out out >> out\wf_post.log 2>> out\wf_post.err
if ($LASTEXITCODE -ne 0) { Write-Output "[chain] WARN: post_fix walk_forward 退出码 $LASTEXITCODE" }

Write-Output "[chain] POSTFIX CHAIN DONE"
