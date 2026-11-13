"""更新 8 处运行时指纹清单引用（v176->v177 flat, v164->v165 tarot）。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FLAT_OLD = 'live_monitor_fingerprints_20261108_v176_flow.json.gz'
FLAT_NEW = 'live_monitor_fingerprints_20261108_v177_flow.json.gz'
TAROT_OLD = 'tarot_live_manifest_v164_20261108_rel.json.gz'
TAROT_NEW = 'tarot_live_manifest_v165_20261108_rel.json.gz'

FLAT_REFS = [
    ROOT / 'stargazer_desktop/golden_stock_analysis.py',
    ROOT / 'stargazer_desktop/sector_rule_scan.py',
    ROOT / 'stargazer_desktop/on_demand_model_worker.py',
    ROOT / 'stargazer_ml/v5_hybrid/green_common_profile/market_price_evidence_v1.py',
    ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1/tarot_intraday_risk_v2.py',
    ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1/accept_live_postclose_v1.py',
    ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1/live_postclose_job_v1.py',
]
TAROT_REFS = [
    ROOT / 'stargazer_ml/v5_hybrid/green_common_profile/tarot_live_v2.py',
]

for f in FLAT_REFS:
    txt = f.read_text(encoding='utf-8')
    n = txt.count(FLAT_OLD)
    assert n == 1, f'{f}: 期望 1 次 {FLAT_OLD}，实际 {n}'
    f.write_text(txt.replace(FLAT_OLD, FLAT_NEW), encoding='utf-8')
    print('updated flat ref:', f.relative_to(ROOT).as_posix())

for f in TAROT_REFS:
    txt = f.read_text(encoding='utf-8')
    n = txt.count(TAROT_OLD)
    assert n == 1, f'{f}: 期望 1 次 {TAROT_OLD}，实际 {n}'
    f.write_text(txt.replace(TAROT_OLD, TAROT_NEW), encoding='utf-8')
    print('updated tarot ref:', f.relative_to(ROOT).as_posix())

print('done: 8 references updated to v177/v165.')
