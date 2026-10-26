"""打包监控机资产包 stargazer_assets_v1.0.0-rc1.zip + sha256manifest.txt（路线 #18.46）。

内容 = 发布模型 + 必需小资产 + 相对键指纹清单（项目相对路径布局，解压到项目根即可）。
注意：这些文件已随 git tag 入库，资产包是 Release 上的便利/校验 bundle。
"""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'release_audit_v1' / 'release'
OUT_DIR.mkdir(parents=True, exist_ok=True)
ZIP = OUT_DIR / 'stargazer_assets_v1.0.0-rc1.zip'

ASSETS = []
for name in ('live_monitor_fingerprints_20261026_v27_rel.json',
             'joint_shadow_fingerprints_rel_v1.json'):
    ASSETS.append(f'stargazer_ml/v5_hybrid/risk_r0_v1/{name}')
ASSETS += [
    'stargazer_ml/v5_hybrid/green_common_profile/tarot_live_manifest_v18_20261026_rel.json.gz',
    'stargazer_ml/v5_hybrid/green_common_profile/round_a_v0_1_impl_fix/fold_model_F3_B2.pkl',
    'stargazer_ml/v5_hybrid/green_common_profile/round_a_v0_1_impl_fix/fold_model_FINAL_B2.pkl',
    'stargazer_ml/v5_hybrid/green_common_profile/round_a_v0_1_impl_fix/full_master_table_v2.py',
    'stargazer_ml/v5_hybrid/green_common_profile/round_a_v0_1_impl_fix/full_master_table_v2.parquet',
    'stargazer_ml/v5_hybrid/green_common_profile/round_a_v0_1_impl_fix/full_intraday_replay_v2.parquet',
    'stargazer_ml/v5_hybrid/green_common_profile/round_a_v0_1_impl_fix/rolling_lr_v2_predictions.parquet',
    'stargazer_ml/v5_hybrid/green_common_profile/tarot_a_second_leg_fit_20261022_v1/run.py',
    'stargazer_ml/v5_hybrid/green_common_profile/tarot_a_second_leg_fit_20261022_v1/initial_node_cash_fit_20261023_v1/model.pkl',
    'stargazer_ml/v5_hybrid/green_common_profile/tarot_a_second_leg_fit_20261022_v1/execution_alignment_diagnostic_20261022_v1/diagnose_execution.py',
    'stargazer_ml/v5_hybrid/green_common_profile/tarot_a_second_leg_fit_20261022_v1/label_alignment_diagnostic_20261022_v1/diagnose.py',
    'stargazer_ml/v5_hybrid/green_common_profile/tarot_a_second_leg_fit_20261022_v1/stockdb_sdk_adapter_diagnostic_20261022_v1/adapter.py',
    'stargazer_ml/v5_hybrid/green_common_profile/gcp_round_b_small_sample_v1/minute_grid.json',
    'stargazer_ml/v5_hybrid/green_common_profile/gcp_round_b_small_sample_v1/pentacles_golden_v1/trigger_replay_v3.py',
    'stargazer_ml/v5_hybrid/opportunity_v2/o5_sector_lag/_upstream/stockdb_boards_snapshot.parquet',
    'stargazer_ml/v5_hybrid/opportunity_v2/o5_sector_lag/pit_membership.parquet',
    'stargazer_ml/v5_hybrid/risk_r0_v1/relative_stall_5m_20261018_v1/raw_input_ledger.json',
    'stargazer_ml/v5_hybrid/risk_r0_v1/ths_full_concept_scope_20261017_v1.json',
    'stargazer_ml/v5_hybrid/step9_preflight/trading_calendar.py',
    'stargazer_ml/pullback_train/quarantine_2026.csv',
]
for d in ('auxiliary_120m_fixed_fit_20261018_v1', 'common_candidate_fixed_fit_20261019_v1'):
    for f in ('lightgbm_model.txt.gz', 'lightgbm_preprocessing.json', 'manifest.json', 'plan.json'):
        ASSETS.append(f'stargazer_ml/v5_hybrid/risk_r0_v1/{d}/{f}')
# 部署文档与自检脚本一并入包
ASSETS += ['release_audit_v1/DEPLOY_MONITOR.md',
           'release_audit_v1/requirements-monitor.txt',
           'release_audit_v1/verify_monitor_env.py']

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

missing = [a for a in ASSETS if not (ROOT / a).exists()]
assert not missing, missing
manifest = {}
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for rel in ASSETS:
        z.write(ROOT / rel, rel)
        manifest[rel] = sha256(ROOT / rel)
(ZIP.parent / 'sha256manifest.txt').write_text(
    ''.join(f'{v}  {k}\n' for k, v in sorted(manifest.items())), encoding='utf-8')
print('assets:', len(ASSETS), 'zip:', ZIP, ZIP.stat().st_size, 'bytes')
