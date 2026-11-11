# Safe demo build: preserve external private config/data; never embed them.
from PyInstaller.utils.hooks import collect_all
import os
import runpy
import akshare

_RELEASE_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(SPEC)))
_release=runpy.run_path(os.path.join(_RELEASE_ROOT,'release_audit_v1','verify_live_release.py'))['verify_release'](_RELEASE_ROOT)
if not _release['ok']:raise RuntimeError('RELEASE_PREFLIGHT_FAILED:'+repr(_release['errors']))

datas = [('web', 'web'), ('STRATEGY.md', '.'), ('config.example.yaml', '.'),
         ('PENTACLES_LEARNING.md', '.'),
         ('assistant_graph_worker.py', '.'), ('minute_history_v1.py', '.'),
         ('on_demand_model_worker.py', '.'),
         ('golden_stock_analysis.py', '.'), ('minute_plan_review.py', '.'),
         ('sector_rule_scan.py', '.'), ('strong_sector_watch_v1.py', '.'),
         ('persistent_records_v1.py', '.'),
         ('cross_process_lock_v1.py', '.'), ('instance_state_store_v1.py', '.'),
         (os.path.join(os.path.dirname(akshare.__file__), 'file_fold'), 'akshare/file_fold')]
hiddenimports = [
    'ui_calendar_api', 'ui_plan_versions_api', 'ui_global_overview_api',
    'ui_backend_config', 'pentacles_review_data',
    'acceptance_ui_integration_v1', 'acceptance_ui2_serve_v1','yaml', 'services', 'notifier', 'paths', 'ml_bridge', 'plan_bridge', 'sector_rule_scan',
                 'intraday_monitor', 'pentacles_demo', 'pentacles_monitor', 'pentacles_journal', 'golden_watch_groups',
                 'deepseek_api', 'assistant_workflow', 'pentacles_review_data', 'seven_day_review',
                 'assistant_context', 'chat_archive', 'model_slots', 'assistant_runtime', 'snapshot_contracts_v2',
                 'learning_review', 'trade_plan_v1', 'shadow_ui', 'shadow_journal',
                 'oracles_v1', 'jev_adapter_v1', 'decision_layer', 'index_minutes_v1',
                 'joint_risk_bridge',
                 'plan_evidence', 'plan_board_monitor',
                 'postclose_risk_runtime',
                 'postclose_prepared_v1', 'storage_roots_v1', 'asset_migrate_v1',
                 'runtime_disk_limit_v1',
                 'stock_analysis_dialogue', 'conversation_intent',
                 'golden_rules', 'golden_monitor', 'tarot_monitor_v2', 'tarot_event_v1', 'trade_facts_v1', 'risk_outlook_v1',
                 'push_layout_v1','push_presenter_v1','strong_wave_facts_v1','strong_sector_rules_v1','strong_low_indicators_v1','strong_sector_watch_v1',
                 'strong_touch_lane_v1','quote_budget_v1',
                 'agent.brain', 'golden_stock_analysis', 'minute_plan_review', 'conversation_plans', 'plan_rule_profiles', 'tools.financials', 'tools.lhb', 'tools.market', 'tools.news',
                 'tools.research', 'strategy.exits', 'strategy.funnel', 'strategy.oversold',
                 'strategy.scorer', 'info_panel', 'psycopg', 'pyarrow',
                 'akshare.stock_feature.stock_info', 'akshare.index.index_research_sw',
                 'akshare.index.index_stock_us_sina', 'akshare.futures.futures_foreign']
extra_data, binaries, extra_imports = collect_all('py_mini_racer')
datas += extra_data
hiddenimports += extra_imports

# ---- I34 announcement-importance demo (frozen I33, zero-fit inference) ----
# Whitelist ONLY what the runtime reads. Never pull in the whole impact_v1_1 tree,
# and never embed private config.yaml / data (that is what the safe build is for).
try:
    _SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))   # injected by PyInstaller
except NameError:
    _SPEC_DIR = os.getcwd()
_CORVUS = os.path.join(os.path.dirname(_SPEC_DIR), 'c3', 'corvus', 'impact_v1_1')

# entry package (keep in sync with the file list in c3/.../i34_demo/i34_integration_test.py)
_I34_DEMO_FILES = ['i34_infer.py', 'i34_desktop_integration.py', 'i34_vocab.json',
                   'i34_receipt_index.json', 'i34_deepseek_input.json',
                   'i34_stock_linkage_v1.py']
datas += [(os.path.join(_CORVUS, 'i34_demo', _f), 'i34_demo') for _f in _I34_DEMO_FILES]

# frozen research assets actually used at runtime. i24_1 is here because
# i25/extractor_v3.py dynamically loads ../i24_1_.../extractor_v2.py.
for _d in ('i25_quantity_repair_fixed_fit_20261029_v1',
           'i24_1_zero_fit_quantity_contract_20261029_v1',
           'i28_time_clause_role_repair_20261029_v1',
           'i29_bound_clause_time_encoding_20261029_v1',
           'i33_clause_role_fit_20261031_v1'):
    # bundled flat at the bundle root: i34_infer derives paths from its own location
    datas.append((os.path.join(_CORVUS, _d), _d))

datas += [('ths_board_frozen_members_v1.json', '.')]
datas += [('ths_industry_source_seed_v1.json.gz', '.')]
hiddenimports += ['cross_process_lock_v1', 'instance_guard_v1', 'instance_state_store_v1',
                  'plan_board_change_v1', 'ths_board_identity_v1', 'ths_board_supply_v1', 'monitor_release_acceptance_v1',
                  'strong_sector_pool_prep_v1', 'daily_tarot_v1']
hiddenimports += ['i34_bridge', 'self_check_realtime_v1']
hiddenimports += ['strong_pullback_policy_v1','strong_pullback_context_v1','strong_pullback_boards_v1']
hiddenimports += ['strong_pullback_positions_v1']
hiddenimports += ['persistent_records_v1']

a = Analysis(['app.py'], pathex=['../stargazer_ml/v5_hybrid/opportunity_v2/contracts',
             '../stargazer_ml/v5_hybrid/risk_r0_v1',
             os.path.join(_CORVUS, 'i34_demo')], binaries=binaries, datas=datas,
             hiddenimports=hiddenimports, hookspath=[], hooksconfig={}, runtime_hooks=[],
             excludes=['langgraph', 'langchain', 'langchain_core', 'langsmith',
                       'torch', 'torchvision', 'transformers', 'tensorflow',
                       'matplotlib', 'IPython', 'jupyter', 'notebook'], noarchive=False, optimize=0)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name='Stargazer', debug=False,
          bootloader_ignore_signals=False, strip=False, upx=True, upx_exclude=[],
          runtime_tmpdir=None, console=False, disable_windowed_traceback=False,
          argv_emulation=False, target_arch=None, codesign_identity=None,
          entitlements_file=None, icon=['app.ico'])
