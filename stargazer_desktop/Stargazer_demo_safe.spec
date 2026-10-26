# Safe demo build: preserve external private config/data; never embed them.
from PyInstaller.utils.hooks import collect_all
import os
import akshare

datas = [('web', 'web'), ('STRATEGY.md', '.'), ('config.example.yaml', '.'),
         ('PENTACLES_LEARNING.md', '.'),
         ('assistant_graph_worker.py', '.'),
         ('on_demand_model_worker.py', '.'),
         (os.path.join(os.path.dirname(akshare.__file__), 'file_fold'), 'akshare/file_fold')]
hiddenimports = ['yaml', 'services', 'notifier', 'paths', 'ml_bridge', 'plan_bridge',
                 'intraday_monitor', 'pentacles_demo', 'pentacles_monitor', 'pentacles_journal',
                 'deepseek_api', 'assistant_workflow', 'pentacles_review_data', 'seven_day_review',
                 'assistant_context', 'chat_archive', 'model_slots', 'assistant_runtime', 'snapshot_contracts_v2',
                 'learning_review', 'trade_plan_v1', 'shadow_ui', 'shadow_journal',
                 'oracles_v1', 'jev_adapter_v1', 'decision_layer', 'index_minutes_v1',
                 'joint_risk_bridge',
                 'plan_evidence', 'plan_board_monitor',
                 'postclose_risk_runtime',
                 'stock_analysis_dialogue', 'conversation_intent',
                 'golden_rules', 'golden_monitor', 'tarot_monitor_v2', 'tarot_event_v1', 'trade_facts_v1', 'risk_outlook_v1',
                 'agent.brain', 'tools.financials', 'tools.lhb', 'tools.market', 'tools.news',
                 'tools.research', 'strategy.exits', 'strategy.funnel', 'strategy.oversold',
                 'strategy.scorer', 'info_panel', 'psycopg', 'pyarrow',
                 'akshare.stock_feature.stock_info', 'akshare.index.index_research_sw',
                 'akshare.index.index_stock_us_sina', 'akshare.futures.futures_foreign']
extra_data, binaries, extra_imports = collect_all('py_mini_racer')
datas += extra_data
hiddenimports += extra_imports
a = Analysis(['app.py'], pathex=['../stargazer_ml/v5_hybrid/opportunity_v2/contracts'], binaries=binaries, datas=datas,
             hiddenimports=hiddenimports, hookspath=[], hooksconfig={}, runtime_hooks=[],
             excludes=['langgraph', 'langchain', 'langchain_core', 'langsmith'], noarchive=False, optimize=0)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name='Stargazer', debug=False,
          bootloader_ignore_signals=False, strip=False, upx=True, upx_exclude=[],
          runtime_tmpdir=None, console=False, disable_windowed_traceback=False,
          argv_emulation=False, target_arch=None, codesign_identity=None,
          entitlements_file=None, icon=['app.ico'])
