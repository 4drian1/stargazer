"""端到端演示入口（stargazer_plan/demo_step4.py）。

幂等构建演示链路：akshare 新浪分钟线 → 明日计划 session（is_demo=True）
→ 人工冻结 → 次日状态机监控 → 触发/失效 → 人工决策/成交 → 盘后五结局
与未执行反事实（CENSORED_TAIL 绝不填 0），全部落 stargazer_plan schema。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_db.demo import main

if __name__ == "__main__":
    main()
