"""plan_db 包：第 4 步最小交易计划数据库（规格 plan_db_v1.md）。

模块：
- db       PostgreSQL 连接（复用 corvus/config/pg.yaml）
- ddl      幂等 DDL：9 张表 + trigger_event + 冻结/防回填触发器
- dsl      确定性规则 DSL v1 解析与求值（纯函数）
- machine  次日状态机：分钟重放、inputs_hash、trigger_event 落库
- daily    个股日线读取（D:/a股数据/data_daily_parts，CENSORED_TAIL 口径）
- outcome  五种结局分类 + 未执行反事实
- repo     幂等写入层（ON CONFLICT 幂等锚）
- demo     akshare 新浪分钟线端到端演示（is_demo=True）
"""
from .db import get_pg_conn, pg_dsn                       # noqa: F401
from .ddl import ensure_schema                            # noqa: F401
from . import dsl, machine, outcome, repo, daily, demo    # noqa: F401
