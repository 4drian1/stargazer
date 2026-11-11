import os
import sys
import yaml
import datetime as dt
import daily_tarot_v1 as T

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def get_cfg():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = get_cfg()
for d in ["2026-10-27", "2026-10-26", "2026-11-02", "2026-10-23"]:
    now = dt.datetime.strptime(d, "%Y-%m-%d").replace(hour=9, minute=0)
    card = T.draw_card(d)
    text = T.compose_daily_tarot(cfg, now)
    # 只抽取“引导语神态”那一句 + 收尾，方便对比神态是否雷同、收尾是否复读
    print("=" * 50)
    print("%s · %s （%s）" % (card["name"], card["orientation_cn"], d))
    print(text.split("\n")[0])  # 开场+神态引导句
    print("--- 收尾 ---")
    print(text.strip().split("—")[-1] if "—" in text else text[-60:])
