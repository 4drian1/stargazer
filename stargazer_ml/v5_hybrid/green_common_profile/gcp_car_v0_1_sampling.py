"""GREEN-CAR-v0.1 R0: 候选股票日确定性抽样 (合同 §2, 路线记录#11.4).

状态: DRAFT_FOR_PRE_REGISTRATION — 本脚本只读本地文件 (池成员 parquet + 日线镜像 parquet
+ 已有分钟缓存), 0 HTTP 请求。输出为 run sheet 草案的登记材料, 待用户审定签署;
签署前不抓取、不训练、不生成任何实验结果。

抽样程序 (提案 P-SAMP, 结果前冻结):
  1. 合格宇宙: {(code, E, setup)} — (code, P, setup) ∈ 冻结池成员
     (pool_a_sdp_members / pool_b_lvb_members, date 列=P 资格日),
     E = P 的下一交易所交易日, E ∈ [20250102, 20260917] (分钟纪元窗)。
     北交所 (4/8/92 开头, 无分钟) 显式排除并计数; 主板/创业板/科创板分别标记。
  2. 股票抽样: 每 setup 内 code 升序, RNG(20260929) 无放回抽 10 股 → ≤20 股。
     RNG 调用顺序固定: 先 A 后 B, 先股票后 episode (可复现)。
  3. 日抽样: 每股合格 E 日升序 → 划分 episode (相邻 E 间隔 >2 交易日开新 episode,
     提案 P-EP) → 无放回抽 ≤5 个 episode, 取每 episode 首日 → 每股 ≤5 候选日
     (≤合同上限10), A/B 各 ≤50 日 (合同上限), 总计 ≤100 候选股票日。
     配额未满如实登记, 不凑 n、不重抽。
  4. episode/candidate 编号: candidate_episode_id=(code,setup,episode_idx);
     event_group_id 在事件层 (同 symbol+E 多锚点/多周期同组), 由后续事件提取赋值。

预算核算 (合同 §7): 逐候选登记 E 日分钟缓存命中; R 组同刻历史基线按每股 ≤60 交易日
统计缓存命中/缺失 → 估算新增请求股日, 对照 ≤2000 上限 (仅核算, 不请求)。

输出 (contracts/):
  - car_v0_1_candidate_list.txt   行="code,setup,E" 字典序; 字节SHA256 与
    规范化(UTF-8/LF/末尾换行)SHA256 分别登记, 不混用
  - car_v0_1_sampling_summary.json
运行: python gcp_car_v0_1_sampling.py
"""
from __future__ import annotations

import bisect
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "step9_preflight"))
from data_source_provenance import get_daily_parquet  # noqa: E402

OUT = HERE / "contracts"
SNAP_OLD = HERE / "gcp_round_b_small_sample_v1" / "minute"
SNAP_EXT = HERE / "gcp_round_b_small_sample_v1" / "minute_ext"

SEED = 20260929                    # 合同 §2.4 提案, 待确认
E_START, E_END = 20250102, 20260917   # 分钟纪元窗 (run_sheet minute_window)
N_STOCKS_PER_SETUP = 10            # 合同 §2.5 上限 20 股
MAX_EPISODES_PER_STOCK = 5         # 提案: 每股 ≤5 episode 首日 (合同每股≤10日)
MAX_DAYS_PER_SETUP = 50            # 合同 §2.5 A/B 各 50 日上限
MAX_TOTAL = 100                    # 合同 §2.5 总上限
HIST_BASELINE_DAYS = 60            # 合同 §7: 历史输入每股 ≤60 交易日 (R组基线核算用)
EPISODE_GAP_TD = 2                 # 提案 P-EP: 相邻 E 间隔 >2 交易日开新 episode


def board_tag(code: str) -> str:
    c = str(code).zfill(6)
    if c[0] in "48" or c.startswith("92"):
        return "BSE"                 # 北交所: 无分钟, 显式排除
    if c.startswith("688"):
        return "STAR"
    if c.startswith("30"):
        return "CHINEXT"
    return "MAIN"


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = get_daily_parquet()
    cal = sorted(int(d) for d in panel["date"].unique())

    # ---- 1. 合格宇宙 ----
    pool = pd.concat([
        pd.read_parquet(HERE / "round_a_v0_1_impl_fix" / "pool_a_sdp_members.parquet")[
            ["date", "code", "setup_type"]],
        pd.read_parquet(HERE / "round_a_v0_1_impl_fix" / "pool_b_lvb_members.parquet")[
            ["date", "code", "setup_type"]],
    ])
    pool["code"] = pool["code"].astype(str).str.zfill(6)
    pool["date"] = pool["date"].astype(int)
    pool = pool.drop_duplicates(["date", "code", "setup_type"])

    cal_idx = {d: i for i, d in enumerate(cal)}
    elig = []
    bse_stock_days = 0          # 北交所排除: 股票日数 (逐池记录计数)
    bse_unique_codes = set()    # 北交所排除: 去重股票数
    for p, code, setup in pool.itertuples(index=False):
        bt = board_tag(code)
        if bt == "BSE":
            bse_stock_days += 1
            bse_unique_codes.add(code)
            continue
        # P 收盘合格 → E=下一交易所交易日; P 须为日历交易日
        i = cal_idx.get(p)
        if i is None or i + 1 >= len(cal):
            continue
        e = cal[i + 1]
        if not (E_START <= e <= E_END):
            continue
        elig.append({"code": code, "setup": setup, "P": p, "E": e, "board": bt})
    edf = pd.DataFrame(elig).drop_duplicates(["code", "setup", "E"])
    print(f"[合格宇宙] 候选(code,E,setup)={len(edf)}  "
          f"股数={edf['code'].nunique()}  "
          f"北交所排除: 股票日={bse_stock_days} 去重股数={len(bse_unique_codes)}")
    print(f"  板块分布: {edf['board'].value_counts().to_dict()}")

    # ---- 2/3. 确定性抽样 (RNG 顺序固定: A 股票→A 各股 episode→B 股票→B 各股 episode) ----
    # 注: 这是"先抽股票, 再抽 episode 首日"的工程抽样, 不是候选股票日等概率抽样;
    # 用于机制/因果/数据分流验收, 不作为统计研究阶段的推断抽样。
    # "相邻日期不独立"只用于 episode 分组, 不代表事件已独立; 后续统计应另定抽样与聚类规则。
    rng = np.random.default_rng(SEED)
    selected = []
    for setup in ("A_SDP", "B_LVB"):
        sub = edf[edf["setup"] == setup]
        codes = sorted(sub["code"].unique())
        pick_n = min(N_STOCKS_PER_SETUP, len(codes))
        picked = sorted(rng.choice(codes, size=pick_n, replace=False).tolist())
        for code in picked:
            days = sorted(sub[sub["code"] == code]["E"].tolist())
            # episode 划分: 相邻 E 间隔 >EPISODE_GAP_TD 个交易日 → 新 episode
            # (P-EP: 间隔>2日仅用于分组, 不代表事件已独立)
            episodes, cur = [], [days[0]]
            for a, b in zip(days, days[1:]):
                if bisect.bisect_left(cal, b) - bisect.bisect_left(cal, a) > EPISODE_GAP_TD:
                    episodes.append(cur)
                    cur = [b]
                else:
                    cur.append(b)
            episodes.append(cur)
            ep_n = min(MAX_EPISODES_PER_STOCK, len(episodes))
            ep_pick = sorted(rng.choice(len(episodes), size=ep_n, replace=False).tolist())
            for k in ep_pick:
                selected.append({"code": code, "setup": setup, "board": board_tag(code),
                                 "E": episodes[k][0], "P": cal[cal_idx[episodes[k][0]] - 1],
                                 "episode_first_day": episodes[k][0],
                                 "episode_len": len(episodes[k]),
                                 "candidate_episode_id": f"{code}|{setup}|ep{k:03d}",
                                 "n_episodes_stock": len(episodes),
                                 "n_eligible_days_stock": len(days)})
    sdf = pd.DataFrame(selected)
    # A/B 各 50 日上限 (合同); 抽样超限时按 E 升序截断并登记
    truncated = {}
    keep = []
    for setup in ("A_SDP", "B_LVB"):
        s = sdf[sdf["setup"] == setup].sort_values(["code", "E"])
        if len(s) > MAX_DAYS_PER_SETUP:
            truncated[setup] = len(s) - MAX_DAYS_PER_SETUP
            s = s.head(MAX_DAYS_PER_SETUP)
        keep.append(s)
    sdf = pd.concat(keep).reset_index(drop=True)
    print(f"[抽样] 股数={sdf['code'].nunique()} (A={sdf[sdf['setup']=='A_SDP']['code'].nunique()} "
          f"B={sdf[sdf['setup']=='B_LVB']['code'].nunique()})  "
          f"候选股票日={len(sdf)} (A={int((sdf['setup']=='A_SDP').sum())} "
          f"B={int((sdf['setup']=='B_LVB').sum())})  截断={truncated}")
    print(f"  每股候选日: min={sdf.groupby('code').size().min()} "
          f"max={sdf.groupby('code').size().max()}")

    # ---- 4. 缓存核算 (去重 (code,date) 集合; 拆四类请求; 只读不请求) ----
    cache_days = {}
    for d in (SNAP_OLD, SNAP_EXT):
        if not d.exists():
            continue
        for pf in d.glob("*.parquet"):
            code = pf.stem
            m = pd.read_parquet(pf, columns=["date"])
            days = set(m["date"].astype(str).str[:8].astype(int).unique().tolist())
            cache_days.setdefault(code, set()).update(days)

    # 按请求类别生成 (code,date) 集合, 去重后扣缓存
    # 类1 特征输入(E日截至S前缀) + 类2 标签输入(未来窗, 60m/持续性)
    #   → 两类都落在 E 日 (下午信号未来窗可能跨到下一交易日, 持续性标签更长)
    #   → 合并为 "E日 + 未来窗可能触及的交易日" 集合, 去重
    # 类3 R组同刻历史基线 (≤60交易日/股, 独立增量消融, 默认本轮不作)
    # 类4 复权链 (每股一次, 独立于分钟)
    adj_cache_path = HERE / "gcp_round_b_small_sample_v1" / "adj_cache.json"
    adj_cache_existing = set()
    if adj_cache_path.exists():
        adj_cache_existing = set(json.loads(adj_cache_path.read_text(encoding="utf-8")).keys())
    req_feature_label = set()   # (code, date): E日 + 未来窗触及日
    req_hist_baseline = set()  # (code, date): R组历史基线 (本轮默认不作, 仅核算)
    req_adj = set()            # code: 复权链请求
    for r in sdf.itertuples(index=False):
        have = cache_days.get(r.code, set())
        i_e = cal_idx[r.E]
        # 类1+2 合并: E 日 + 未来 60 交易分钟可能跨到的下一交易日
        #   下午信号 (≥13:00) 的 60 交易分钟窗可能跨到 E+1 日;
        #   持续性标签(收回后保持10交易分钟)在 60m 窗内则不延长, 跨日则可能到 E+1
        #   保守取 E 和 E+1 两个交易日 (if exists)
        future_days = {r.E}
        if i_e + 1 < len(cal):
            future_days.add(cal[i_e + 1])
        for d in future_days:
            if d not in have:
                req_feature_label.add((r.code, d))
        # 类3 R组历史基线 (≤60 交易日, 去重)
        for d in cal[max(0, i_e - HIST_BASELINE_DAYS):i_e]:
            if d not in have:
                req_hist_baseline.add((r.code, d))
        # 类4 复权链 (每股一次)
        if r.code not in adj_cache_existing:
            req_adj.add(r.code)

    n_req_feat = len(req_feature_label)
    n_req_hist = len(req_hist_baseline)
    n_req_adj = len(req_adj)
    n_req_total_noR = n_req_feat + n_req_adj
    n_req_total_withR = n_req_feat + n_req_hist + n_req_adj
    print(f"[缓存核算·去重后] (code,date) 集合:")
    print(f"  类1+2 特征+标签分钟 (E日+未来窗触及日, 去重): {n_req_feat}")
    print(f"  类3   R组历史基线 (≤60日/股, 去重, 本轮默认不作): {n_req_hist}")
    print(f"  类4   复权链 (每股一次): {n_req_adj}")
    print(f"  合计 (不含R组, 本轮默认): {n_req_total_noR}")
    print(f"  合计 (含R组, 仅供参考):  {n_req_total_withR} (合同上限 2000; 仅核算, 未请求)")
    print(f"  注: vwap_distance_atr 只需 E 日截至 S 的量额前缀, 不需要历史分钟; "
          f"历史同刻量比/额比后置 (R组消融)。")

    # ---- 5. 落盘: 候选清单 (双哈希) + 摘要 ----
    lines = sorted(f"{r.code},{r.setup},{r.E}" for r in sdf.itertuples(index=False))
    text = "\n".join(lines) + "\n"
    list_f = OUT / "car_v0_1_candidate_list.txt"
    list_f.write_bytes(text.encode("utf-8"))          # LF (write_bytes 不做换行转换)
    byte_sha = sha256_hex(list_f.read_bytes())
    norm_sha = sha256_hex(text.encode("utf-8"))        # 规范化 = 本文件本身即 LF+末尾换行

    def fsha(p: Path) -> str:
        return sha256_hex(p.read_bytes())

    summary = {
        "status": "DRAFT_FOR_PRE_REGISTRATION — 待用户审定签署; 签署前不抓取不训练",
        "date": "2026-09-29",
        "seed": SEED,
        "user_rulings": [
            "先抽股票再抽episode首日=工程抽样, 不是候选股票日等概率抽样; 用于机制/因果/数据分流验收",
            "相邻日期不独立仅用于episode分组, 不代表事件已独立; 后续统计研究另定抽样与聚类规则",
            "33日不足主要由股票抽样后资格日期稀疏造成, 非长连续episode归并; 本轮保留33日不重抽",
            "P-BUDGET: 本轮不做历史同刻量比消融, 不缩短历史不提高上限; 先修正预算核算(去重后)",
            "P-END: 候选截止20260917; 标签可用数据截止另列, 窗口未走完就截断",
            "P-SLOPE: 5交易日变化量公式, 注明不是回归斜率",
            "P-HOLD: 收回后保持10交易分钟, 仅辅助标签, 不能救主标签",
            "P-LAMBDA/P-L2: λ=20, C=1可登记固定默认值, 但本轮不拟合不调参",
            "P-EMB: 15交易日可作附加隔离; 真正边界必须由逐行label_end决定",
            "P-BLK/P-GATE: 留到统计阶段签署, 33日不计算晋级结论",
            "P-DATA: 接受补必要复权链; 失败保持UNKNOWN不当空链",
            "承接组vwap_distance_atr保留为预注册候选(只需E日截至S前缀); 历史同刻量比/额比后置",
            "区域绝对价格(zone_lower/upper/mid)只保留审计表不进LR; 用ATR标准化距离/宽度/来源/合流数",
            "穿透深度÷已观察分钟=平均穿透速率, 不是真正的分段下杀速度",
            "本轮暂不训练CAR-G/CAR-LR: 拟合段9日/开发段8日/诊断段15日, 只验证计算/因果/分流不验证效果",
            "成功标准: 每个事件'什么时候/接近哪条锚点/当时是否已收回/之后是否收回'算对可重放, 非跑出盈利",
        ],
        "sampling_program": {
            "universe": "(code,P,setup)∈冻结池; E=P下一交易日; E∈[20250102,20260917]",
            "episode_rule": f"相邻E间隔>{EPISODE_GAP_TD}交易日开新episode (提案P-EP)",
            "stock_pick": f"每setup code升序RNG无放回抽{N_STOCKS_PER_SETUP}股",
            "day_pick": f"每股无放回抽≤{MAX_EPISODES_PER_STOCK}个episode, 取episode首日",
            "rng_call_order": "default_rng(20260929); A股票→A逐股episode→B股票→B逐股episode",
            "caps": {"stocks": 20, "days_per_setup": 50, "days_per_stock": 10,
                     "total": 100},
        },
        "universe_stats": {
            "n_candidates": int(len(edf)),
            "n_stocks": int(edf["code"].nunique()),
            "board_distribution": edf["board"].value_counts().to_dict(),
            "bse_excluded_stock_days": bse_stock_days,
            "bse_excluded_unique_codes": len(bse_unique_codes),
            "bse_note": "北交所逐池记录计数: 股票日数 + 去重股票数分别登记",
            "per_setup": {s: int((edf["setup"] == s).sum()) for s in ("A_SDP", "B_LVB")},
        },
        "selected_stats": {
            "n_stocks": int(sdf["code"].nunique()),
            "n_candidates": int(len(sdf)),
            "per_setup_days": {s: int((sdf["setup"] == s).sum()) for s in ("A_SDP", "B_LVB")},
            "per_setup_stocks": {s: int(sdf[sdf["setup"] == s]["code"].nunique())
                                 for s in ("A_SDP", "B_LVB")},
            "quota_truncated": truncated,
            "per_stock_days": sdf.groupby("code").size().to_dict(),
            "episode_len_stats": [int(sdf["episode_len"].min()), float(sdf["episode_len"].mean()),
                                  int(sdf["episode_len"].max())],
        },
        "cache_accounting": {
            "method": "去重 (code,date) 集合后扣缓存; 旧版按候选累计会重复计数",
            "class_1_2_feature_label_minute": n_req_feat,
            "class_1_2_note": "E日 + 未来60交易分钟可能跨到的下一交易日 (下午信号); 持续性标签在60m窗内不延长",
            "class_3_hist_baseline": n_req_hist,
            "class_3_note": "R组同刻历史基线 ≤60交易日/股; 独立增量消融; 本轮默认不作",
            "class_4_adj_chain": n_req_adj,
            "class_4_note": "复权链每股一次; 失败保持UNKNOWN不当空链",
            "total_without_R": n_req_total_noR,
            "total_with_R_for_reference": n_req_total_withR,
            "budget_cap": 2000,
            "vwap_note": "vwap_distance_atr 只需 E日截至S的量额前缀, 不需历史分钟; 历史同刻量比/额比后置",
        },
        "hashes": {
            "candidate_list_byte_sha256": byte_sha,
            "candidate_list_canonical_sha256": norm_sha,
            "canonical_note": "规范化=UTF-8/LF/末尾换行; 与字节哈希分别登记, 不混用",
            "gcp_v0_1_spec.py": fsha(HERE / "gcp_v0_1_spec.py"),
            "gcp_v0_1_pools.py": fsha(HERE / "gcp_v0_1_pools.py"),
            "gcp_v0_1_round_a.py": fsha(HERE / "gcp_v0_1_round_a.py"),
            "gcp_v0_1_round_b_spec.py": fsha(HERE / "gcp_v0_1_round_b_spec.py"),
            "gcp_round_b_engine.py": fsha(HERE / "gcp_round_b_engine.py"),
            "pool_a_sdp_members.parquet": fsha(
                HERE / "round_a_v0_1_impl_fix" / "pool_a_sdp_members.parquet"),
            "pool_b_lvb_members.parquet": fsha(
                HERE / "round_a_v0_1_impl_fix" / "pool_b_lvb_members.parquet"),
            "contract_conditional_anchor_reaction_v0_1.md": fsha(
                HERE / "contracts" / "conditional_anchor_reaction_v0_1.md"),
            "本脚本_gcp_car_v0_1_sampling.py": fsha(Path(__file__).resolve()),
        },
        "selected": sdf[["code", "setup", "board", "P", "E", "candidate_episode_id",
                         "episode_len", "n_episodes_stock", "n_eligible_days_stock"]].to_dict(
            orient="records"),
    }
    sum_f = OUT / "car_v0_1_sampling_summary.json"
    sum_f.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                     encoding="utf-8")
    print(f"\n[落盘] {list_f}")
    print(f"       字节SHA256={byte_sha}")
    print(f"       规范化SHA256={norm_sha}")
    print(f"[落盘] {sum_f}")


if __name__ == "__main__":
    main()
