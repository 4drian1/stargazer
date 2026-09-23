"""B1 后续诊断 (纯诊断, 不训练):
v2 trigger 样本的 edge 是否在"等待确认"期间被价格吃掉?

对每个 scored 样本, 用 RAW 口径一致计算 (日内常数复权因子在 return 中相消):
  open_to_score = (price@score_ts - D_open) / D_open
  score_to_fill = (actual_fill   - price@score_ts) / price@score_ts   # 确认等待侵蚀
  open_to_fill  = (actual_fill   - D_open) / D_open                   # 入场前已跑掉的 edge
  fill_to_H1    = (H1_close      - actual_fill) / actual_fill         # 实际可交易 edge
  open_to_H1    = (1+open_to_fill)*(1+fill_to_H1)-1
按 trigger_path_v2 切片, 看 open_to_fill 是否显著 > fill_to_H1.
"""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, '.')
sys.path.insert(0, '../step9_preflight')
import latr_builder as L

df = pd.read_parquet('datasets/timing_universe_v2_train.parquet')
df = df.dropna(subset=['score_ts', 'actual_fill', 'trigger_path_v2'])
print(f"[diag] eligible rows={len(df)}", flush=True)

rows = []
cache = {}
t0 = time.time()
for _, r in df.iterrows():
    code = r['symbol']; d = int(r['plan_date'])
    m = cache.get((code, d))
    if m is None:
        m = L.load_minute(code, d)
        if len(cache) > 400:
            cache.clear()
        cache[(code, d)] = m
    if m is None or len(m) == 0:
        continue
    d_open = float(m.iloc[0]['open'])
    s = int(r['score_ts'])
    sub = m[m['ts'] <= s]
    if len(sub) == 0:
        continue
    price_score = float(sub.iloc[-1]['close'])
    # H1 close = 最后一笔 ts <= score_ts + 60min (截断到 15:00)
    h1_ts = s + 10000  # +60min in hhmmss encoding
    h1cap = min(h1_ts, int(f"{int(s)//1000000}150000"))
    subh = m[m['ts'] <= h1cap]
    if len(subh) == 0:
        continue
    h1_close = float(subh.iloc[-1]['close'])
    fill = float(r['actual_fill'])
    if d_open <= 0 or price_score <= 0 or fill <= 0:
        continue
    rows.append({
        'trigger_path_v2': r['trigger_path_v2'],
        'open_to_score': (price_score - d_open) / d_open,
        'score_to_fill': (fill - price_score) / price_score,
        'open_to_fill': (fill - d_open) / d_open,
        'fill_to_H1': (h1_close - fill) / fill,
    })

out = pd.DataFrame(rows)
print(f"[diag] computed={len(out)}  ({time.time()-t0:.0f}s)", flush=True)

def block(g):
    return pd.Series({
        'n': len(g),
        'open_to_score': g['open_to_score'].mean(),
        'score_to_fill': g['score_to_fill'].mean(),
        'open_to_fill': g['open_to_fill'].mean(),
        'fill_to_H1': g['fill_to_H1'].mean(),
    })

agg = out.groupby('trigger_path_v2').apply(block).round(4)
print("\n=== by trigger_path_v2 (raw returns) ===")
print(agg.to_string())
print("\n=== ALL ===")
print(block(out).round(4).to_string())

# edge 被入场前吃掉的比例
otf = out['open_to_fill'].mean()
fth = out['fill_to_H1'].mean()
oth = (1 + otf) * (1 + fth) - 1
print(f"\nopen_to_H1(total)={oth:.4f}  open_to_fill(pre-entry)={otf:.4f}  fill_to_H1(tradable)={fth:.4f}")
if oth > 0:
    print(f"pre-entry share of total edge = {otf/oth*100:.1f}%  (higher => edge eaten before entry)")
else:
    print("total edge ~0; no positive edge to be eaten")
