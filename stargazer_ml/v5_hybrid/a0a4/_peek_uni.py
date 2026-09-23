import pandas as pd
for pool in ("a0", "a14"):
    p = f"d:/stargazer/stargazer_ml/v5_hybrid/a0a4/datasets/{pool}_universe_candidates.parquet"
    df = pd.read_parquet(p)
    print(f"=== {pool} shape={df.shape}")
    print(sorted(df.columns))
