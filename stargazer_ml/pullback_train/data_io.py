# -*- coding: utf-8 -*-
"""数据路径治理（2026-09-09 评审拍板，修复 3dp 旧版样本导致 PM 复核崩溃的坑）。

规则：每年唯一正式样本路径由本模块给出，禁止脚本自行拼路径猜测。
启动时断言 AM/PM 决策点齐全（7dp），缺失立即报错而不是静默返回空池。
"""
import json
import os

import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
EXPECTED_DPS = (945, 1000, 1030, 1330, 1400, 1430, 1450)
# 正式路径清单（唯一事实来源；旧文件不得再被脚本引用）
MANIFEST = {
    "samples": {
        2025: "minute_samples_2025.parquet",
        2026: "minute_labels/minute_samples_2026.parquet",
    },
    "known_stale": {
        # 根目录 3dp 旧版：无 PM 决策点，仅供考古，禁止入模
        "minute_samples_2026.parquet": "3dp legacy, superseded by minute_labels/",
    },
    "labels": {y: f"dual_head_labels_{y}.parquet" for y in (2025, 2026)},
    "t7_features": {y: f"t7_features_{y}.parquet" for y in (2025, 2026)},
    "path_features": {y: f"minute_path_features_{y}.parquet" for y in (2025, 2026)},
    "t2pm_features": {y: f"t2pm_features_{y}.parquet" for y in (2025, 2026)},
}


def manifest_path():
    return os.path.join(OUT_DIR, "data_manifest.json")


def dump_manifest():
    with open(manifest_path(), "w", encoding="utf-8") as f:
        json.dump(MANIFEST, f, ensure_ascii=False, indent=2)


def load_samples(year, dps=EXPECTED_DPS):
    """读取某年正式分钟样本表，并断言决策点齐全。缺失即抛错。"""
    rel = MANIFEST["samples"][year]
    path = os.path.join(OUT_DIR, rel)
    df = pd.read_parquet(path)
    missing = sorted(set(dps) - set(df["dp"].unique().tolist()))
    if missing:
        raise RuntimeError(
            f"{rel} 缺少决策点 {missing}（期望 {sorted(dps)}）——"
            f"疑似旧版/损坏文件，禁止入模，请检查 data_manifest.json")
    return df


if __name__ == "__main__":
    dump_manifest()
    for y in (2025, 2026):
        d = load_samples(y, columns=None) if False else load_samples(y)
        print(f"{y}: {len(d):,} 行, dp 齐全 {sorted(d['dp'].unique())}")
    print("manifest ->", manifest_path())
