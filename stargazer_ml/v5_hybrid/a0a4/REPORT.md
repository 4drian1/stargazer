# A0-A4 第一轮报告 (Green V1 Verdict 前置)

- SAMPLE_IDENTITY_PASS = **True**
- FEATURE_DELTA_PASS = **True**

## 消融主表 (IC = OOF Rank IC, 3 seed x 4 fold)

| track | tier | IC mean | IC std | IC IR | top label | pooled label | loss rate | failed rebound |
|---|---|---|---|---|---|---|---|---|
| timing | a1 | -0.0020 | +0.0156 | -0.1292 | -0.00% | +0.12% | +52.68% | +55.79% |
| timing | a2 | +0.0530 | +0.0124 | +4.2572 | +0.04% | +0.12% | +52.68% | +55.79% |
| timing | a3 | +0.3948 | +0.0077 | +51.5376 | +2.38% | +0.12% | +52.68% | +55.79% |
| timing | a4 | +0.5606 | +0.0124 | +45.1912 | +5.93% | +0.12% | +52.68% | +55.79% |
| s0 | a1 | -0.0085 | +0.0195 | -0.4356 | -0.07% | +0.11% | +51.43% | +55.79% |
| s0 | a2 | +0.0105 | +0.0243 | +0.4328 | -0.03% | +0.11% | +51.43% | +55.79% |
| s0 | a3 | +0.2098 | +0.0153 | +13.7347 | +2.37% | +0.11% | +51.43% | +55.79% |
| s0 | a4 | +0.3196 | +0.0178 | +17.9605 | +5.76% | +0.11% | +51.43% | +55.79% |

## Q1-Q4

### Q1_candidate_increment
- GREEN-WATCH-v1.1 相比 V0 是否真的改变了负基线? (A0->A1 = Candidate System Increment)
- a0_universe_timing_label_mean: 0.001831
- a0_old_score_top10_timing_label_mean: 0.003582
- a1_top_timing_label_mean: -2.7711934156378218e-05
- a1_ic: -0.002011

### Q2_zone_increment
- Zone/Structural Liquidity 有没有独立增量?
- ic_a1: -0.002011
- ic_a2: 0.052993
- top_a1: -2.7711934156378218e-05
- top_a2: 0.00035713374485596715

### Q3_dip_increment
- Dip depth/speed 有没有独立增量?
- ic_a2: 0.052993
- ic_a3: 0.394826
- top_a2: 0.00035713374485596715
- top_a3: 0.023805539094650205

### Q4_recovery_increment
- Recovery/Reclaim 是否是最关键的 Timing 信息?
- ic_a3: 0.394826
- ic_a4: 0.560562
- top_a3: 0.023805539094650205
- top_a4: 0.059291713991769544

## 证伪切片 v1_rejected_v11_accepted (A4 timing)
- n: 2080
- label_mean: -0.001276
- label_median: -0.00294
- alpha_mean: -0.002363
- mfe_30m: 0.014619
- mae_30m: -0.014263
- mfe_h1: 0.032047
- mae_h1: -0.030538
- loss_rate: 0.539904
- failed_rebound_rate: 0.518269
- ic: 0.5689634329990247
- top: {'n': 208, 'label_mean': 0.047766, 'label_median': 0.04381, 'alpha_mean': 0.042493, 'mfe_30m': 0.037339, 'mae_30m': -0.010348, 'mfe_h1': 0.091417, 'mae_h1': -0.014616, 'loss_rate': 0.163462, 'failed_rebound_rate': 0.149038}

## FLOAT_MV regime (强制切片)
- a1: pre n=7726 label=0.00022 / post n=1955 label=0.004963
- a4: pre n=7726 label=0.00022 / post n=1955 label=0.004963

## X0 20 槽账本

| tier | trades | ann | win | avg net | MDD(trade) | occ |
|---|---|---|---|---|---|---|
| a1 | 1169 | -21.09% | +39.26% | -0.68% | -100.00% | +15.3 |
| a2 | 1160 | -15.55% | +41.47% | -0.42% | -99.99% | +15.4 |
| a3 | 1091 | +25.81% | +50.32% | +0.85% | -97.32% | +15.6 |
| a4 | 1068 | +124.02% | +60.49% | +2.83% | -84.89% | +15.7 |
| a0_control | 1859 | -1.41% | +44.32% | -0.03% | -99.96% | +15.5 |