# 共同成员候选对照准备交付

日期：2026-10-19。状态：COMPLETE_PREPARATION_NOT_FIT，非训练就绪。

## 本轮已完成

- 验证器在重放前拒绝任何已有目录，并独占写入报告/指纹；已有原报告与指纹保留。caveats_and_correction_v2.md追加澄清版本身份与缺失语义。
- common_candidate_aggregation_v1.py实现候选—板块等权汇总与候选侧表挂载：包括NaN占位的全部预期板块须在场、重复/未知板块拒绝、全未知仍NaN、键/旧特征/标签不变。不扩候选行、不添加所属板块数样本权重；全板块共享上下文与所属板块分列。
- 独立新学习列34，完整旧232保留，共266逻辑列/532含缺失指示。新增支持度和覆盖不在学习清单中。
- 6个新unittest、6个旧矩阵unittest、44项原共同成员断言全部通过；四个改动/新增Python文件内存语法编译通过。环境无Ruff，未声称lint通过。
- 复用原派生输入核验四段人口：train60712/select18071/calibration4548/test2988，共86319行。键/标签/232字段指纹存入报告；268项新准备绑定复核无漂移，原验收报告/指纹哈希也包含在内。

## 交付位置

- contracts/COMMON_MOTION_CANDIDATE_COMPARISON_20261019.md：独立研究对照规范，父配置精确SHA与模型参数在新config中继承。
- common_candidate_preparation_20261019_v1/：plan、config、report、manifest、stop_record；合计69641B，终局COMPLETE_PREPARATION_NOT_FIT，不续写/重跑。
- prepare_common_candidate_comparison_v1.py：只有--prepare，没有训练路径。对已存在本次目录重跑会拒绝，不应删目录规避。

## 未完成与下一步

未新增HTTP，未重放4196原始文件，未生成真实34列，未拟合预处理或模型，未部署监控/EXE/JEV。
原四字段板块赋值覆盖验收不是新34列完整候选输入验收。本轮只验证汇总/挂载机制及旧人口绑定，不能报告新候选输入已ready。
后续只需一次消费已冻结源，在原候选截止构造全板块/上涨组共享上下文与所属板块LOO字段，按本规范汇总并挂载真实34列；实际验证原键/标签/232不变、train新列非全空、前缀/未来扰动、LOO及逐列指纹。可用内存侧表，不必复制大型Parquet。
随后绑定真实输入/入口验收与配置SHA取得独立FIT授权，再执行固定Ridge/LGBM各一次；不继承旧一次性FIT授权，不按test选参数。
四根4GiB累计护栏仍有效。本轮核验后余382487B；后续FIT提案3MiB明显不足，未预留。须明确资源方案/扩容授权，不删旧模型或换目录绕预算。
