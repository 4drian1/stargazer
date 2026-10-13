"""R0-V1 日频风险标签构建器（合同 DRAFT_FOR_REVIEW 的实现草案，v4 门禁加固版）。

设计要点（对应 RISK_LABEL_CONTRACT_V1.md v1.0 DRAFT_FOR_REVIEW）：
- 精度 DAILY_PRICE_OBSERVATION：MAE_5d = min(Low_d/R_P - 1, d in W)，Y_tail_5d = int(MAE_5d <= -delta_P)。
- delta_P 固定 8%（PROPOSED，待冻结）；边界含等号 + 1e-9 浮点容差。
- pre_close 禁用；窗口内公司行为用复权链 SymbolChain 归一化（待验方案）；缺链/因子无效 -> ADJ_EVIDENCE_CONFLICT。
- **基本不变量 + 门禁（v4 加固）**：
  1) 价格与因子：OHLC 全有限、正数，且 low<=open<=high、low<=close<=high；复权因子与归一结果须有限且正数；
     失败保持 NULL，绝不生成 NaN / 负标签。
  2) 日期与日历：返回柱日期必须 == 请求日期；窗口须**恰为 P 后连续 horizon 个交易日**（W[0]>P、全严格递增、
     无重复、长度=horizon）；日历须为**已验收**源（calendar_verified），未验收阻断。
  3) 资格与可用时刻：实际校验 frozen_eligibility 与 candidate_rule_version；未知资格阻断；
     decision_available_ts 须为**真实 14 位合法时间戳，且落在冻结盘后决策窗口 [P 15:00, 下一交易日开盘) 内**
     （< W[0] 09:30）；窗口日内的决策（P+1..P+5）会让标签里部分未来价格已发生 -> 仅工程记录，
     training_eligible=False（不训练、不移动 P 或标签窗口）。
  4) 复权链来源状态分离 + 逐码覆盖证据：MISSING/FAIL -> 冲突；状态回调整**未提供 / UNKNOWN -> 默认阻断**
     （不能靠 factor_at() 返回正数放行）；**OK 与 EMPTY 都须有覆盖 [P,W末日] 的逐码证据**
     （chain_coverage_verified），否则 DATA_INSUFFICIENT（有链事件不代表链完整）；EMPTY 已验覆盖才按无公司行为。
  5) 输出与授权门：补齐 window_dates / source_refs / training_eligible / label_end_time（价格观察窗口结束）
     / data_complete_time（日线到达+公司行为核销完成，另行核销）/ 状态；
     真实入口为**真实授权核验**（scope 精确等于 LABEL_BUILD + 绑定合同/人口版本 + record_hash），
     两布尔前置（authorize + 开关）不等于有效授权；authorize=True 单独不能绕过。

**真实数据构建入口默认拒绝**：构建器只承载契约逻辑；真实人口标签数据集的生成须单独的 LABEL_BUILD 授权。
本文件自带合成测试（不触碰真实人口）；__main__ 在任一测试失败时以非零退出。
"""
from __future__ import annotations
import math
import sys
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional

DELTA_P = 0.08            # 固定 8%（PROPOSED，待冻结）
HORIZON = 5              # 后续交易日数
REAL_DATA_BUILD_ENABLED = False   # 默认拒绝真实人口构建（须经 enable_real_data_build 开启）
_REAL_BUILD_AUTH: Optional[dict] = None

# 冻结候选资格已知取值（来自 dual_path_e_day_registry.json 的 setup 名；其余按具体注册表扩展）
KNOWN_ELIGIBILITY = {"A_SDP", "B_LVB", "ELIGIBLE"}
DEFAULT_CONTRACT_VERSION = "RISK_LABEL_CONTRACT_V1"
DEFAULT_POPULATION_VERSION = "DUAL_PATH_1013"


class RefuseRealDataError(RuntimeError):
    """真实人口构建被拒绝：需单独的标签构建授权。"""


@dataclass
class Candidate:
    code: str
    P_date: int                       # P 日（决策/参考日），8 位日期 YYYYMMDD
    decision_available_ts: Optional[int] = None   # 未知则 None（不伪造）
    frozen_eligibility: str = "UNKNOWN"           # 取自冻结候选记录，须已知
    candidate_rule_version: str = ""              # 候选生成规则版本，须非空
    setup: str = ""


@dataclass
class DailyBar:
    date: int
    open: float
    high: float
    low: float
    close: float

    def valid(self) -> bool:
        xs = (self.open, self.high, self.low, self.close)
        if not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in xs):
            return False
        # 次序：low <= open <= high 且 low <= close <= high（close>high 等次序错 -> 无效）
        if not (self.low <= self.open <= self.high):
            return False
        if not (self.low <= self.close <= self.high):
            return False
        return True


@dataclass
class LabelResult:
    code: str
    P_date: int
    R_P: Optional[float]
    mae_5d: Optional[float]
    y_tail: Optional[int]
    status: str               # LABEL_COMPLETE / DATA_INSUFFICIENT / ADJ_EVIDENCE_CONFLICT / CENSORED_TAIL
    decision_available_ts: Optional[int]
    training_eligible: bool
    window_dates: Optional[List[int]]
    source_refs: dict
    label_end_time: Optional[int]      # 价格观察窗口结束时刻（W[-1] 15:00），非完整数据可用时刻
    data_complete_time: Optional[int]  # 日线到达 + 公司行为核销完成时刻（与 label_end_time 分离；合成测试为 None）
    reason: str = ""


def validate_decision_ts(ts, P_date, next_open_date=None):
    """校验决策可用时刻：未知/None、非 14 位整数、非法日历日期或时刻、早于 P 日 15:00、不早于下一交易日开盘均无效。
    决策须落在冻结盘后窗口 [P 15:00, 下一交易日开盘) 内：
    - 参考价取 P 收盘（>= P 日 15:00）；
    - 且须**早于下一交易日开盘**（< 下一交易日 09:30 = W[0] 开盘）；否则标签里部分"未来"价格已发生，
      不再是"决策后的未来五日风险"，仅可作工程记录，不得训练（不移动 P 或标签窗口）。"""
    if ts is None:
        return False, "unknown"
    if not isinstance(ts, int) or isinstance(ts, bool):
        return False, "not_int"
    s = str(ts)
    if len(s) != 14:
        return False, "not_14_digits"
    try:
        y, mo, d, hh, mm, ss = (int(s[0:4]), int(s[4:6]), int(s[6:8]),
                                int(s[8:10]), int(s[10:12]), int(s[12:14]))
        datetime(y, mo, d, hh, mm, ss)   # 非法月/日/时刻抛 ValueError
    except Exception:
        return False, "unparseable"
    if ts < int(f"{P_date}150000"):       # 参考价取 P 收盘，决策信息须收盘后可得
        return False, "before_P_close"
    if next_open_date is not None and ts >= int(f"{next_open_date}093000"):
        return False, "at_or_after_next_open"  # 已到/过下一交易日开盘 -> 工程记录，不训练
    return True, "ok"


class RiskLabelBuilder:
    def __init__(self,
                 calendar_next: Callable[[int], int],
                 daily_lookup: Callable[[str, int], Optional[DailyBar]],
                 chain_for: Callable[[str], Optional[object]],
                 delta_p: float = DELTA_P,
                 horizon: int = HORIZON,
                 calendar_source: str = "UNVERIFIED",
                 calendar_verified: bool = False,
                 chain_status_for: Optional[Callable[[str], Optional[str]]] = None,
                 chain_coverage_verified: bool = False,
                 contract_version: str = DEFAULT_CONTRACT_VERSION,
                 population_version: str = DEFAULT_POPULATION_VERSION,
                 source_refs: Optional[dict] = None):
        self.calendar_next = calendar_next
        self.daily_lookup = daily_lookup
        self.chain_for = chain_for
        self.delta_p = delta_p
        self.horizon = horizon
        self.calendar_source = calendar_source
        self.calendar_verified = calendar_verified
        self.chain_status_for = chain_status_for
        self.chain_coverage_verified = chain_coverage_verified
        self.contract_version = contract_version
        self.population_version = population_version
        self.source_refs = source_refs or {
            "chain_source": "SymbolChain",
            "calendar_source": calendar_source,
            "contract_version": contract_version,
            "population_version": population_version,
        }

    def _normalize_low(self, code: str, raw_low: float, d: int, P_date: int):
        """将 d 日原始 low 归一化到 P 日同存储尺度；因子或结果非有限/非正 -> 冲突（保持 NULL）。"""
        chain = self.chain_for(code)
        if chain is None:
            return None, True
        fp = chain.factor_at(P_date)
        fd = chain.factor_at(d)
        if not (isinstance(fp, (int, float)) and math.isfinite(fp) and fp > 0):
            return None, True
        if not (isinstance(fd, (int, float)) and math.isfinite(fd) and fd > 0):
            return None, True
        norm = raw_low * (fd / fp)
        if not (math.isfinite(norm) and norm > 0):
            return None, True
        return norm, False

    def build_one(self, cand: Candidate) -> LabelResult:
        ts = cand.decision_available_ts
        src = dict(self.source_refs)

        def _res(status, reason, R_P=None, mae=None, y=None, W=None, training_eligible=False,
                 label_end_time=None, data_complete_time=None):
            return LabelResult(cand.code, cand.P_date, R_P, mae, y, status, ts,
                               training_eligible, W, src, label_end_time,
                               data_complete_time, reason)

        # 3) 资格与可用时刻：实际校验
        elig_known = cand.frozen_eligibility in KNOWN_ELIGIBILITY
        ver_known = bool(cand.candidate_rule_version)
        if not (elig_known and ver_known):
            return _res("DATA_INSUFFICIENT", "frozen eligibility/rule version unknown")

        # 2) 日历须已验收（未验收阻断）
        if not self.calendar_verified:
            return _res("DATA_INSUFFICIENT", "calendar source not verified")

        # 1) R_P（价格校验 + 日期匹配）
        p_bar = self.daily_lookup(cand.code, cand.P_date)
        if p_bar is None or p_bar.date != cand.P_date or not p_bar.valid():
            return _res("DATA_INSUFFICIENT", "P 日柱缺失/日期不符/价格无效")
        R_P = p_bar.close

        # 2) W = P 之后连续 horizon 个交易日（须 W[0] > P，严格递增、无重复、长度=horizon）
        W = []
        d = cand.P_date
        for _ in range(self.horizon):
            d = self.calendar_next(d)
            if d is None:
                break
            W.append(d)
        if len(W) < self.horizon:
            return _res("CENSORED_TAIL", f"窗口仅 {len(W)} 交易日 < {self.horizon}", R_P=R_P, W=W)
        if not (W[0] > cand.P_date
                and all(W[i] < W[i + 1] for i in range(len(W) - 1))
                and len(set(W)) == len(W)):
            return _res("DATA_INSUFFICIENT", "窗口非 P 之后严格递增交易日（含 P 当天/重复/非递增）",
                        R_P=R_P, W=W)

        # 3) 复权链来源状态分离 + 逐码覆盖证据（OK/EMPTY 都须覆盖 [P,W末日]）
        status = self.chain_status_for(cand.code) if self.chain_status_for else None
        if status is None or status == "UNKNOWN":
            # 未提供状态回调 / UNKNOWN：缺逐码覆盖证据，默认阻断；不能靠 factor_at() 返回正数放行
            return _res("DATA_INSUFFICIENT", "复权链状态未知/未提供（缺逐码覆盖证据），默认阻断",
                        R_P=R_P, W=W)
        if status in ("MISSING", "FAIL"):
            return _res("ADJ_EVIDENCE_CONFLICT", f"复权链状态 {status}（缺失/失败）",
                        R_P=R_P, W=W)
        if not self.chain_coverage_verified:
            # 有链事件不代表链完整：OK/EMPTY 都须有覆盖 [P,W末日] 的逐码证据，否则阻断（不判冲突）
            return _res("DATA_INSUFFICIENT",
                        f"复权链状态 {status} 缺 [P,W末日] 逐码覆盖证据（有链≠链完整）",
                        R_P=R_P, W=W)
        no_action = (status == "EMPTY")   # EMPTY 且已验覆盖 -> 无公司行为（因子 1.0）

        # 3) 决策可用时刻真实校验（上限 = 下一交易日开盘，非窗口末日）
        ts_valid, ts_reason = validate_decision_ts(ts, cand.P_date, W[0])

        lows_norm = []
        for d in W:
            bar = self.daily_lookup(cand.code, d)
            if bar is None or bar.date != d:               # 2) 返回柱日期必须匹配请求
                return _res("DATA_INSUFFICIENT", f"窗口日 {d} 缺失/日期不符", R_P=R_P, W=W)
            if not bar.valid():                            # 1) 必要价格无效 -> 整条 NULL
                return _res("DATA_INSUFFICIENT",
                            f"窗口日 {d} 价格无效(low<=0/次序错等)", R_P=R_P, W=W)
            if no_action:
                norm_low = bar.low                          # EMPTY 已验：按无公司行为（因子 1.0）
            else:
                norm_low, conflict = self._normalize_low(cand.code, bar.low, d, cand.P_date)
                if conflict:                               # 1) 缺链/因子无效
                    return _res("ADJ_EVIDENCE_CONFLICT",
                                f"窗口日 {d} 缺复权链/因子无效，无法归一化", R_P=R_P, W=W)
            lows_norm.append(norm_low)

        mae = min(l / R_P - 1 for l in lows_norm)
        y = int(mae <= -self.delta_p + 1e-9)               # 边界含等号 + 浮点容差
        training_eligible = ts_valid                        # 资格/版本已已知；ts 落在 [P15:00, W[-1]15:00] 才适格
        label_end_time = int(f"{W[-1]}150000")              # 价格观察窗口结束时刻（W[-1] 15:00）
        # data_complete_time：日线实际到达 + 公司行为核销完成时刻，须另行核销；合成测试为 None
        data_complete_time = None
        return _res("LABEL_COMPLETE", "ok", R_P=R_P, mae=mae, y=y, W=W,
                    training_eligible=training_eligible, label_end_time=label_end_time,
                    data_complete_time=data_complete_time)

    def build_real_population(self, candidates: List[Candidate], authorize: bool = False):
        """真实人口构建入口：**真实授权核验**（非两布尔前置）。默认拒绝；须经 enable_real_data_build
        开启 REAL_DATA_BUILD_ENABLED，且调用时 authorize=True，并复核授权记录绑定的合同/人口版本一致。"""
        if not authorize:
            raise RefuseRealDataError(
                "真实人口标签构建被拒绝（默认）。须经单独的「标签构建授权」显式传入 authorize=True 后方可运行；"
                "DRAFT 名称不等于标签生成授权。")
        if not REAL_DATA_BUILD_ENABLED:
            raise RefuseRealDataError(
                "REAL_DATA_BUILD_ENABLED=False；须先经单独的 LABEL_BUILD 授权调用 "
                "enable_real_data_build(auth_record) 开启。authorize=True 单独不能绕过此门。")
        if (_REAL_BUILD_AUTH is None
                or _REAL_BUILD_AUTH.get("contract_version") != self.contract_version
                or _REAL_BUILD_AUTH.get("population_version") != self.population_version):
            raise RefuseRealDataError(
                "授权记录绑定的合同/人口版本与本构建器不符（scope 须精确 LABEL_BUILD 且版本一致）；"
                "两布尔前置条件不等于有效授权核验。")
        return [self.build_one(c) for c in candidates]


def enable_real_data_build(auth_record: dict) -> None:
    """开启真实人口构建。**单独的标签构建授权**入口：须提供经签署的最小授权记录（真实核验，非两布尔）。"""
    global REAL_DATA_BUILD_ENABLED, _REAL_BUILD_AUTH
    required = {"signer", "authorized_at", "scope", "record_hash",
                "contract_version", "population_version"}
    missing = required - auth_record.keys()
    if missing:
        raise ValueError(f"auth_record 缺少字段: {missing}")
    if auth_record["scope"] != "LABEL_BUILD":                      # 精确匹配，禁子串
        raise ValueError("scope 必须精确等于 'LABEL_BUILD'（禁子串匹配如 NOT_LABEL_BUILD）")
    payload = "|".join([str(auth_record["signer"]), str(auth_record["authorized_at"]),
                        str(auth_record["scope"]), str(auth_record["contract_version"]),
                        str(auth_record["population_version"])]).encode("utf-8")
    if auth_record["record_hash"] != hashlib.sha256(payload).hexdigest():
        raise ValueError("record_hash 不匹配（须为 sha256(signer|authorized_at|scope|contract_version|population_version)）")
    REAL_DATA_BUILD_ENABLED = True
    _REAL_BUILD_AUTH = auth_record


# ---------------- 合成测试（不触碰真实人口） ----------------
class _FakeChain:
    def __init__(self, events: List[tuple]):
        self.events = sorted(events)

    def factor_at(self, asof: int) -> float:
        f = 1.0
        for d, m in self.events:
            if d <= asof and m and m > 0:
                f *= m
        return f


class _NaNChain:
    def factor_at(self, asof: int) -> float:
        return float("nan")


def _make_calendar(dates: List[int]):
    s = sorted(dates)
    idx = {d: i for i, d in enumerate(s)}

    def nxt(d: int) -> Optional[int]:
        i = idx.get(d)
        if i is None or i + 1 >= len(s):
            return None
        return s[i + 1]
    return nxt


def _bar(date, o, h, l, c):
    return DailyBar(date, o, h, l, c)


def _lookup_factory(bars_by_code: dict):
    def lk(code, date):
        return bars_by_code.get(code, {}).get(date)
    return lk


def _mk_cand(code="X", P=20260101, ts=None, elig="A_SDP", ver="TEST_v0"):
    return Candidate(code, P, decision_available_ts=ts, frozen_eligibility=elig,
                     candidate_rule_version=ver)


def _builder(cal, bars, chain_for, chain_status_for=lambda c: "OK",
             calendar_verified=True, chain_coverage_verified=True, **kw):
    # 默认状态回调返回 "OK"（合成测试显式提供状态；真实管线须由缓存状态回填）
    return RiskLabelBuilder(cal, _lookup_factory(bars), chain_for,
                           calendar_verified=calendar_verified,
                           chain_status_for=chain_status_for,
                           chain_coverage_verified=chain_coverage_verified, **kw)


def run_synthetic_tests() -> dict:
    rep = {}
    cal = _make_calendar([20260101 + i for i in range(20)])

    # 1) 8% 边界（含等号）
    bars = {20260101: _bar(20260101, 100, 101, 100, 100)}
    for d in [20260102, 20260103, 20260104, 20260105, 20260106]:
        bars[d] = _bar(d, 92.0, 93, 92.0, 93)
    b = _builder(cal, {"X": bars}, lambda c: _FakeChain([]))
    r = b.build_one(_mk_cand())
    rep["t1_boundary_exact"] = (r.status == "LABEL_COMPLETE" and r.y_tail == 1 and abs(r.mae_5d - (-0.08)) < 1e-9)
    bars2 = dict(bars)
    for d in [20260102, 20260103, 20260104, 20260105, 20260106]:
        bars2[d] = _bar(d, 92.01, 93, 92.01, 93)
    rep["t1_boundary_off"] = _builder(cal, {"X": bars2}, lambda c: _FakeChain([])).build_one(_mk_cand()).y_tail == 0

    # 2) 公司行为归一（拆分 mult=2.0 在第 3 窗口日）
    bars3 = {20260101: _bar(20260101, 100, 101, 100, 100)}
    for d in [20260102, 20260103]:
        bars3[d] = _bar(d, 99, 100, 99, 100)
    bars3[20260104] = _bar(20260104, 48, 49, 48, 49)
    for d in [20260105, 20260106]:
        bars3[d] = _bar(d, 48, 49, 48, 49)
    chain = _FakeChain([(20260104, 2.0)])
    r3 = _builder(cal, {"X": bars3}, lambda c: chain).build_one(_mk_cand())
    rep["t2_ca_normalize_no_false_tail"] = (r3.status == "LABEL_COMPLETE" and r3.y_tail == 0 and abs(r3.mae_5d - (-0.04)) < 1e-9)
    bars3b = dict(bars3); bars3b[20260104] = _bar(20260104, 46, 47, 46, 47)
    r3b = _builder(cal, {"X": bars3b}, lambda c: chain).build_one(_mk_cand())
    rep["t2_ca_normalize_real_tail"] = (r3b.y_tail == 1 and abs(r3b.mae_5d - (-0.08)) < 1e-9)

    # 3) 缺一天
    bars4 = {20260101: _bar(20260101, 100, 101, 100, 100)}
    for d in [20260102, 20260103, 20260105, 20260106]:
        bars4[d] = _bar(d, 95, 96, 95, 96)
    rep["t3_missing_day"] = _builder(cal, {"X": bars4}, lambda c: _FakeChain([])).build_one(_mk_cand()).status == "DATA_INSUFFICIENT"

    # 4) 无效价格（low<=0）
    bars5 = {20260101: _bar(20260101, 100, 101, 100, 100)}
    for d in [20260102, 20260103, 20260104, 20260105, 20260106]:
        bars5[d] = _bar(d, 90, 91, 90, 91)
    bars5[20260103] = _bar(20260103, 0, 0, 0, 90)
    rep["t4_invalid_price_whole_null"] = _builder(cal, {"X": bars5}, lambda c: _FakeChain([])).build_one(_mk_cand()).status == "DATA_INSUFFICIENT"

    # 5) 日历尾部
    short_cal = _make_calendar([20260101, 20260102, 20260103, 20260104])
    bars6 = {20260101: _bar(20260101, 100, 101, 100, 100),
             20260102: _bar(20260102, 95, 96, 95, 96), 20260103: _bar(20260103, 95, 96, 95, 96),
             20260104: _bar(20260104, 95, 96, 95, 96)}
    rep["t5_calendar_tail"] = _builder(short_cal, {"X": bars6}, lambda c: _FakeChain([])).build_one(_mk_cand()).status == "CENSORED_TAIL"

    # 6) 未知可用时间 -> LABEL_COMPLETE 但 training_eligible=False
    bars6b = {20260101: _bar(20260101, 100, 101, 100, 100)}
    for d in [20260102, 20260103, 20260104, 20260105, 20260106]:
        bars6b[d] = _bar(d, 95, 96, 95, 96)
    r6 = _builder(cal, {"X": bars6b}, lambda c: _FakeChain([])).build_one(_mk_cand(ts=None))
    rep["t6_unknown_ts"] = (r6.decision_available_ts is None and r6.status == "LABEL_COMPLETE" and r6.training_eligible is False)

    # 7) 缺链 -> 冲突
    bars7 = {20260101: _bar(20260101, 100, 101, 100, 100)}
    for d in [20260102, 20260103, 20260104, 20260105, 20260106]:
        bars7[d] = _bar(d, 95, 96, 95, 96)
    rep["t7_missing_chain"] = _builder(cal, {"X": bars7}, lambda c: None).build_one(_mk_cand()).status == "ADJ_EVIDENCE_CONFLICT"

    # 8) 复权因子 NaN -> 冲突，不生成 NaN 标签
    bars8 = {20260101: _bar(20260101, 100, 101, 100, 100)}
    for d in [20260102, 20260103, 20260104, 20260105, 20260106]:
        bars8[d] = _bar(d, 95, 96, 95, 96)
    r8 = _builder(cal, {"X": bars8}, lambda c: _NaNChain()).build_one(_mk_cand())
    rep["t8_nan_factor"] = (r8.status == "ADJ_EVIDENCE_CONFLICT" and r8.mae_5d is None and r8.y_tail is None)

    # 9) 返回柱日期与请求不符 -> NULL（lookup 返回 date+1 的柱）
    def lk_wrong(code, date):
        b = _lookup_factory({"X": bars8})(code, date)
        if b is None:
            return None
        return _bar(date + 1, b.open, b.high, b.low, b.close)
    r9 = RiskLabelBuilder(cal, lk_wrong, lambda c: _FakeChain([]), calendar_verified=True,
                          chain_status_for=lambda c: "OK").build_one(_mk_cand())
    rep["t9_wrong_date_bar"] = (r9.status == "DATA_INSUFFICIENT" and "日期不符" in r9.reason)

    # 10) 日历重复返回同一天 -> 窗口非严格递增 -> NULL
    order = [20260101, 20260102, 20260102, 20260103, 20260104, 20260105, 20260106]
    def dup_cal(d: int) -> Optional[int]:
        i = order.index(d) if d in order else -1
        if i < 0 or i + 1 >= len(order):
            return None
        return order[i + 1]
    bars10 = {20260101: _bar(20260101, 100, 101, 100, 100)}
    for d in [20260102, 20260103, 20260104, 20260105, 20260106]:
        bars10[d] = _bar(d, 95, 96, 95, 96)
    r10 = _builder(dup_cal, {"X": bars10}, lambda c: _FakeChain([])).build_one(_mk_cand())
    rep["t10_dup_calendar_day"] = (r10.status == "DATA_INSUFFICIENT" and "非递增" in r10.reason)

    # 11) 候选资格 UNKNOWN / 版本缺失 -> 阻断
    r11a = _builder(cal, {"X": bars8}, lambda c: _FakeChain([])).build_one(_mk_cand(elig="UNKNOWN"))
    r11b = _builder(cal, {"X": bars8}, lambda c: _FakeChain([])).build_one(_mk_cand(ver=""))
    rep["t11_unknown_eligibility"] = (r11a.status == "DATA_INSUFFICIENT" and r11b.status == "DATA_INSUFFICIENT")

    # 12) close>high 次序错 -> valid()=False；窗口内该日 -> NULL
    bad_bar = _bar(20260103, 100, 100, 100, 101)
    rep["t12_close_gt_high_unit"] = (bad_bar.valid() is False)
    bars12 = {20260101: _bar(20260101, 100, 101, 100, 100)}
    for d in [20260102, 20260104, 20260105, 20260106]:
        bars12[d] = _bar(d, 95, 96, 95, 96)
    bars12[20260103] = bad_bar
    r12 = _builder(cal, {"X": bars12}, lambda c: _FakeChain([])).build_one(_mk_cand())
    rep["t12_close_gt_high_window"] = (r12.status == "DATA_INSUFFICIENT" and "价格无效" in r12.reason)

    # 13) EMPTY / MISSING / FAIL 分离
    # 13a EMPTY 未验覆盖 -> DATA_INSUFFICIENT（不判冲突）
    r13a = _builder(cal, {"X": bars8}, lambda c: _FakeChain([]),
                    chain_status_for=lambda c: "EMPTY", chain_coverage_verified=False).build_one(_mk_cand())
    rep["t13_empty_no_evidence"] = (r13a.status == "DATA_INSUFFICIENT" and "EMPTY" in r13a.reason)
    # 13b EMPTY 已验覆盖 -> LABEL_COMPLETE（因子 1.0，无伪尾部），ts 有效则适格
    r13b = _builder(cal, {"X": bars8}, lambda c: _FakeChain([]),
                    chain_status_for=lambda c: "EMPTY", chain_coverage_verified=True,
                    ).build_one(_mk_cand(ts=20260101190000))
    rep["t13_empty_verified"] = (r13b.status == "LABEL_COMPLETE" and r13b.training_eligible is True
                                and abs(r13b.mae_5d - (-0.05)) < 1e-9)
    # 13c MISSING -> 冲突
    r13c = _builder(cal, {"X": bars8}, lambda c: _FakeChain([]),
                    chain_status_for=lambda c: "MISSING").build_one(_mk_cand())
    rep["t13_missing_conflict"] = (r13c.status == "ADJ_EVIDENCE_CONFLICT")
    # 13d FAIL -> 冲突
    r13d = _builder(cal, {"X": bars8}, lambda c: _FakeChain([]),
                    chain_status_for=lambda c: "FAIL").build_one(_mk_cand())
    rep["t13_fail_conflict"] = (r13d.status == "ADJ_EVIDENCE_CONFLICT")

    # 14) 日历未验收 / 窗口含 P 当天
    # 14a 未验收 -> 阻断
    r14a = _builder(cal, {"X": bars8}, lambda c: _FakeChain([]),
                    calendar_verified=False).build_one(_mk_cand())
    rep["t14_calendar_unverified"] = (r14a.status == "DATA_INSUFFICIENT" and "calendar" in r14a.reason)
    # 14b 日历返回 P 当天（next(P)=P）-> 窗口含 P -> 阻断
    def p_inclusive_cal(d: int) -> Optional[int]:
        seq = [20260101, 20260101, 20260102, 20260103, 20260104, 20260105, 20260106]
        i = seq.index(d) if d in seq else -1
        if i < 0 or i + 1 >= len(seq):
            return None
        return seq[i + 1]
    r14b = _builder(p_inclusive_cal, {"X": bars8}, lambda c: _FakeChain([])).build_one(_mk_cand())
    rep["t14_window_includes_P"] = (r14b.status == "DATA_INSUFFICIENT" and "P 之后" in r14b.reason)

    # 15) 决策可用时刻格式校验
    bars15 = {20260101: _bar(20260101, 100, 101, 100, 100)}
    for d in [20260102, 20260103, 20260104, 20260105, 20260106]:
        bars15[d] = _bar(d, 95, 96, 95, 96)
    mk = lambda ts: _builder(cal, {"X": bars15}, lambda c: _FakeChain([])).build_one(_mk_cand(ts=ts))
    r15_zero = mk(0)
    r15_badmonth = mk(20261301120000)      # 13 月非法
    r15_noon = mk(20260101120000)          # P 日中午，早于收盘
    r15_ok = mk(20260101190000)            # P 日盘后（早于下一交易日开盘）-> 适格
    rep["t15_ts_zero_invalid"] = (r15_zero.status == "LABEL_COMPLETE" and r15_zero.training_eligible is False)
    rep["t15_ts_badmonth_invalid"] = (r15_badmonth.status == "LABEL_COMPLETE" and r15_badmonth.training_eligible is False)
    rep["t15_ts_noon_invalid"] = (r15_noon.status == "LABEL_COMPLETE" and r15_noon.training_eligible is False)
    rep["t15_ts_ok_eligible"] = (r15_ok.status == "LABEL_COMPLETE" and r15_ok.training_eligible is True
                                and r15_ok.label_end_time == 20260106150000)

    # 17) 复权覆盖门须管 OK 与未知状态（有链≠链完整）
    # 17a OK + 无逐码覆盖证据 -> 阻断（不靠 factor_at 正数放行）
    r17a = _builder(cal, {"X": bars8}, lambda c: _FakeChain([(20260104, 2.0)]),
                    chain_status_for=lambda c: "OK", chain_coverage_verified=False).build_one(_mk_cand())
    rep["t17_ok_no_coverage"] = (r17a.status == "DATA_INSUFFICIENT" and "覆盖证据" in r17a.reason)
    # 17b 未提供状态回调（chain_status_for=None）-> 默认阻断
    r17b = RiskLabelBuilder(cal, _lookup_factory({"X": bars8}), lambda c: _FakeChain([]),
                           calendar_verified=True, chain_status_for=None,
                           chain_coverage_verified=True).build_one(_mk_cand())
    rep["t17_status_none_blocked"] = (r17b.status == "DATA_INSUFFICIENT" and "未知" in r17b.reason)
    # 17c 状态 UNKNOWN -> 默认阻断
    r17c = _builder(cal, {"X": bars8}, lambda c: _FakeChain([]),
                    chain_status_for=lambda c: "UNKNOWN").build_one(_mk_cand())
    rep["t17_status_unknown_blocked"] = (r17c.status == "DATA_INSUFFICIENT" and "未知" in r17c.reason)
    # 17d OK + 已验覆盖 -> 正常（因子生效）
    r17d = _builder(cal, {"X": bars8}, lambda c: _FakeChain([(20260104, 2.0)]),
                    chain_status_for=lambda c: "OK", chain_coverage_verified=True).build_one(_mk_cand(ts=20260101190000))
    rep["t17_ok_verified"] = (r17d.status == "LABEL_COMPLETE" and r17d.training_eligible is True)

    # 18) 决策窗口 = 冻结盘后窗口 [P 15:00, 下一交易日开盘)；窗口日内的决策不再适格
    # P=20260105，合成连续日历下 W=[20260106..20260110]，下一交易日 W[0]=20260106，开盘 09:30
    bars18 = {20260105: _bar(20260105, 100, 101, 100, 100)}
    for d in [20260106, 20260107, 20260108, 20260109, 20260110]:
        bars18[d] = _bar(d, 95, 96, 95, 96)
    # 18a P 日盘后（19:00，早于下一交易日开盘）-> 适格
    r18a = _builder(cal, {"X": bars18}, lambda c: _FakeChain([]),
                    chain_status_for=lambda c: "OK").build_one(
        _mk_cand(P=20260105, ts=20260105190000))
    rep["t18_postmarket_eligible"] = (r18a.status == "LABEL_COMPLETE"
                                     and r18a.training_eligible is True
                                     and r18a.label_end_time == 20260110150000
                                     and r18a.data_complete_time is None)
    # 18b 恰在下一交易日开盘（09:30）-> 已到未来日，不适格（工程记录）
    r18b = _builder(cal, {"X": bars18}, lambda c: _FakeChain([]),
                    chain_status_for=lambda c: "OK").build_one(
        _mk_cand(P=20260105, ts=20260106093000))
    rep["t18_at_next_open_not_eligible"] = (r18b.status == "LABEL_COMPLETE"
                                           and r18b.training_eligible is False)
    # 18c 窗口第 2 日 15:00（20260107）-> 标签部分价格已发生，不适格
    r18c = _builder(cal, {"X": bars18}, lambda c: _FakeChain([]),
                    chain_status_for=lambda c: "OK").build_one(
        _mk_cand(P=20260105, ts=20260107150000))
    rep["t18_within_window_not_eligible"] = (r18c.status == "LABEL_COMPLETE"
                                            and r18c.training_eligible is False)
    # 18d 远晚于整个窗口（20260112 16:00）-> 工程记录
    r18d = _builder(cal, {"X": bars18}, lambda c: _FakeChain([]),
                    chain_status_for=lambda c: "OK").build_one(
        _mk_cand(P=20260105, ts=20260112160000))
    rep["t18_decision_after_window"] = (r18d.status == "LABEL_COMPLETE"
                                       and r18d.training_eligible is False
                                       and r18d.label_end_time == 20260110150000
                                       and r18d.data_complete_time is None)

    # 16) 授权 scope 精确匹配 + 版本绑定
    b_guard = RiskLabelBuilder(cal, lambda c, d: None, lambda c: None)

    def make_auth_record(scope="LABEL_BUILD", contract=DEFAULT_CONTRACT_VERSION,
                         pop=DEFAULT_POPULATION_VERSION):
        # 补齐所有合法字段 + 匹配哈希；只改 scope/版本以隔离被测项
        payload = "|".join(["user", "2026-10-10", scope, contract, pop]).encode()
        return {"signer": "user", "authorized_at": "2026-10-10", "scope": scope,
                "contract_version": contract, "population_version": pop,
                "record_hash": hashlib.sha256(payload).hexdigest()}

    raised_a = False
    try:
        b_guard.build_real_population([], authorize=False)
    except RefuseRealDataError:
        raised_a = True
    rep["t_g1_refuse_default"] = raised_a
    raised_b = False
    try:
        b_guard.build_real_population([], authorize=True)   # authorize=True 但开关关
    except RefuseRealDataError:
        raised_b = True
    rep["t_g2_authorize_alone_blocked"] = raised_b

    # 16a scope 子串漏洞：NOT_LABEL_BUILD 不能开启（其余字段合法+哈希匹配，只改 scope）
    scope_substr_caught = False
    scope_substr_reason = ""
    try:
        enable_real_data_build(make_auth_record(scope="NOT_LABEL_BUILD"))
    except ValueError as e:
        scope_substr_caught = True
        scope_substr_reason = str(e)
    rep["t16_scope_substring_rejected"] = (scope_substr_caught and "scope" in scope_substr_reason.lower())
    globals()["REAL_DATA_BUILD_ENABLED"] = False
    globals()["_REAL_BUILD_AUTH"] = None

    # 16b 合法授权但绑定错误版本 -> 真实入口拒绝
    wrong_bind_accepted = True
    try:
        enable_real_data_build(make_auth_record(contract="OTHER_CONTRACT", pop="OTHER_POP"))
        # 构建器绑定默认版本，应与授权记录版本不符 -> 拒绝
        b_guard.build_real_population([], authorize=True)
    except RefuseRealDataError:
        wrong_bind_accepted = False
    except Exception:
        wrong_bind_accepted = False
    rep["t16_wrong_version_rejected"] = (wrong_bind_accepted is False)
    globals()["REAL_DATA_BUILD_ENABLED"] = False
    globals()["_REAL_BUILD_AUTH"] = None

    # 16c 合法授权 + 版本一致 -> 两道门均过（不触碰真实数据，空候选）
    try:
        enable_real_data_build(make_auth_record())
        passed = (b_guard.build_real_population([], authorize=True) == [])
        rep["t_g3_two_factor_pass"] = passed
    except Exception:
        rep["t_g3_two_factor_pass"] = False
    finally:
        globals()["REAL_DATA_BUILD_ENABLED"] = False
        globals()["_REAL_BUILD_AUTH"] = None

    rep["ALL_PASS"] = all(v for k, v in rep.items() if k.startswith("t"))
    return rep


if __name__ == "__main__":
    rep = run_synthetic_tests()
    print("=== R0-V1 合成测试报告（v5 决策窗口收口版） ===")
    for k, v in rep.items():
        print(f"  {k}: {v}")
    if not rep["ALL_PASS"]:
        print("TESTS FAILED -> 非零退出")
        sys.exit(1)
    print("ALL TESTS PASS")
