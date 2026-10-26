"""Bounded offline verification for the common-member (3-bar stable) local
momentum caliber (diagnosis #18.109 §4, defect-fixed review pass). No stockdb /
no model / no disk IO: pure arithmetic on synthetic returns.

Covers the five defects found in the first review pass:
 #1 fixed common set (no fake momentum when membership changes),
 #2 session-local window (no lunch bridging),
 #3 all/positive scopes write distinct fields (no overwrite),
 #4 leader identity determined first, never silently promoted,
 #5 basket retracement (giveback) not published as a completed field.
"""
import os, sys
import numpy as np

import continuous_motion_matrix_v1 as cmm_orig
import continuous_motion_common_v1 as cmm


def _make(shape, base=0.001):
    return np.full(shape, base, dtype=float)


def _stocks(r):
    return cmm_orig.path_arrays(1 + r, np.isfinite(r), 5)


def _common_mask(values, width):
    """Replicate the 3-bar session-local common window (audit reference)."""
    valid = np.isfinite(values)
    M, n = values.shape
    days = n // width
    half = width // 2
    v = valid.reshape(M, days, width)
    c3 = v & np.roll(v, 1, axis=2) & np.roll(v, 2, axis=2)
    c3[:, :, [0, 1, 2, half, half + 1, half + 2]] = False
    return c3.reshape(M, n)


def _same_set_decel(values, width):
    """Reference: deceleration computed ENTIRELY within common(t) (the correct fix)."""
    cm = _common_mask(values, width)
    n = cm.sum(axis=0)
    safe = n >= 1
    # NOTE: use np.where, NOT (cm * values) -- False * nan == nan in numpy and would
    # poison the sum at masked positions that hold nan source values.
    zv = np.where(cm, values, 0)
    x0 = np.divide(zv.sum(axis=0), np.where(safe, n, 1))
    x1 = np.divide(np.where(cm, np.roll(values, 1, axis=1), 0).sum(axis=0), np.where(safe, n, 1))
    x2 = np.divide(np.where(cm, np.roll(values, 2, axis=1), 0).sum(axis=0), np.where(safe, n, 1))
    cur = (1 + x0) / (1 + x1) - 1
    prev = (1 + x1) / (1 + x2) - 1
    return np.where(safe, prev - cur, np.nan)


def _roll_decel(values, width):
    """Reference: the BUGGY roll-based deceleration (crosses common(t-1) vs common(t))."""
    cm = _common_mask(values, width)
    n = cm.sum(axis=0)
    safe = n >= 1
    x0 = np.divide(np.where(cm, values, 0).sum(axis=0), np.where(safe, n, 1))
    level = np.where(safe, x0, np.nan)
    r = level / np.roll(level, 1, axis=0) - 1   # r[t] = level(t) / level(t-1)
    r = np.where(safe, r, np.nan)
    return np.roll(r, 1, axis=0) - r


fails = []
def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    if not cond:
        fails.append(name)


FREQ = 5
W = 240 // FREQ          # 48 bars/day
HALF = W // 2            # 24 (PM open index)
DAYS = 2
T = DAYS * W             # 96
DEAD_MOD = [0, 1, 2, HALF, HALF + 1, HALF + 2]
within_day = ~np.isin(np.arange(W), DEAD_MOD)

# ----- base setup (layout is (stocks, bars)) -----
N = 4
r = _make((N, T))
stocks = _stocks(r)
members = [0, 1, 2, 3]

# 1) GAP IN THE 3-BAR WINDOW EXCLUDES A MEMBER FROM THE COMMON SET
#    member 0 missing only at bar 51 (day2, bar index 3): at bar 52 (day2 bar4, a
#    clean 3-bar window 50,51,52) member 0 is absent -> excluded; at bar 54 (window
#    52,53,54) it is back. (Bars 48,49,50 of day2 are session-open dead and empty.)
r_gap = r.copy(); r_gap[0, 51] = np.nan
stocks_gap = _stocks(r_gap)
out = cmm.board_motion_common(r_gap, members, FREQ, stocks_gap)
check("common set excludes member with gap at t-1 (t52 m0 out)",
      bool(out["common_count"][52] == 3) and bool(out["denominator"][52] == N))
check("member returns to common once 3-bar window clean (t54 m0 in)",
      bool(out["common_count"][54] == 4))
check("coverage disclosed = 3/4 at t52", abs(float(out["coverage"][52]) - 0.75) < 1e-12)
check("coverage = 1.0 at fully-common t54", abs(float(out["coverage"][54]) - 1.0) < 1e-12)
check("local level finite at t54 despite earlier gap", np.isfinite(out["day_return"][54]))
dead_idx = [i for i in (0, 1, 2, HALF, HALF + 1, HALF + 2, W, W + 1, W + 2,
                        W + HALF, W + HALF + 1, W + HALF + 2)]
check("local level NaN at every session-open dead bar (no 3-bar window)",
      all(np.isnan(out["day_return"][i]) for i in dead_idx))

# 2) SESSION-LOCAL: a gap in day1 must NOT bleed into day2; AND lunch not bridged
r_dl = r.copy(); r_dl[0, 47] = np.nan   # last bar of day1 only
stocks_dl = _stocks(r_dl)
out_dl = cmm.board_motion_common(r_dl, members, FREQ, stocks_dl)
check("no cross-day bridging (day1 gap does not exclude day2 common)",
      bool(out_dl["common_count"][54] == 4))

# defect #2: PM-open bars must not inherit AM membership across lunch (hardened:
# the gap bar AND the two following bars are cleared, so the 3-bar window never
# reaches the lunch gap; symmetric AM-open handled too).
r_ses = r.copy()
r_ses[0, :HALF] = np.nan      # member0 valid only in PM
r_ses[1, HALF:] = np.nan      # member1 valid only in AM
stocks_ses = _stocks(r_ses)
out_ses = cmm.board_motion_common(r_ses, members, FREQ, stocks_ses)
check("PM bar does not bridge to AM (m1 absent at PM t30 -> count 3)",
      bool(out_ses["common_count"][30] == 3))
check("member1 (AM-only) is NOT in PM common at t30 (not count 4)",
      bool(out_ses["common_count"][30] != 4))
check("PM session-open bars (t24,t25,t26) have empty common",
      bool(out_ses["common_count"][HALF] == 0 and out_ses["common_count"][HALF + 1] == 0
            and out_ses["common_count"][HALF + 2] == 0))
check("member1 (AM-only) IS in AM common at t6; member0 (PM-only) correctly absent",
      bool(out_ses["common_count"][6] == 3))

# 3) LEGACY COMPLETE-MEMBER STILL BLOCKS (regression guard; old caliber unchanged)
legacy = cmm_orig.board_motion(r_gap, members, FREQ, stocks_gap)
check("legacy board_motion blocks bar with any missing member (return_ NaN)",
      np.isnan(legacy["return_"][51]))
check("new common caliber is NOT blocked by an isolated gap elsewhere",
      np.isfinite(out["day_return"][52]) and np.isnan(legacy["return_"][51]))

# 4) min_common GUARD (disclosure threshold, not credibility gate)
r_one = r.copy()
r_one[1, 50] = r_one[1, 51] = np.nan
r_one[2, 50] = r_one[2, 51] = np.nan
r_one[3, 50] = r_one[3, 51] = np.nan
stocks_one = _stocks(r_one)
out_one_1 = cmm.board_motion_common(r_one, members, FREQ, stocks_one, min_common=1)
out_one_2 = cmm.board_motion_common(r_one, members, FREQ, stocks_one, min_common=2)
check("min_common=1 still yields level with 1 common member",
      np.isfinite(out_one_1["day_return"][52]))
check("min_common=2 guards empty/under-supported output (NaN)",
      np.isnan(out_one_2["day_return"][52]))
check("coverage disclosed even for 1/4 support (no silent credibility)",
      abs(float(out_one_1["coverage"][52]) - 0.25) < 1e-12)

# 5) DEFECT #1 -- fixed common set: no fake momentum when a member recovers
#    member0 = +0.10/bar constant; member1 = 0/bar but missing until bar 50.
r1 = _make((2, T), 0.0)
r1[0, :] = 0.10
r1[1, :50] = np.nan
stocks1 = _stocks(r1)
out1 = cmm.board_motion_common(r1, [0, 1], FREQ, stocks1)
fin_r = out1["return_"][np.isfinite(out1["return_"])]
fin_d = out1["deceleration"][np.isfinite(out1["deceleration"])]
check("DEFECT#1 no fake momentum on member recovery (return_ all ~0)",
      fin_r.size > 0 and np.max(np.abs(fin_r)) < 1e-12)
check("DEFECT#1 deceleration all ~0 on member recovery",
      fin_d.size > 0 and np.max(np.abs(fin_d)) < 1e-12)

# 5a) DEFECT #1 (review 2) -- member change AND price change: deceleration must be
#     computed within a single common(t) set. Construct A,C varying identically and B
#     of a different mean, B joining the window partway. The basket return varies, and
#     the membership changes at the join; a roll-based decel would spike ~3pp there,
#     while the same-set decel stays ~0. The assertion below matches the code's output
#     to the same-set reference (tight) and confirms it is far from the roll-based
#     (buggy) value -- so reverting to roll would fail this test.
r1b = _make((3, T), 0.0)
t_idx = np.arange(T)
r1b[0, :] = 0.10 + 0.01 * np.sin(0.3 * t_idx)   # A
r1b[2, :] = 0.10 + 0.01 * np.sin(0.3 * t_idx)   # C (identical to A)
r1b[1, :] = 0.20 + 0.01 * np.sin(0.3 * t_idx)   # B different mean, joins later
r1b[1, :21] = np.nan                            # B missing until bar 21
stocks1b = _stocks(r1b)
out1b = cmm.board_motion_common(r1b, [0, 1, 2], FREQ, stocks1b)
ref_ss = _same_set_decel(r1b, W)
ref_roll = _roll_decel(r1b, W)
fin = np.isfinite(out1b["deceleration"])
check("DEFECT#1b deceleration matches same-set reference (member change + price change)",
      np.allclose(out1b["deceleration"][fin], ref_ss[fin], atol=1e-12))
check("DEFECT#1b decel is small (no spurious spike at membership join)",
      np.nanmax(np.abs(out1b["deceleration"])) < 1e-2)
check("DEFECT#1b decel is NOT the roll-based (buggy) value at join",
      np.nanmax(np.abs(out1b["deceleration"] - ref_roll)) > 1e-3)

# 5b) DEFECT #3 -- all vs positive scopes write DISTINCT fields (no overwrite)
#     board A: positive day, decel=+k ; board B: negative day, decel=-k.
r3 = _make((2, T), 0.0)
t_idx = np.arange(T)
r3[0, :] = 0.10 - 0.0001 * t_idx          # positive day, decel = +0.0001
r3[1, :] = -0.10 + 0.0001 * t_idx         # negative day, decel = -0.0001
stocks3 = _stocks(r3)
ctx3 = cmm.context_common(r3, [[0], [1]], {"sse": np.ones(T), "cyb": np.ones(T)},
                             FREQ, stocks=stocks3)
all_key = "cm_common5_all_deceleration_mean"
pos_key = "cm_common5_positive_deceleration_mean"
check("DEFECT#3 both all and positive fields exist",
      all_key in ctx3 and pos_key in ctx3)
check("DEFECT#3 all-mean != positive-mean (no overwrite)",
      np.nanmax(np.abs(ctx3[all_key] - ctx3[pos_key])) > 1e-9)
# positive scope must select a strict subset of boards; its mean therefore reflects
# only positive-day boards, which is distinct from the all-boards mean above.

# 5c) DEFECT #4 -- leader identity determined first, never silently promoted
r4 = _make((3, T), 0.01)
r4[0, :] = 0.10     # member0 is the true (highest) leader everywhere
stocks4 = _stocks(r4)
out4_excl = cmm.board_motion_common(r4, [0, 1, 2], FREQ, stocks4, exclude=0)
check("DEFECT#4 excluded true leader not published (current_deceleration all NaN)",
      np.all(np.isnan(out4_excl["current_deceleration"])))
check("DEFECT#4 no silent promotion: excluded case has no finite leader metric",
      not np.any(np.isfinite(out4_excl["current_deceleration"])))
out4_full = cmm.board_motion_common(r4, [0, 1, 2], FREQ, stocks4)
check("DEFECT#4 true leader published when not excluded (finite somewhere)",
      np.any(np.isfinite(out4_full["current_deceleration"])))

# 5c2) DEFECT #2 (review 2) -- open-only leader must still be captured. A leads ONLY
#      at the session-open dead bars (0,1) then drops; B leads every later bar. With
#      the old code the leader was only recorded where a common window existed (dead
#      bars empty), so A was never captured. The new code records the real leader every
#      bar; A is therefore captured and, because A has a distinctive deceleration at a
#      valid common bar (bar 4), the captured_deceleration there equals mean(A,B) decel
#      = 0.05 (not 0). This proves A (open-only leader) is tracked and published.
r2b = _make((3, T), 0.0)
r2b[0, 0] = r2b[0, 1] = 0.10          # A leads only at open (dead bars)
r2b[0, 2] = r2b[0, 3] = 0.0
r2b[0, 4:] = -0.10                    # A drops -> B leads from bar 2 on
r2b[1, :] = 0.05                      # B leads at every non-dead bar
stocks2b = _stocks(r2b)
out2b = cmm.board_motion_common(r2b, [0, 1, 2], FREQ, stocks2b)
# bar 4 is the first valid common bar (needs 2,3,4); A decel there = 0.0-(-0.10)=0.10
check("DEFECT#2 open-only leader (A) is captured and published (captured_decel ~0.05)",
      abs(out2b["captured_deceleration"][4] - 0.05) < 1e-9)
check("DEFECT#2 current leader at bar4 is B (decel ~0)",
      abs(out2b["current_deceleration"][4] - 0.0) < 1e-9)

# 5c3) DEFECT #2 (review 3) -- an all-invalid bar must NOT mint a phantom leader that is
#      then captured forever. 3-stock board with STRICTLY DISTINCT baselines (A<B<C? no:
#      A=0.001 < C=0.0015 < B=0.002) so B is the ONLY genuine leader at every bar. A jump at
#      bar 14 makes B's per-stock deceleration nonzero, and bar 12 has NO valid quote on any
#      stock. Old code: argmax over an all -inf column returns index 0 (A) at bar 12, so A is
#      captured and pollutes captured_deceleration at later valid bars (bar 15) with a bogus
#      0-deceleration stock -> value halves. Fix: no leader recorded at an all-invalid bar, so
#      captured_deceleration[15] == B's own deceleration only. (Ties are avoided on purpose:
#      with equal baselines argmax returns index 0 anyway, which would be a different,
#      unrelated artifact and would not isolate the all-invalid bug.)
r2c = _make((3, T), 0.001)   # A baseline 0.001
r2c[1, :] = 0.002            # B strictly highest baseline -> sole real leader
r2c[2, :] = 0.0015           # C between A and B
r2c[1, 14] = 0.201           # B jump -> B's per-stock deceleration at bar 15 is nonzero
r2c[:, 12] = np.nan          # ALL invalid bar 12 -> would mint phantom A in old code
stocks2c = _stocks(r2c)
out2c = cmm.board_motion_common(r2c, [0, 1, 2], FREQ, stocks2c)
b_dec = stocks2c['deceleration'][1, 15]          # B's own per-stock deceleration at bar 15
check("DEFECT#2c no leader at all-invalid bar (current_deceleration NaN)",
      np.isnan(out2c["current_deceleration"][12]))
check("DEFECT#2c phantom avoided: captured_deceleration == real leader B only",
      abs(out2c["captured_deceleration"][15] - b_dec) < 1e-9)
check("DEFECT#2c captured_deceleration is NOT the halved (phantom-included) value",
      abs(out2c["captured_deceleration"][15] - b_dec / 2) > 1e-3)
# Contrast: when A IS a genuine leader (wins bar 3) it must be captured and INCLUDED. This
# proves the fix excludes only the phantom, not every early leader.
r2c_b = _make((3, T), 0.001)
r2c_b[1, :] = 0.002
r2c_b[2, :] = 0.0015
r2c_b[0, 3] = 0.050          # A wins bar 3 -> genuine leader
r2c_b[1, 5] = 0.050          # B wins bar 5 -> genuine leader
r2c_b[1, 14] = 0.201         # B jump -> nonzero decel at bar 15
stocks2c_b = _stocks(r2c_b)
out2c_b = cmm.board_motion_common(r2c_b, [0, 1, 2], FREQ, stocks2c_b)
a_dec_b = stocks2c_b['deceleration'][0, 15]
b_dec_b = stocks2c_b['deceleration'][1, 15]
check("DEFECT#2c genuine leader A is still included (captured == mean(A,B) decel)",
      abs(out2c_b["captured_deceleration"][15] - (a_dec_b + b_dec_b) / 2) < 1e-9)

# 5e) DEFECT #1 (review 3) -- candidate-keyed LOO must pass the candidate's GLOBAL member
#      identity (its code), not its position within the board. The module matches `exclude`
#      as `exclude in members`; for members [10,20,30] candidate 10: passing the code 10
#      excludes correctly (denominator 2, common_count 2); passing the position m.index(10)
#      == 0 does NOT exclude (0 not in [10,20,30]) -> denominator 3, common_count 3. This is
#      exactly the verifier bug fixed at verify_...coverage_v1 line 80 (exclude=c).
r_e = np.full((40, T), 0.001, dtype=float)
members_e = [10, 20, 30]
stocks_e = cmm_orig.path_arrays(1 + r_e, np.isfinite(r_e), FREQ)
out_e_code = cmm.board_motion_common(r_e, members_e, FREQ, stocks_e, exclude=10)
out_e_pos = cmm.board_motion_common(r_e, members_e, FREQ, stocks_e, exclude=0)
check("DEFECT#1e exclude by MEMBER VALUE (code 10) -> denominator 2, common 2",
      (out_e_code["denominator"][5] == 2) and (out_e_code["common_count"][5] == 2))
check("DEFECT#1e buggy position exclude (0) fails to exclude -> denominator 3, common 3",
      (out_e_pos["denominator"][5] == 3) and (out_e_pos["common_count"][5] == 3))

# 5d) DEFECT #5 -- basket retracement (giveback) is NOT published as a field
out5 = cmm.board_motion_common(r, members, FREQ, stocks)
check("DEFECT#5 basket giveback not published in common caliber",
      "giveback" not in out5)
check("DEFECT#5 local deceleration IS published", "deceleration" in out5)
check("DEFECT#5 deceleration NaN at session-open dead bars",
      all(np.isnan(out5["deceleration"][i]) for i in dead_idx))

# 6) CONTEXT_COMMON aggregates and matches schema (markets required, as in context)
markets = {"sse": np.ones(T, dtype=float), "cyb": np.ones(T, dtype=float)}
ctx = cmm.context_common(r, [members], markets, FREQ, stocks=stocks)
check("context_common keys == schema_common", list(ctx) == cmm.schema_common(FREQ))
cov = ctx[cmm.schema_common(FREQ)[-1]]
within = ~np.isin(np.arange(T) % W, DEAD_MOD)
check("context_common pooled coverage = 1.0 on full data (outside session-open dead bars)",
      np.allclose(cov[within], 1.0) and np.allclose(cov[~within], 0.0))
check("context_common runs for positive scope (no crash)",
      cmm.schema_common(FREQ)[0] in ctx)

# 7) VALIDATION of the fast LOO derivation (common_loo_derive_v1) against the REAL module.
#    The population acceptance verifier uses the fast path (one board compute + algebraic
#    per-member LOO) instead of calling board_motion_common per membership, because real
#    concept boards average ~80 members (~80k memberships total). This proves the fast path
#    is EXACTLY board_motion_common -- board-level fields AND each member's exclude output --
#    on small and large (real-scale) boards.
import common_loo_derive_v1 as derv
FIELDS_ALL = ('return_', 'deceleration', 'current_deceleration', 'captured_deceleration')


def _validate_loo_derivation(Msize, seed):
    rng = np.random.default_rng(seed)
    rnd = rng.standard_normal((Msize, T)) * 0.01
    rnd[rng.random((Msize, T)) < 0.05] = np.nan          # random gaps
    members_t = list(range(Msize))
    stocks_t = cmm_orig.path_arrays(1 + rnd, np.isfinite(rnd), FREQ)
    ref = cmm.board_motion_common(rnd, members_t, FREQ, stocks_t)
    inter = derv.compute_board(rnd, members_t, FREQ, stocks_t)
    bf = derv.board_fields(inter)
    for f in FIELDS_ALL:
        if not np.allclose(bf[f], ref[f], atol=1e-12, equal_nan=True):
            return False, f"board-level {f}"
    for i in members_t:
        loo_ref = cmm.board_motion_common(rnd, members_t, FREQ, stocks_t, exclude=i)
        for f in FIELDS_ALL:
            if not np.allclose(derv.loo_field(inter, i, f), loo_ref[f], atol=1e-12, equal_nan=True):
                return False, f"loo {f} i={i}"
    return True, "ok"


ok_small, why_small = _validate_loo_derivation(5, 1)
ok_large, why_large = _validate_loo_derivation(80, 2)
check("LOO-DERIVE matches board_motion_common exactly (small M=5 board)", ok_small)
check("LOO-DERIVE matches board_motion_common exactly (real-scale M=80 board)", ok_large)


def _validate_loo_all(Msize, seed):
    rng = np.random.default_rng(seed)
    rnd = rng.standard_normal((Msize, T)) * 0.01
    rnd[rng.random((Msize, T)) < 0.05] = np.nan
    members_t = list(range(Msize))
    stocks_t = cmm_orig.path_arrays(1 + rnd, np.isfinite(rnd), FREQ)
    inter = derv.compute_board(rnd, members_t, FREQ, stocks_t)
    for f in FIELDS_ALL:
        allm = derv.loo_all_members(inter, f)
        for i in members_t:
            if not np.allclose(allm[i], derv.loo_field(inter, i, f), atol=1e-12, equal_nan=True):
                return False, f"loo_all {f} i={i}"
    return True, "ok"


ok_all_small, _ = _validate_loo_all(5, 3)
ok_all_large, _ = _validate_loo_all(80, 4)
check("LOO-ALL vectorized == loo_field scalar (small M=5 board)", ok_all_small)
check("LOO-ALL vectorized == loo_field scalar (real-scale M=80 board)", ok_all_large)

print("\nCOMMON-MEMBER TEST:", "ALL PASS" if not fails else f"{len(fails)} FAIL -> {fails}")
sys.exit(1 if fails else 0)
