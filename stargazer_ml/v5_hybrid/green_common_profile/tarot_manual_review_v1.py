"""Near-miss A/SECOND rows for human review; never formal trading admission."""
from types import FunctionType
import numpy as np


def latest_record(g, core, old_a, calendar):
    from descending_tops_v1 import for_frame
    if for_frame(g)['excluded']:return None
    p = len(g)-1
    p_date = int(g.date.iloc[p])
    expected = [d for d in calendar if d <= p_date][-70:]
    if len(expected) != 70 or not np.array_equal(g.date.iloc[-70:].to_numpy(dtype=int), expected):
        return None
    failures = {}

    def conditions(frame, anchor, atr):
        result = core.new_conditions(frame, anchor, atr)
        wave = result['wave']
        if (not wave.get('observation_node') or wave.get('rise', 0) <= 0
                or wave.get('peak_date',0) <= wave.get('pivot_date',0)):
            return result
        reasons = []
        if not result['volume_gate']: reasons.append('VOLUME_GATE_NOT_PASSED')
        if wave['rise'] <= .15: reasons.append('RISE_NOT_ABOVE_15PCT')
        if not .236 <= wave['retrace'] <= .786: reasons.append('RETRACE_OUTSIDE_23_6_TO_78_6')
        failures[('A', anchor)] = reasons
        return {**result, 'new_gate': True}

    def second(frame):
        c,v,low,high = (frame[k].to_numpy() for k in ('close','volume','low','high'))
        down = c < frame.pre_close.to_numpy()
        for t in range(max(8,p-7),p-4):
            if not (c[t] < frame.open.iloc[t] and v[t] > 0 and down[t+1:t+4].all()): continue
            pivot = next((j for j in range(t-3,max(1,t-60)-1,-1)
                          if j>=2 and low[j]<=min(low[j-2:j+3])),None)
            if pivot is None: continue
            peak = max(high[pivot:t+1]);rise = peak/low[pivot]-1
            if rise <= 0: continue
            rebounds = [j for j in range(t+4,p) if c[j]>frame.pre_close.iloc[j]]
            if not rebounds or not down[rebounds[-1]+1:p+1].all(): continue
            longest = current = 0
            for flag in down[t:p+1]:
                current = current+1 if flag else 0;longest = max(longest,current)
            reasons = []
            if not v[t] > 1.25*frame.volume.iloc[t-8:t].mean(): reasons.append('PANIC_VOLUME_BELOW_1_25X')
            if not v[t]>v[t+1]>v[t+2]>v[t+3]: reasons.append('FIRST_LEG_VOLUME_NOT_DECREASING')
            if rise <= .15: reasons.append('RISE_NOT_ABOVE_15PCT')
            failures[('SECOND',t)] = reasons
            return {p:dict(stage='SECOND',anchor=t,pivot=pivot,peak=float(peak),prior_rise=float(rise),
                top_count=len(core.top_structure(frame,t,p)),longest_down=longest,
                old_v3_price_gate=bool(c[p]<c[t+3]),first_end=t+3)}
        return {}

    namespace = dict(core.candidates.__globals__, P_END=int(g.date.iloc[p]),
                     new_conditions=conditions, second_candidates=second)
    # Frozen candidates still enforces ST/liquidity, top count, six-down and feature recipe.
    records,_ = FunctionType(core.candidates.__code__,namespace)(g,old_a)
    for rec in records:
        if rec['P'] != int(g.date.iloc[p]) or not rec['position_ready']: continue
        anchor = int(g.index[g.date==rec['anchor_date']][0])
        reasons = failures.get((rec['stage'],anchor),[])
        if reasons: return {**rec,'failed_gates':reasons}
    return None
