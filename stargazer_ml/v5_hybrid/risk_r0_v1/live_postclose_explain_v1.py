"""Expose selected actual frozen-model input cells, not newly invented signals."""
import math


def values(row, keys):
    result={}
    for key in keys:
        value=row.get(key)
        try:
            value=float(value)
            result[key]=value if math.isfinite(value) else None
        except (TypeError,ValueError):
            result[key]=None
    return result


def attach_model_signals(pack, row232, row266):
    context=pack.risk_context
    context['market_state']['model_inputs']=values(row232,[
        'sse_5m_rsi14','sse_15m_rsi14','cyb_15m_rsi14',
        'sse_5m_momentum_change','sse_15m_momentum_change','new_sse_golden_position'])
    context['all_board_signals']=values(row232,[
        'mb5_positive_share','mb15_positive_share',
        'mb5_sse_positive_falling_rate','mb15_sse_positive_falling_rate',
        'mb5_sse_positive_not_new_high_rate','mb15_sse_positive_not_new_high_rate'])
    context['all_board_signals'].update(values(row266,[
        'ccm_v1_5_positive_deceleration','ccm_v1_15_positive_deceleration']))
    context['board_states'][0]['model_inputs']=values(row232,[
        'context_sector_breadth_loo','context_sector_common_change_loo_raw',
        'context_sector_minus_sse_return_raw'])
    context['leader_states'][0]['model_inputs']=values(row232,[
        'context_current_leader_rsi14_5m_raw','context_current_leader_rsi14_15m_raw',
        'context_captured_leader_mean_return_loo_raw','context_captured_leader_common_change_loo_raw'])
    context['leader_states'][0]['model_inputs'].update(values(row266,[
        'ccm_v1_5_own_current_deceleration','ccm_v1_15_own_current_deceleration']))
    context['signal_note']='实际冻结输入单元；NaN保留未知，5/15分钟观察值不是新增硬否决或交易指令'
