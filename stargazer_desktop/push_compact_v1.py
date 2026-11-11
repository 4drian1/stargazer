"""Two concise Markdown cards; facts only, no I/O or new execution conditions."""
import math
from datetime import datetime

def finite(x):return type(x) in (int,float) and math.isfinite(x)
def number(x,n=2):return (('%.*f'%(n,x)).rstrip('0').rstrip('.')) if finite(x) else '未知'
def clock(x):return datetime.strptime(str(x),'%Y%m%d%H%M%S').strftime('%H:%M')
def check(x):return '✓' if x is True else '✗' if x is False else '待核'
def flat(x):return ' '.join(str(x or '').split())
def identity(p):return flat(p.get('name'))+' '+flat(p.get('code'))
def indicator_lines(p):
    lines=[]
    for period in ('5','15'):
        pair=(p.get('period_pairs') or {}).get(period) or {};a=pair.get('previous') or {};b=pair.get('current') or {}
        basis='实时试算、未收盘' if pair.get('provisional') else '完整K线' if pair.get('status')=='KNOWN' else '待核'
        lines += [period+'分钟｜'+basis+'｜低点 '+number(a.get('price'))+' → '+number(b.get('price')),
            'RSI6 '+number(a.get('rsi6'))+' → '+number(b.get('rsi6'))+' ｜ RSI15 '+number(a.get('rsi15'))+' → '+number(b.get('rsi15')),
            'MACD：DIF '+number(a.get('dif'),4)+' → '+number(b.get('dif'),4)+' ｜ DEA '+number(a.get('dea'),4)+' → '+number(b.get('dea'),4),
            '跨日底背离：DIF '+check(pair.get('dif_divergence'))+' ｜ RSI '+check(pair.get('rsi_divergence'))]
    return lines

def strong(p):
    priority=flat(p.get('priority_label'));title='异动'+('·'+priority if priority else '')
    lower,upper=p.get('level_lower'),p.get('level_upper');price=p.get('price');low=((p.get('period_pairs') or {}).get('5') or {}).get('current') or {}
    state='状态：低点'+number(low.get('price'))+' ｜ '+flat(p.get('level_label') or '黄金位')+number(lower)+'—'+number(upper)
    if finite(price) and finite(lower) and finite(upper) and 0<lower<=upper:
        if lower<=price<=upper:state+=' ｜ 现价'+number(price)+'，带内'
        else:
            edge=upper if price>upper else lower;state+=' ｜ 现价'+number(price)+'，离带%+.1f%% ⚠️'%((price/edge-1)*100)
    flags=[((p.get('period_pairs') or {}).get(t) or {}).get(k) for t in ('5','15') for k in ('dif_divergence','rsi_divergence')]
    passed=sum(x is True for x in flags);known=sum(type(x) is bool for x in flags)
    strength=('未齐（%d/4项已知）'%known if known<4 else ('强' if passed>=3 else '中' if passed==2 else '弱' if passed==1 else '无')+'（%d/4）'%passed)
    rules=[]
    if p.get('pullback_rule'):
        from strong_pullback_policy_v1 import render
        rules=render(p['pullback_rule'])
    lines=['【'+title+'】'+identity(p)+' ｜ '+flat(p.get('board') or '板块待核')+' ｜ '+clock(p['asof'])+'触发',
        '━━━━━━━━━━━━━━━━━━━',state,*rules,'背离强度：'+strength+'（仅指标计数）','───────────────────',*indicator_lines(p),
        '━━━━━━━━━━━━━━━━━━━','程序未下单。']
    return dict(title=title,body='\n\n'.join(lines))

def trigger(p):
    title='计划票·扳机';votes=p.get('votes') or {};required=p.get('required_votes',2)
    passed=sum(votes.get(k) is True for k in ('repair','momentum','oversold'))>=required
    combo='三选二' if required==2 else '组合（需%d项）'%required
    state='状态：'+combo+('✓' if passed else '待核')+'（'+' '.join(label+check(votes.get(k)) for k,label in [('repair','位置'),('momentum','动能'),('oversold','超卖')])+')'
    verified=p.get('advice_verified') is True and p.get('advice_status')=='ADVICE_READY'
    decision=p.get('decision') if verified else None
    label={'BUY':'建议买入（请人工决定）','DEFER':'⏸️暂缓（技术组合与AI意见分列）'}.get(decision,'风险/AI研判尚未完成')
    if not verified:
        label={'AI_FAILED':'AI分析失败（本地事实保留，不自动重试）',
               'ENTRY_SESSION_CLOSED':'已结束当日入场，保留观察',
               'WAIT_RISK':'等待风险证据核验',
               'AI_ATTEMPT_RESERVED':'AI研判中'}.get(p.get('advice_status'),label)
    lines=['【'+title+'】'+identity(p)+' ｜ '+clock(p['asof']),'━━━━━━━━━━━━━━━━━━━',state,'判定：'+label]
    if p.get('provisional'):lines.append('实时试算、未收盘；指标可变。')
    if verified:
        reasons=p.get('risk_reasons') or []
        lines+=['AI风险依据：'+flat(x) for x in reasons[:3]] or ['AI风险依据：未列出反对项']
        if p.get('summary_reason'):lines.append('AI权衡：'+flat(p['summary_reason']))
        pending=p.get('pending_facts') or []
        if pending:lines+=['───────────────────','待核信息（不等于否决）：',*['□ '+flat(x) for x in pending[:3]]]
    else:
        lines.append('依据：尚无通过校验的AI结论，不生成环境否决项。')
        if p.get('wait_reason') and p.get('advice_status')!='ADVICE_READY':lines.append('等待说明：'+flat(p['wait_reason'])[:240])
    if p.get('period_pairs'):lines+=['───────────────────',*indicator_lines(p)]
    lines+=['━━━━━━━━━━━━━━━━━━━','程序未下单。']
    return dict(title=title,body='\n\n'.join(lines))
