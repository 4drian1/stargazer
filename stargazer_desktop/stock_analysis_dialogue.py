"""Read-only stock discussion, independent of plan draft generation."""
from datetime import datetime
import json
import re

import assistant_workflow as workflow
import services
from deepseek_api import complete


def analysis_only(message):
    codes = workflow.requested_analysis_codes(message)
    if not codes:
        return []
    intent = re.sub(r'不(?:要|用|需)?(?:修改|调整|改变|更改|加入|添加)(?:计划|条件|名单|板块|提醒)?', '', message)
    if re.search(r'加入|添加|移除|删除|剔除|改为|改成|重新制定|确认|调整(?:计划|条件|板块|名单)|设置(?:条件|提醒)|修改(?:计划|条件|名单)', intent):
        return []
    return codes


def local_facts(results):
    lines=[]
    for item in results:
        code=item['code']
        if item.get('status') in ('AVAILABLE','PARTIAL'):
            lines.append(f"{code}：{item.get('data_date','未知')}的已绑定模型结果；本次未重新推理。")
            selection=item.get('selection') or {}
            score=selection.get('personal_score')
            if isinstance(score,(int,float)) and not isinstance(score,bool):
                lines.append(f"Tarot原评分：{score}；只表示该模型研究输出。")
            risk=item.get('risk') or {}
            if risk.get('status')=='AVAILABLE':
                ctx=risk.get('context') or {}
                lines.append(f"Pyxis未来120交易分钟不利下探幅度：232主版{ctx.get('amp_232','未知')}，266对照版{ctx.get('amp_266','未知')}；不是概率或整个次日方向。")
            else:
                lines.append('Pyxis同日风险证据未完整绑定。')
        else:
            lines.append(f"{code}：{item.get('status','UNAVAILABLE')}；当前入口没有已绑定的同规格模型评分，风险幅度未知。")
    return '\n'.join(lines)


def discuss(cfg, message, codes):
    results=[]
    for code in codes:
        try:
            results.append(services.model_stock_analysis(code))
        except (OSError,ValueError) as exc:
            results.append({'code':code,'status':'UNAVAILABLE','reason':type(exc).__name__})
    facts=local_facts(results)
    system=('你是Stargazer个股分析助手。只依据给定模型事实解释，未知保持未知；'
            '不得新增股票、修改计划、调用交易或说成刚重新推理。'
            'Tarot选股分数与Pyxis未来120交易分钟不利下探幅度不可相减；幅度不是概率。'
            '简短回答，区分模型事实、解释、待核对。')
    try:
        response=complete(cfg,[{'role':'system','content':system},
                               {'role':'user','content':json.dumps({'question':message,'model_facts':results},ensure_ascii=False)}],
                          purpose='chat')
        explanation=(response.choices[0].message.content or '').strip()
        if not explanation:
            raise ValueError('EMPTY_CONTENT')
        reply=f'模型事实：\n{facts}\n\nAI解释：\n{explanation}'
        ai_status='AVAILABLE'
    except ValueError as exc:
        reason='EMPTY_CONTENT' if 'EMPTY_CONTENT' in str(exc) else type(exc).__name__
        reply=f'模型事实：\n{facts}\n\nAI解释失败：{reason}。本次未修改正式计划，也未自动重试。'
        ai_status='FAILED'
    workflow.store('stock_analysis',{'created_at':datetime.now().isoformat(),
                                     'question':message,'codes':codes,'model_facts':results,
                                     'reply':reply,'ai_status':ai_status})
    return {'reply':reply,'analysis':{'codes':codes,'ai_status':ai_status},'tools':[]}
