"""Route explicit plan work without trapping ordinary questions in a draft."""
import re


def asks_for_model(message):
    return bool(re.search(r'Tarot|Pyxis|星币侍从|模型|重新推理|跑(?:一下|一遍)?评分', message, re.I))


def model_scope(message):
    tarot = bool(re.search(r'Tarot|星币侍从|选股模型', message, re.I))
    pyxis = bool(re.search(r'Pyxis|风险模型', message, re.I))
    return 'both' if tarot == pyxis else ('tarot' if tarot else 'pyxis')


def asks_for_sector_scan(message):
    if re.search(r'不(?:要|用|需)(?:再)?(?:扫描|筛选)',message):
        return False
    return bool(re.search(r'扫描|扫一下|扫一遍|筛选.*(?:板块|股票)|板块.*(?:符合.*规则|有没有.*股票|符合.*股票)',message))


def plan_work(message, *, drafting=False):
    text = re.sub(r'不(?:要|用|需)?(?:修改|调整|改变|更改)(?:交易)?计划?', '', message)
    if asks_for_sector_scan(text) and not re.search(r'制定|生成.*计划|修改.*计划|加入.*计划|删除.*计划',text):
        return False
    if re.search(r'如果|假设|会怎么样|如何修改|怎么修改', text) and not re.search(r'请把|帮我把|现在就|直接(?:修改|加入|删除)', text):
        return False
    # 移出类指令只认肯定句；问句和否定句仍走普通对话，避免误删计划
    removal = r'移出|移走|拿掉|移除|删除|剔除|去掉'
    if re.search(removal, text):
        if re.search(r'(?:不要|不用|别|勿|暂不|先不).{0,10}(?:' + removal + ')', text) and not re.search(r'请把|帮我|现在就|直接', text):
            return False
        if re.search(r'吗|？|\?', text) and not re.search(r'请把|帮我|现在就|直接', text):
            return False
    if drafting and (re.fullmatch(r'\s*(?:\d+|[零一二三四五六七八九十百两]+)\s*[只个]?\s*[。！!]?\s*', text)
                     or re.search(r'(?:只要|保留|选出|只看|选|要|给我|最多|就看|控制在)\s*(?:\d+|[零一二三四五六七八九十百两]+)\s*只', text)):
        return True
    action = re.search(r'制定|生成|重新制定|重选|修改|调整|加入|添加|加进|加到|加上|移出|移走|拿掉|移除|删除|剔除|去掉|保留|只要|只留|换成|改成|设置|设为|取消|增补', text)
    target = re.search(r'计划|名单|草案|板块|人工观察|人工选择|人工选股|人工复核|关注条件|提醒|候选|明天|次日', text)
    member_edit = drafting and re.search(r'加入|添加|加进|加到|加上|移出|移走|拿掉|移除|删除|剔除|去掉|保留|只留|换成|改成', text) and re.search(r'\d{6}|这只|那只|它', text)
    return bool((action and target) or member_edit)


def asks_for_plan(message):
    return bool(re.search(r'今天的计划|当前计划|我的计划|已确认计划|计划里|计划中|明天.*名单|次日.*名单', message))


def asks_for_portfolio(message):
    return bool(re.search(r'持仓|仓位|我(?:的)?股票|我买的|要不要卖|该不该卖', message))


def asks_for_portfolio_risk(message):
    return asks_for_portfolio(message) and bool(re.search(r'买|卖|加仓|减仓|止损|止盈|风险', message))
