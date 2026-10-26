"""大脑：DeepSeek + function calling + 策略手册system prompt。"""
import json
import os

import yaml

import services

try:
    import paths
    BASE = paths.APP_DIR           # 可写目录（打包后=exe旁）
    RES_DIR = paths.BUNDLE_DIR     # 只读资源目录
except ImportError:
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RES_DIR = BASE


def load_config():
    for name in ("config.yaml", "config.yml"):
        for base in (BASE, RES_DIR):
            p = os.path.join(base, name)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    return yaml.safe_load(f)
    raise FileNotFoundError("未找到 config.yaml，请复制 config.example.yaml 并填入 API key")


def load_system_prompt(cfg):
    p = cfg["agent"].get("strategy_file", "STRATEGY.md")
    if not os.path.isabs(p):
        p = os.path.join(BASE, p)
        if not os.path.exists(p):
            p = os.path.join(RES_DIR, cfg["agent"].get("strategy_file", "STRATEGY.md"))
    with open(p, encoding="utf-8") as f:
        handbook = f.read()
    return ("你是 Stargazer，默认用中文自然对话。普通问题直接回答，不套选股或计划模板。"
            "需要当前计划、持仓、行情、新闻或模型事实时先调用对应只读工具；标明数据日期，区分事实、解释和未知，不凭记忆编实时数据。"
            "用户明确要求Tarot/Pyxis/星币侍从模型分析时才调用模型工具；旧绑定结果必须说明不是本次新推理，"
            "池外工具若返回on_demand.selection且quality_state=OK，可解释本次冻结选股头研究分数，但不能说它已进入正式候选。"
            "on_demand.risk为NOT_COMPUTED时，Pyxis仍未知，不能把选股分数说成风险。"
            "无同规格输入时保持未知。Pyxis输出是未来120交易分钟不利下探幅度，不是概率或整个次日方向。"
            "交易计划的生成、修改、确认走独立流程；本对话不得修改已确认计划、提醒或自动下单。"
            "询问持仓买卖时，先读真实持仓和check_portfolio风控证据，再讨论条件与不确定性；取数失败时说明未知，不替用户决定。"
            "用户明确报告已成交且给出足够成交信息时，可按原记录规则调用record_trade。"
            "以下交易手册仅用于相关交易问题，不约束闲聊的答复格式。\n==== 手册全文 ====\n" + handbook)


TOOLS = [
    {"type": "function", "function": {
        "name": "confirmed_plan", "description": "只读已确认交易计划和股票/板块条件；不生成或修改草案。",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "portfolio_positions", "description": "只读本地真实持仓；不含最新价格，行情需要另查。",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "model_stock_analysis", "description": "查询个股真实Tarot/Pyxis同日绑定模型输出与风险上下文；未覆盖明确报告，零取数/不重训/不修改计划。个股分析优先调用。",
        "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "六位A股代码"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "model_stock_inference", "description": "明确要求模型分析时调用。已绑定则读同日证据；池外仅尝试单股冻结选股头新推理，Pyxis缺完整板块分钟上下文时保持未知；不修改计划。",
        "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "六位A股代码"},
            "model": {"type": "string", "enum": ["tarot", "pyxis", "both"], "description": "用户指定的模型；未指定时both"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "scan_sector",
        "description": "对指定板块跑妖股胚子四层漏斗扫描(技术三层+吸筹信号+v2.3因子)。耗时约1-3分钟。默认排除ST股。",
        "parameters": {"type": "object", "properties": {
            "board_name": {"type": "string", "description": "板块名称或模糊关键词，如 '存储芯片'、'电力'"},
            "with_score": {"type": "boolean", "description": "是否对通过者继续做垃圾度财务打分(更慢)"},
            "max_stocks": {"type": "integer", "description": "最多扫描股票数，默认80"},
            "include_st": {"type": "boolean", "description": "默认false排除ST股; 仅当用户明确说要扫ST/含ST时设true, 并在回答中提示ST退市风险"}},
            "required": ["board_name"]}}},
    {"type": "function", "function": {
        "name": "score_stocks",
        "description": "垃圾度打分：对股票列表按十项财务指标池内百分位打分(0-100, 越低基本面越差越是胚子)。",
        "parameters": {"type": "object", "properties": {
            "codes": {"type": "array", "items": {"type": "string"}, "description": "6位股票代码列表"},
            "year": {"type": "integer", "description": "年报年度，默认最近已披露年度"}},
            "required": ["codes"]}}},
    {"type": "function", "function": {
        "name": "stock_check",
        "description": "个股体检：实时报价+技术位置+漏斗判定+超卖判定+三档参考止损位。",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "6位股票代码"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "check_portfolio",
        "description": "检查持仓(结构止损/时间止损/移动止盈/止盈阶梯)与观察名单(单日-5%超卖预警)。",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "market_flow",
        "description": "板块主力资金流向榜(流入/流出前12)，用于判断轮动方向。",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "market_overview",
        "description": "大盘概览：主要指数(涨跌+成交额)、两市成交额、涨停/跌停家数、最高连板、炸板率、涨停明细。分析盘面/市场情绪必调。",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "sector_detail",
        "description": "板块解剖：涨跌家数广度、平均涨幅、点火判定、领涨龙头榜(带成交额)、领跌榜。分析某个板块强不强、龙头带没带队时调用。",
        "parameters": {"type": "object", "properties": {
            "board_name": {"type": "string", "description": "板块名称或BK代码"},
            "top": {"type": "integer", "description": "龙头榜条数，默认10"}},
            "required": ["board_name"]}}},
    {"type": "function", "function": {
        "name": "research_search",
        "description": "深度网络搜索(Tavily)：查投行/券商研报观点、公司公告求证(照妖镜:题材是真是假)、海外深度消息。比 news_digest 快讯更深更全。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "搜索词，中英文均可，如 '安彩高科 光伏玻璃 公告' 或 'Goldman Sachs China A-shares'"},
            "max_results": {"type": "integer", "description": "结果条数，默认8"},
            "days": {"type": "integer", "description": "只搜最近N天，可选"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "us_market",
        "description": "隔夜美股关键指数与风向标(道指/纳指/标普/费城半导体/金龙中国/英伟达/台积电)，用于晨报和题材映射判断。",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "news_digest",
        "description": "拉取最新财经快讯；给关键词时计算板块新闻密度(刷屏=共识已满只卖不买)。",
        "parameters": {"type": "object", "properties": {
            "keywords": {"type": "array", "items": {"type": "string"}, "description": "板块关键词，可选"},
            "pages": {"type": "integer", "description": "快讯页数，默认2(约200条)"}}}}},
    {"type": "function", "function": {
        "name": "lhb_today",
        "description": "龙虎榜总览：最近一个有数据的交易日全部上榜股票，按净买额排序，看主力资金进攻/出货方向。数据T日傍晚发布。",
        "parameters": {"type": "object", "properties": {
            "top": {"type": "integer", "description": "净买入榜条数，默认20"}}}}},
    {"type": "function", "function": {
        "name": "lhb_stock",
        "description": "某股龙虎榜席位明细：买方/卖方前五席位是谁(机构/北向/知名游资/散户营)，分析连板股接力环境必看。",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "6位股票代码"},
            "date": {"type": "string", "description": "交易日YYYY-MM-DD，缺省=最近有数据的交易日"}},
            "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "wl_add_watch",
        "description": "把股票加入观察名单。",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"}, "reason": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "wl_add_position",
        "description": "登记持仓(买入后调用)，用于每日风控检查。",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"}, "name": {"type": "string"},
            "entry": {"type": "number", "description": "买入价"},
            "entry_date": {"type": "string", "description": "买入日期YYYY-MM-DD"},
            "low250": {"type": "number", "description": "250日低点(结构止损线)"},
            "logic": {"type": "string", "description": "买入逻辑"},
            "catalyst_date": {"type": "string", "description": "催化日期YYYY-MM-DD，无则省略"},
            "sector_bk": {"type": "string", "description": "所属板块东财代码，如BK1106，可省略"}},
            "required": ["code", "name", "entry", "entry_date", "low250"]}}},
    {"type": "function", "function": {
        "name": "wl_remove",
        "description": "从观察名单或持仓中移除股票。",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "record_trade",
        "description": "记录用户的真实成交（买入或卖出某只股票）到交易数据库。当用户在对话里明确说明自己按策略买入或卖出了某只股票时，必须调用此工具把成交计入数据库，这是用户日后做回测验证的依据。记录后在回复里向用户确认。",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["buy", "sell"], "description": "买入或卖出"},
            "code": {"type": "string", "description": "6位股票代码，如 600207"},
            "name": {"type": "string", "description": "股票名称，如 安彩高科"},
            "price": {"type": "number", "description": "成交价（元）"},
            "shares": {"type": "integer", "description": "成交股数，可选"},
            "trade_date": {"type": "string", "description": "成交日期 YYYY-MM-DD，缺省为今天"},
            "reason": {"type": "string", "description": "买入/卖出逻辑或策略依据，可选"}},
            "required": ["action", "code", "name", "price"]}}},
]


class Brain:
    def __init__(self, fresh=False):
        self.cfg = load_config()
        import chat_archive
        self.archive, self.messages = chat_archive.open_session(
            {"role": "system", "content": load_system_prompt(self.cfg)}, fresh=fresh)
        self.max_rounds = self.cfg["agent"].get("max_tool_rounds", 6)

    def _dispatch(self, name, args):
        fn = {"model_stock_analysis": services.model_stock_analysis,
              "model_stock_inference": services.model_stock_inference,
              "scan_sector": services.scan_sector, "score_stocks": services.score_stocks,
              "stock_check": services.stock_check, "market_flow": services.market_flow,
              "news_digest": services.news_digest, "wl_add_watch": services.wl_add_watch,
              "us_market": services.us_market,
              "market_overview": services.market_overview,
              "sector_detail": services.sector_detail,
              "lhb_today": services.lhb_today,
              "lhb_stock": services.lhb_stock,
              "wl_remove": services.wl_remove}.get(name)
        if name == "research_search":
            return services.research_search(self.cfg, **args)
        if name == "confirmed_plan":
            return services.confirmed_plan()
        if name == "portfolio_positions":
            return services.portfolio_positions(self.cfg)
        if name == "check_portfolio":
            return services.check_portfolio(self.cfg)
        if name == "wl_add_position":
            return services.wl_add_position(self.cfg, **args)
        if name == "record_trade":
            return services.trade_add(self.cfg, **args)
        if name in ("wl_add_watch", "wl_remove"):
            return fn(self.cfg, **args) if fn else {"error": "unknown"}
        if fn is None:
            return {"error": f"未知工具 {name}"}
        return fn(**args)

    def chat(self, user_input, on_tool=None):
        import chat_archive
        import re
        from conversation_intent import asks_for_model, asks_for_plan, asks_for_portfolio, asks_for_portfolio_risk, model_scope
        from assistant_context import bounded_messages
        self.messages.append({"role": "user", "content": user_input})
        chat_archive.append(self.archive, [self.messages[-1]])
        # Ordinary stock discussion does not silently become a model request.
        from assistant_workflow import requested_analysis_codes
        prefetched = {}
        codes = requested_analysis_codes(user_input) if asks_for_model(user_input) else []
        scope = model_scope(user_input)
        if codes:
            calls = [{'id': f'bound-model-{len(self.messages)}-{code}', 'type': 'function',
                      'function': {'name': 'model_stock_inference', 'arguments': json.dumps({'code': code, 'model': scope})}}
                     for code in codes]
            self.messages.append({'role': 'assistant', 'tool_calls': calls})
            chat_archive.append(self.archive, [self.messages[-1]])
            for code, call in zip(codes, calls):
                if on_tool:
                    on_tool('model_stock_inference', {'code': code, 'model': scope})
                try:
                    result = self._dispatch('model_stock_inference', {'code': code, 'model': scope})
                except Exception as exc:
                    result = {'code': code, 'status': 'UNAVAILABLE', 'reason': type(exc).__name__}
                prefetched[(code, scope)] = result
                self.messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                      'content': chat_archive.tool_content(result, self.archive)})
                chat_archive.append(self.archive, [self.messages[-1]])
        for name, wanted in (('confirmed_plan', asks_for_plan(user_input)),
                             ('portfolio_positions', asks_for_portfolio(user_input)),
                             ('check_portfolio', asks_for_portfolio_risk(user_input))):
            if not wanted:
                continue
            call = {'id': f'local-{len(self.messages)}-{name}', 'type': 'function',
                    'function': {'name': name, 'arguments': '{}'}}
            self.messages.append({'role': 'assistant', 'tool_calls': [call]})
            chat_archive.append(self.archive, [self.messages[-1]])
            if on_tool:
                on_tool(name, {})
            try:
                result = self._dispatch(name, {})
            except Exception as exc:
                result = {'status': 'UNAVAILABLE', 'reason': type(exc).__name__}
            self.messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                  'content': chat_archive.tool_content(result, self.archive)})
            chat_archive.append(self.archive, [self.messages[-1]])
        from deepseek_api import complete
        for _ in range(self.max_rounds):
            watch_edit = bool(re.search(r'加入观察名单|添加观察|登记持仓|移除观察|删除观察', user_input))
            trade_report = bool(re.search(r'成交|买入了|卖出了|清仓了|我买了|我卖了', user_input))
            active_tools = [t for t in TOOLS if t['function']['name'] not in
                            ('scan_sector', 'score_stocks', 'model_stock_analysis')
                            and (asks_for_model(user_input) or t['function']['name'] != 'model_stock_inference')
                            and (watch_edit or t['function']['name'] not in
                                 ('wl_add_watch', 'wl_add_position', 'wl_remove'))
                            and (trade_report or t['function']['name'] != 'record_trade')]
            rsp = complete(self.cfg, bounded_messages(self.messages[:1], self.messages[1:]), tools=active_tools)
            msg = rsp.choices[0].message
            self.messages.append(msg.model_dump(exclude_none=True))
            chat_archive.append(self.archive, [self.messages[-1]])
            if not msg.tool_calls:
                return (msg.content or '').strip() or 'DeepSeek本次未返回正文；没有自动重试，也没有修改计划。'
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                if on_tool:
                    on_tool(tc.function.name, args)
                try:
                    if tc.function.name == 'model_stock_inference' and (args.get('code'), args.get('model', 'both')) in prefetched:
                        result = prefetched[(args['code'], args.get('model', 'both'))]
                    else:
                        result = self._dispatch(tc.function.name, args)
                except Exception as e:
                    result = {"error": f"工具执行失败: {e}"}
                self.messages.append({"role": "tool", "tool_call_id": tc.id,
                                      "content": chat_archive.tool_content(result, self.archive)})
                chat_archive.append(self.archive, [self.messages[-1]])
        return "（工具调用轮数已达上限，请缩小问题范围再问）"
