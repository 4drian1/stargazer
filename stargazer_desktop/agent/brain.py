"""大脑：DeepSeek + function calling + 策略手册system prompt。"""
import json
import os

import yaml

import services
from golden_stock_analysis import stock_golden_levels

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
            "复盘用review_plan_minutes读取历史名单和本地分钟。用户可说今天选的怎么样、昨天看上的那几只触发没、有没有避开下跌等，不要求固定口令。某次板块扫描为空不代表历史计划或人工复核为空，复盘以实际工具名单为准。结合上下文区分选股日与观察日，确有歧义才问；不把缺历史AI记录说成预测正确，不把模拟浮盈说成真实利润。"
            "用户明确要求Tarot/Pyxis/星币侍从模型分析时才调用模型工具；旧绑定结果必须说明不是本次新推理，"
            "池外工具若返回on_demand.selection且quality_state=OK，可解释本次冻结选股头研究分数，但不能说它已进入正式候选。"
            "on_demand.risk为NOT_COMPUTED时，Pyxis仍未知，不能把选股分数说成风险。"
            "黄金节点、共振位和关注区通过stock_golden_levels取得，未通过选股门也可计算位置；"
            "这是项目现有黄金规则的代码结果，不是模型学习出的预测或LLM手算。不得因手册没写就否认工具能力。"
            "逐股报告数据日、收盘、ATR、工具返回的关注区和原始共振带；两者分列，非买点。V4使用至少三来源低位共振、上沿受昨收下跌超过3%限制的ATR观察区，多档全列；不得擅自改回贴收盘单档。"
            "共振须按source_waves计不同波段，共享高点相关；无共振就明确无，不把单节点说成共振。"
            "临时端点和STALE_REFERENCE_ONLY不得当已确认的次日计划位置。黄金计算结果按本对话留存。用户说把刚算的股票加入明天计划时，调用prepare_golden_plan；未选规则先问默认三选二或自定义，草案仍须按钮确认。"
            "用户指定板块扫描或询问板块是否有符合现行规则的股票时调用scan_sector_rules，不能拿旧scan_sector漏斗或财务垃圾度分冒充Tarot。"
            "按返回的formal_count/mode分别说明正式候选与未过门人工复核，模型分数降序；正式候选为空才展示人工复核最多五只，不足如实。"
            "报告data_date、归档成员版本和缺数据数量；歧义名称按choices询问，不将未扫描当作无合格股。"
            "total_catalog_members是归档总成员，member_count是模型支持市场内成员；daily_validated_count仅核验日线，不是评分数，评分数只用scored_count。不得编人数或自行调整模型排序。"
            "逐股给observation_bands全部V4黄金共振加ATR观察区；无区写暂无，不拿其他价格补。分数不是胜率，盘中仍须组合信号和风险判断。扫描不修改计划。"
            "NO_CONFIRMED_MAP表示本规则暂无合格已确认端点，报告现有收盘/ATR和缺口，不编位置；各项事实逐项标数据日，不能把不同日期拼成同日综合。"
            "PRIOR_CONFIRMED_FALLBACK是已确认历史波段回退图，不称最新上涨段已确认；披露锚点是否后来被跌破。"
            "temporary_references的当前临时高点位置可单列参考，不并入已确认共振来源，不说正式计划已经启用。"
            "无同规格输入时保持未知。Pyxis输出是未来120交易分钟不利下探幅度，不是概率或整个次日方向。"
            "交易计划的生成、修改、确认走独立流程；本对话不得修改已确认计划、提醒或自动下单。"
            "查看已确认计划时，stocks中source=USER_MANUAL_REVIEW是用户人工选择，source=USER是人工观察；两者与manual_observations分列，不能声称计划没有人工选择。"
            "询问持仓买卖时，先读真实持仓和check_portfolio风控证据，再讨论条件与不确定性；取数失败时说明未知，不替用户决定。"
            "用户明确报告已成交且给出足够成交信息时，可按原记录规则调用record_trade。"
            "以下交易手册仅用于相关交易问题，不约束闲聊的答复格式。\n==== 手册全文 ====\n" + handbook)


TOOLS = [
    {"type":"function","function":{"name":"review_plan_minutes","description":"只读五分钟复盘。理解用户自然表达和上下文，不要求固定口令。看今天选的如何、昨天那几只今天触发没、人工复核表现、是否拦截等均可。日期指行情观察日，不是选股日；不清楚才追问。未确认名单可假设重放；实际AI与成交分列，不编造获利。",
      "parameters":{"type":"object","properties":{"message":{"type":"string"},"day":{"type":"integer","description":"行情观察日YYYYMMDD，可从对话确定；有歧义先询问"},"codes":{"type":"array","items":{"type":"string"}},"scope":{"type":"string","enum":["all","plan","manual","strong"]},"archive_id":{"type":"string"}}}}},
    {"type":"function","function":{"name":"prepare_golden_plan","description":"仅在用户要求加入计划/选择规则时，将本对话最近一批已计算黄金关注区原样生成待确认草案。未提供profile先问用默认还是自定义。profile仅支持三项所需票数、RSI上穿次数和超卖比例；不支持条件须澄清。正式确认仍由按钮完成。",
      "parameters":{"type":"object","properties":{"codes":{"type":"array","items":{"type":"string"}},"profile":{"type":"object","properties":{"required_votes":{"type":"integer"},"momentum_crosses_required":{"type":"integer"},"oversold_ratio":{"type":"number"}},"required":["required_votes","momentum_crosses_required","oversold_ratio"],"additionalProperties":False}}}}},
    {"type": "function", "function": {
        "name": "scan_sector_rules", "description": "只读扫描指定板块全部本地成员：现行Tarot一段/二段规则+冻结模型排序+V4多档黄金共振与ATR观察区。无正式入围才给最多5只可评分未过门人工复核；不改计划、不实时买入。名称不唯一返回choices。",
        "parameters": {"type": "object", "properties": {"board_name": {"type": "string"},
            "top": {"type": "integer", "minimum": 1, "maximum": 10, "description": "最多展示数，默认5"}}, "required": ["board_name"]}}},
    {"type": "function", "function": {
        "name": "stock_golden_levels", "description": "只读本地单股黄金地图：已确认波段节点、节点+0.2ATR关注区、至少两来源的共振带及临时端点。独立于选股准入，不伪造分数/风险，不改计划。",
        "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "六位A股代码"}}, "required": ["code"]}}},
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
        if name == 'review_plan_minutes':
            from minute_plan_review import review
            return review(cfg=self.cfg, **args)
        if name == 'prepare_golden_plan':
            from conversation_plans import prepare
            result=prepare(str(self.archive), **args)
            self.last_plan_result=result
            return result
        if name == 'stock_golden_levels':
            result=stock_golden_levels(**args)
            from conversation_plans import remember
            saved=remember(str(self.archive), getattr(self,'analysis_batch',str(len(self.messages))),result)
            if saved: result=dict(result,analysis_id=saved['analysis_id'])
            return result
        fn = {"stock_golden_levels": stock_golden_levels,
              "scan_sector_rules": services.scan_sector_rules,
              "model_stock_analysis": services.model_stock_analysis,
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
        from conversation_plans import tool_view
        import re
        from conversation_intent import asks_for_model, asks_for_plan, asks_for_portfolio, asks_for_portfolio_risk, model_scope, asks_for_sector_scan
        from assistant_context import bounded_messages
        self.messages.append({"role": "user", "content": user_input})
        chat_archive.append(self.archive, [self.messages[-1]])
        # Ordinary stock discussion does not silently become a model request.
        from assistant_workflow import requested_analysis_codes
        self.analysis_batch = str(len(self.messages))
        prefetched = {}
        review_results = []
        plan_results = []
        sector_results = []
        def local_scan_after_error():
            from sector_rule_scan import render_scan_results
            ready = [r for r in sector_results if r.get('status') == 'READY']
            if review_results:
                text='\n\n'.join(r.get('reply','复盘结果已保存在本地') for r in review_results)
                self.messages.append({'role':'assistant','content':text})
                chat_archive.append(self.archive,[self.messages[-1]])
                return text
            if plan_results:
                self.last_plan_result=plan_results[-1]
                return plan_results[-1].get('reply')
            if not ready:
                return None
            text = render_scan_results(ready, 'AI解读暂不可用；以下为已完成的本地规则扫描结果，不是AI研判。')
            self.messages.append({'role': 'assistant', 'content': text})
            chat_archive.append(self.archive, [self.messages[-1]])
            return text
        def delivered(text):
            from sector_rule_scan import checked_chat_reply
            checked = checked_chat_reply(text, sector_results)
            if checked != text:
                self.messages.append({'role':'assistant','content':checked})
                chat_archive.append(self.archive, [self.messages[-1]])
            return checked
        wants_golden = bool(re.search(r'黄金|分割位|共振|节点|关注区', user_input))
        codes = requested_analysis_codes(user_input) if asks_for_model(user_input) or wants_golden else []
        scope = model_scope(user_input)
        if codes:
            calls = []
            for code in codes:
                names = ['model_stock_inference'] if asks_for_model(user_input) else []
                if wants_golden or scope != 'pyxis':
                    names.append('stock_golden_levels')
                for name in names:
                    args = {'code': code, 'model': scope} if name == 'model_stock_inference' else {'code': code}
                    calls.append({'id': f'local-model-{len(self.messages)}-{code}-{name}', 'type': 'function',
                                  'function': {'name': name, 'arguments': json.dumps(args)}})
            self.messages.append({'role': 'assistant', 'tool_calls': calls})
            chat_archive.append(self.archive, [self.messages[-1]])
            for call in calls:
                name = call['function']['name']
                args = json.loads(call['function']['arguments'])
                if on_tool:
                    on_tool(name, args)
                try:
                    result = self._dispatch(name, args)
                except Exception as exc:
                    result = {'code': args['code'], 'status': 'UNAVAILABLE', 'reason': type(exc).__name__}
                prefetched[(args['code'], scope if name == 'model_stock_inference' else 'golden')] = result
                self.messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                      'content': chat_archive.tool_content(tool_view(result), self.archive)})
                chat_archive.append(self.archive, [self.messages[-1]])
        from minute_plan_review import requested
        for name, wanted in (('review_plan_minutes', requested(user_input)),
                             ('confirmed_plan', asks_for_plan(user_input)),
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
                result = self._dispatch(name, {'message': user_input} if name=='review_plan_minutes' else {})
                if name=='review_plan_minutes':
                    review_results.append(result)
                    prefetched[('review',user_input)]=result
            except Exception as exc:
                result = {'status': 'UNAVAILABLE', 'reason': type(exc).__name__}
                if name in ('review_plan_minutes','confirmed_plan'):
                    result.update(tool=name,detail=str(exc).replace('\n',' ')[:240])
            self.messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                  'content': chat_archive.tool_content(tool_view(result), self.archive)})
            chat_archive.append(self.archive, [self.messages[-1]])
        from deepseek_api import complete
        for _ in range(self.max_rounds):
            watch_edit = bool(re.search(r'加入观察名单|添加观察|登记持仓|移除观察|删除观察', user_input))
            trade_report = bool(re.search(r'成交|买入了|卖出了|清仓了|我买了|我卖了', user_input))
            active_tools = [t for t in TOOLS if t['function']['name'] not in
                            ('scan_sector', 'score_stocks', 'model_stock_analysis')
                            and (asks_for_sector_scan(user_input) or t['function']['name'] != 'scan_sector_rules')
                            and (asks_for_model(user_input) or t['function']['name'] != 'model_stock_inference')
                            and (asks_for_model(user_input) or wants_golden or t['function']['name'] != 'stock_golden_levels')
                            and (watch_edit or t['function']['name'] not in
                                 ('wl_add_watch', 'wl_add_position', 'wl_remove'))
                            and (trade_report or t['function']['name'] != 'record_trade')]
            try:
                rsp = complete(self.cfg, bounded_messages(self.messages[:1], self.messages[1:]), tools=active_tools)
            except ValueError:
                local = local_scan_after_error()
                if local is not None:
                    return local
                raise
            msg = rsp.choices[0].message
            self.messages.append(msg.model_dump(exclude_none=True))
            chat_archive.append(self.archive, [self.messages[-1]])
            if not msg.tool_calls:
                self.last_plan_result = plan_results[-1] if plan_results else None
                return delivered((msg.content or '').strip()) or 'DeepSeek本次未返回正文；没有自动重试，也没有修改计划。'
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                if on_tool:
                    on_tool(tc.function.name, args)
                try:
                    if tc.function.name == 'review_plan_minutes' and ('review',user_input) in prefetched and not any(k in args for k in ('day','codes','scope','archive_id')):
                        result=prefetched[('review',user_input)]
                    elif tc.function.name == 'model_stock_inference' and (args.get('code'), args.get('model', 'both')) in prefetched:
                        result = prefetched[(args['code'], args.get('model', 'both'))]
                    elif tc.function.name == 'stock_golden_levels' and (args.get('code'), 'golden') in prefetched:
                        result = prefetched[(args['code'], 'golden')]
                    else:
                        result = self._dispatch(tc.function.name, args)
                except Exception as e:
                    result = {"error": f"工具执行失败: {e}"}
                if tc.function.name == 'review_plan_minutes':review_results.append(result)
                if tc.function.name == 'prepare_golden_plan':plan_results.append(result)
                if tc.function.name == 'scan_sector_rules':sector_results.append(result)
                self.messages.append({"role": "tool", "tool_call_id": tc.id,
                                      "content": chat_archive.tool_content(tool_view(result), self.archive)})
                chat_archive.append(self.archive, [self.messages[-1]])
        # Finish the current discussion from collected facts instead of a local
        # round-limit error. This synthesis has no tools or mutation authority.
        try:
            final = complete(self.cfg, bounded_messages(self.messages[:1], self.messages[1:]))
        except ValueError:
            local = local_scan_after_error()
            if local is not None:
                return local
            raise
        content = (final.choices[0].message.content or '').strip()
        self.messages.append({'role':'assistant','content':content})
        chat_archive.append(self.archive, [self.messages[-1]])
        return delivered(content) or '本次未返回总结正文；已取得的工具事实仍保存在本地，没有自动重试。'
