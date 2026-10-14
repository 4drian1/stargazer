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
    return ("你是 Stargazer，一个 A 股交易助手。以下是你的完整交易模型手册，是你的行为准则与知识体系，"
            "所有判断必须严格遵守手册纪律（四层漏斗、三层止损、轮动五阶段、回调框架、风险旗标）。\n"
            "工作规则：\n"
            "1. 需要数据时调用工具，不要凭记忆报价；工具返回的数据是事实，你的解读是判断，两者要区分。\n"
            "2. 交付完整筛选路径：最终名单+被剔除者+剔除原因；标注哪些是硬数据、哪些是主观判断。\n"
            "3. 不替用户做仓位决定，不给保本承诺；止损止盈信号严格按手册执行，不含糊。\n"
            "4. 回答用中文，简洁直接，先说结论。\n"
            "5. 用户问新闻、美股、晨报、'今天怎么看'这类问题时，必须先调 us_market / news_digest / market_flow "
            "拿到数据再分析，分析要落到对持仓和观察名单的具体影响上，不要只罗列消息。\n"
            "6. 需要求证题材真伪（照妖镜）、找投行/券商研报观点、查公司公告或海外深度消息时，用 research_search；"
            "news_digest 只是快讯流，两者互补。\n"
            "7. 分析大盘/盘面/情绪时必须先调 market_overview；分析某个板块强不强时必须先调 sector_detail 看龙头和广度，"
            "需要龙头个股细节再调 stock_check。数据不够就明说缺什么，不要凭印象编盘面。\n"
            "8. 用户问龙虎榜、主力资金在哪打架时调 lhb_today；分析连板股/涨停股的接力环境、"
            "或想知道某只异动股是谁在买（机构/游资/北向/散户）时调 lhb_stock 看席位。"
            "龙虎榜是T日傍晚发布的数据，盘中查到的都是上一交易日，回答时说明数据日期。\n"
            "9. 解读任何板块新闻/题材消息时必须做7天回溯：用 research_search（days=7）查同一板块近7天内"
            "是否已有过同类消息，并说明当时市场的实际反应（涨没涨、谁涨了、持续性如何）。"
            "结论要区分：'首次出现的新催化'与'7天内已炒过的旧消息'，后者要提示边际效应递减。\n"
            "10. 分析个股是否值得建仓时必须输出「标签性感度评分卡」：时效性(消息是否新鲜)0-5分、"
            "映射度(公司业务与题材的真实关联)0-5分、稀缺性(A股标的是否稀缺)0-5分，"
            "三项合计<10分时明确提示'标签不够性感，按纪律不建议建仓'。\n"
            "11. 用户明确说明自己买入或卖出了某只股票（例如'我按策略买了安彩高科'、'今天清仓了XX'、'加仓YY到半仓'）时，"
            "必须调用 record_trade 工具把真实成交记入交易数据库（这是用户日后回测验证的依据），并在回复里向用户确认已记录、"
            "不要只口头答应。记录要准确：区分买/卖、填对代码名称与成交价；用户没说日期默认今天，没说股数可不填。\n\n"
            "==== 手册全文 ====\n" + handbook)


TOOLS = [
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
        fn = {"scan_sector": services.scan_sector, "score_stocks": services.score_stocks,
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
        from assistant_context import bounded_messages
        self.messages.append({"role": "user", "content": user_input})
        chat_archive.append(self.archive, [self.messages[-1]])
        from deepseek_api import complete
        for _ in range(self.max_rounds):
            active_tools = [t for t in TOOLS if t['function']['name'] not in ('scan_sector', 'score_stocks')]
            rsp = complete(self.cfg, bounded_messages(self.messages[:1], self.messages[1:]), tools=active_tools)
            msg = rsp.choices[0].message
            self.messages.append(msg.model_dump(exclude_none=True))
            chat_archive.append(self.archive, [self.messages[-1]])
            if not msg.tool_calls:
                return msg.content
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                if on_tool:
                    on_tool(tc.function.name, args)
                try:
                    result = self._dispatch(tc.function.name, args)
                except Exception as e:
                    result = {"error": f"工具执行失败: {e}"}
                self.messages.append({"role": "tool", "tool_call_id": tc.id,
                                      "content": chat_archive.tool_content(result, self.archive)})
                chat_archive.append(self.archive, [self.messages[-1]])
        return "（工具调用轮数已达上限，请缩小问题范围再问）"
