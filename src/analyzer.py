# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - AI分析层
===================================

职责：
1. 封装 Gemini API 调用逻辑
2. 利用 Google Search Grounding 获取实时新闻
3. 结合技术面和消息面生成分析报告
4. 新增：午盘选股策略（11:00前筛选符合条件的股票）
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from json_repair import repair_json
import pandas as pd  # 新增：导入pandas

from src.config import get_config

logger = logging.getLogger(__name__)


# 股票名称映射（常见股票）
STOCK_NAME_MAP = {
    # === A股 ===
    '600519': '贵州茅台',
    '000001': '平安银行',
    '300750': '宁德时代',
    '002594': '比亚迪',
    '600036': '招商银行',
    '601318': '中国平安',
    '000858': '五粮液',
    '600276': '恒瑞医药',
    '601012': '隆基绿能',
    '002475': '立讯精密',
    '300059': '东方财富',
    '002415': '海康威视',
    '600900': '长江电力',
    '601166': '兴业银行',
    '600028': '中国石化',

    # === 美股 ===
    'AAPL': '苹果',
    'TSLA': '特斯拉',
    'MSFT': '微软',
    'GOOGL': '谷歌A',
    'GOOG': '谷歌C',
    'AMZN': '亚马逊',
    'NVDA': '英伟达',
    'META': 'Meta',
    'AMD': 'AMD',
    'INTC': '英特尔',
    'BABA': '阿里巴巴',
    'PDD': '拼多多',
    'JD': '京东',
    'BIDU': '百度',
    'NIO': '蔚来',
    'XPEV': '小鹏汽车',
    'LI': '理想汽车',
    'COIN': 'Coinbase',
    'MSTR': 'MicroStrategy',

    # === 港股 (5位数字) ===
    '00700': '腾讯控股',
    '03690': '美团',
    '01810': '小米集团',
    '09988': '阿里巴巴',
    '09618': '京东集团',
    '09888': '百度集团',
    '01024': '快手',
    '00981': '中芯国际',
    '02015': '理想汽车',
    '09868': '小鹏汽车',
    '00005': '汇丰控股',
    '01299': '友邦保险',
    '00941': '中国移动',
    '00883': '中国海洋石油',
}


def get_stock_name_multi_source(
    stock_code: str,
    context: Optional[Dict] = None,
    data_manager = None
) -> str:
    """
    多来源获取股票中文名称

    获取策略（按优先级）：
    1. 从传入的 context 中获取（realtime 数据）
    2. 从静态映射表 STOCK_NAME_MAP 获取
    3. 从 DataFetcherManager 获取（各数据源）
    4. 返回默认名称（股票+代码）

    Args:
        stock_code: 股票代码
        context: 分析上下文（可选）
        data_manager: DataFetcherManager 实例（可选）

    Returns:
        股票中文名称
    """
    # 1. 从上下文获取（实时行情数据）
    if context:
        # 优先从 stock_name 字段获取
        if context.get('stock_name'):
            name = context['stock_name']
            if name and not name.startswith('股票'):
                return name

        # 其次从 realtime 数据获取
        if 'realtime' in context and context['realtime'].get('name'):
            return context['realtime']['name']

    # 2. 从静态映射表获取
    if stock_code in STOCK_NAME_MAP:
        return STOCK_NAME_MAP[stock_code]

    # 3. 从数据源获取
    if data_manager is None:
        try:
            from data_provider.base import DataFetcherManager
            data_manager = DataFetcherManager()
        except Exception as e:
            logger.debug(f"无法初始化 DataFetcherManager: {e}")

    if data_manager:
        try:
            name = data_manager.get_stock_name(stock_code)
            if name:
                # 更新缓存
                STOCK_NAME_MAP[stock_code] = name
                return name
        except Exception as e:
            logger.debug(f"从数据源获取股票名称失败: {e}")

    # 4. 返回默认名称
    return f'股票{stock_code}'


@dataclass
class AnalysisResult:
    """
    AI 分析结果数据类 - 决策仪表盘版

    封装 Gemini 返回的分析结果，包含决策仪表盘和详细分析
    """
    code: str
    name: str

    # ========== 核心指标 ==========
    sentiment_score: int  # 综合评分 0-100 (>70强烈看多, >60看多, 40-60震荡, <40看空)
    trend_prediction: str  # 趋势预测：强烈看多/看多/震荡/看空/强烈看空
    operation_advice: str  # 操作建议：买入/加仓/持有/减仓/卖出/观望
    decision_type: str = "hold"  # 决策类型：buy/hold/sell（用于统计）
    confidence_level: str = "中"  # 置信度：高/中/低

    # ========== 决策仪表盘 (新增) ==========
    dashboard: Optional[Dict[str, Any]] = None  # 完整的决策仪表盘数据

    # ========== 走势分析 ==========
    trend_analysis: str = ""  # 走势形态分析（支撑位、压力位、趋势线等）
    short_term_outlook: str = ""  # 短期展望（1-3日）
    medium_term_outlook: str = ""  # 中期展望（1-2周）

    # ========== 技术面分析 ==========
    technical_analysis: str = ""  # 技术指标综合分析
    ma_analysis: str = ""  # 均线分析（多头/空头排列，金叉/死叉等）
    volume_analysis: str = ""  # 量能分析（放量/缩量，主力动向等）
    pattern_analysis: str = ""  # K线形态分析

    # ========== 基本面分析 ==========
    fundamental_analysis: str = ""  # 基本面综合分析
    sector_position: str = ""  # 板块地位和行业趋势
    company_highlights: str = ""  # 公司亮点/风险点

    # ========== 情绪面/消息面分析 ==========
    news_summary: str = ""  # 近期重要新闻/公告摘要
    market_sentiment: str = ""  # 市场情绪分析
    hot_topics: str = ""  # 相关热点话题

    # ========== 综合分析 ==========
    analysis_summary: str = ""  # 综合分析摘要
    key_points: str = ""  # 核心看点（3-5个要点）
    risk_warning: str = ""  # 风险提示
    buy_reason: str = ""  # 买入/卖出理由

    # ========== 元数据 ==========
    market_snapshot: Optional[Dict[str, Any]] = None  # 当日行情快照（展示用）
    raw_response: Optional[str] = None  # 原始响应（调试用）
    search_performed: bool = False  # 是否执行了联网搜索
    data_sources: str = ""  # 数据来源说明
    success: bool = True
    error_message: Optional[str] = None

    # ========== 价格数据（分析时快照）==========
    current_price: Optional[float] = None  # 分析时的股价
    change_pct: Optional[float] = None     # 分析时的涨跌幅(%)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'code': self.code,
            'name': self.name,
            'sentiment_score': self.sentiment_score,
            'trend_prediction': self.trend_prediction,
            'operation_advice': self.operation_advice,
            'decision_type': self.decision_type,
            'confidence_level': self.confidence_level,
            'dashboard': self.dashboard,  # 决策仪表盘数据
            'trend_analysis': self.trend_analysis,
            'short_term_outlook': self.short_term_outlook,
            'medium_term_outlook': self.medium_term_outlook,
            'technical_analysis': self.technical_analysis,
            'ma_analysis': self.ma_analysis,
            'volume_analysis': self.volume_analysis,
            'pattern_analysis': self.pattern_analysis,
            'fundamental_analysis': self.fundamental_analysis,
            'sector_position': self.sector_position,
            'company_highlights': self.company_highlights,
            'news_summary': self.news_summary,
            'market_sentiment': self.market_sentiment,
            'hot_topics': self.hot_topics,
            'analysis_summary': self.analysis_summary,
            'key_points': self.key_points,
            'risk_warning': self.risk_warning,
            'buy_reason': self.buy_reason,
            'market_snapshot': self.market_snapshot,
            'search_performed': self.search_performed,
            'success': self.success,
            'error_message': self.error_message,
            'current_price': self.current_price,
            'change_pct': self.change_pct,
        }

    def get_core_conclusion(self) -> str:
        """获取核心结论（一句话）"""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            return self.dashboard['core_conclusion'].get('one_sentence', self.analysis_summary)
        return self.analysis_summary

    def get_position_advice(self, has_position: bool = False) -> str:
        """获取持仓建议"""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            pos_advice = self.dashboard['core_conclusion'].get('position_advice', {})
            if has_position:
                return pos_advice.get('has_position', self.operation_advice)
            return pos_advice.get('no_position', self.operation_advice)
        return self.operation_advice

    def get_sniper_points(self) -> Dict[str, str]:
        """获取狙击点位"""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('sniper_points', {})
        return {}

    def get_checklist(self) -> List[str]:
        """获取检查清单"""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('action_checklist', [])
        return []

    def get_risk_alerts(self) -> List[str]:
        """获取风险警报"""
        if self.dashboard and 'intelligence' in self.dashboard:
            return self.dashboard['intelligence'].get('risk_alerts', [])
        return []

    def get_emoji(self) -> str:
        """根据操作建议返回对应 emoji"""
        emoji_map = {
            '买入': '🟢',
            '加仓': '🟢',
            '强烈买入': '💚',
            '持有': '🟡',
            '观望': '⚪',
            '减仓': '🟠',
            '卖出': '🔴',
            '强烈卖出': '❌',
        }
        advice = self.operation_advice or ''
        # Direct match first
        if advice in emoji_map:
            return emoji_map[advice]
        # Handle compound advice like "卖出/观望" — use the first part
        for part in advice.replace('/', '|').split('|'):
            part = part.strip()
            if part in emoji_map:
                return emoji_map[part]
        # Score-based fallback
        score = self.sentiment_score
        if score >= 80:
            return '💚'
        elif score >= 65:
            return '🟢'
        elif score >= 55:
            return '🟡'
        elif score >= 45:
            return '⚪'
        elif score >= 35:
            return '🟠'
        else:
            return '🔴'

    def get_confidence_stars(self) -> str:
        """返回置信度星级"""
        star_map = {'高': '⭐⭐⭐', '中': '⭐⭐', '低': '⭐'}
        return star_map.get(self.confidence_level, '⭐⭐')


# ========== 新增：午盘选股结果数据类 ==========
@dataclass
class MiddayStockPickResult:
    """午盘选股结果数据类"""
    code: str
    name: str
    score: float  # 选股评分（0-100）
    reason: List[str]  # 入选理由
    reject_reason: List[str] = None  # 落选理由（如果有）
    technical_data: Dict[str, Any] = None  # 入选时的技术数据
    buy_signal: str = "观望"  # 买入信号：强烈买入/买入/观望/回避
    buy_price: Optional[float] = None  # 建议买入价
    stop_loss_price: Optional[float] = None  # 止损价

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "score": self.score,
            "reason": self.reason,
            "reject_reason": self.reject_reason or [],
            "technical_data": self.technical_data or {},
            "buy_signal": self.buy_signal,
            "buy_price": self.buy_price,
            "stop_loss_price": self.stop_loss_price
        }


class GeminiAnalyzer:
    """
    Gemini AI 分析器

    职责：
    1. 调用 Google Gemini API 进行股票分析
    2. 结合预先搜索的新闻和技术面数据生成分析报告
    3. 解析 AI 返回的 JSON 格式结果
    4. 新增：午盘选股策略（11:00前筛选符合条件的股票）

    使用方式：
        analyzer = GeminiAnalyzer()
        result = analyzer.analyze(context, news_context)
        # 新增：午盘选股
        pick_results = analyzer.midday_stock_pick(stock_contexts)
    """

    # ========================================
    # 系统提示词 - 决策仪表盘 v2.0
    # ========================================
    # 输出格式升级：从简单信号升级为决策仪表盘
    # 核心模块：核心结论 + 数据透视 + 舆情情报 + 作战计划
    # ========================================

    SYSTEM_PROMPT = """你是一位专注于趋势交易的 A 股投资分析师，负责生成专业的【决策仪表盘】分析报告。

## 核心交易理念（必须严格遵守）

### 1. 严进策略（不追高）
- **绝对不追高**：当股价偏离 MA5 超过 5% 时，坚决不买入
- **乖离率公式**：(现价 - MA5) / MA5 × 100%
- 乖离率 < 2%：最佳买点区间
- 乖离率 2-5%：可小仓介入
- 乖离率 > 5%：严禁追高！直接判定为"观望"

### 2. 趋势交易（顺势而为）
- **多头排列必须条件**：MA5 > MA10 > MA20
- 只做多头排列的股票，空头排列坚决不碰
- 均线发散上行优于均线粘合
- 趋势强度判断：看均线间距是否在扩大

### 3. 效率优先（筹码结构）
- 关注筹码集中度：90%集中度 < 15% 表示筹码集中
- 获利比例分析：70-90% 获利盘时需警惕获利回吐
- 平均成本与现价关系：现价高于平均成本 5-15% 为健康

### 4. 买点偏好（回踩支撑）
- **最佳买点**：缩量回踩 MA5 获得支撑
- **次优买点**：回踩 MA10 获得支撑
- **观望情况**：跌破 MA20 时观望

### 5. 风险排查重点
- 减持公告（股东、高管减持）
- 业绩预亏/大幅下滑
- 监管处罚/立案调查
- 行业政策利空
- 大额解禁

### 6. 估值关注（PE/PB）
- 分析时请关注市盈率（PE）是否合理
- PE 明显偏高时（如远超行业平均或历史均值），需在风险点中说明
- 高成长股可适当容忍较高 PE，但需有业绩支撑

### 7. 强势趋势股放宽
- 强势趋势股（多头排列且趋势强度高、量能配合）可适当放宽乖离率要求
- 此类股票可轻仓追踪，但仍需设置止损，不盲目追高

## 输出格式：决策仪表盘 JSON

请严格按照以下 JSON 格式输出，这是一个完整的【决策仪表盘】："""

    JSON_FORMAT_PROMPT = """
输出JSON结构如下：
{
    "stock_name": "股票名称",
    "sentiment_score": 0-100,
    "trend_prediction": "强烈看多/看多/震荡/看空/强烈看空",
    "operation_advice": "买入/加仓/持有/减仓/卖出/观望",
    "decision_type": "buy/hold/sell",
    "confidence_level": "高/中/低",
    "dashboard": {
        "core_conclusion": {
            "one_sentence": "一句话结论",
            "signal_type": "🟢买入/🟡观望/🔴卖出/⚠️风险",
            "time_sensitivity": "立即行动/今日内/本周内/不急",
            "position_advice": {
                "no_position": "空仓建议",
                "has_position": "持仓建议"
            }
        },
        "data_perspective": {
            "trend_status": {
                "ma_alignment": "均线描述",
                "is_bullish": true/false,
                "trend_score": 0-100
            },
            "price_position": {
                "current_price": 0,
                "ma5": 0,
                "ma10": 0,
                "ma20": 0,
                "bias_ma5": 0,
                "bias_status": "安全/警戒/危险",
                "support_level": 0,
                "resistance_level": 0
            },
            "volume_analysis": {
                "volume_ratio": 0,
                "volume_status": "放量/缩量/平量",
                "turnover_rate": 0,
                "volume_meaning": ""
            },
            "chip_structure": {
                "profit_ratio": 0,
                "avg_cost": 0,
                "concentration": 0,
                "chip_health": "健康/一般/警惕"
            }
        },
        "intelligence": {
            "latest_news": "",
            "risk_alerts": [],
            "positive_catalysts": [],
            "earnings_outlook": "",
            "sentiment_summary": ""
        },
        "battle_plan": {
            "sniper_points": {
                "ideal_buy": "",
                "secondary_buy": "",
                "stop_loss": "",
                "take_profit": ""
            },
            "position_strategy": {
                "suggested_position": "",
                "entry_plan": "",
                "risk_control": ""
            },
            "action_checklist": []
        }
    },
    "analysis_summary": "",
    "key_points": "",
    "risk_warning": "",
    "buy_reason": "",
    "trend_analysis": "",
    "short_term_outlook": "",
    "medium_term_outlook": "",
    "technical_analysis": "",
    "ma_analysis": "",
    "volume_analysis": "",
    "pattern_analysis": "",
    "fundamental_analysis": "",
    "sector_position": "",
    "company_highlights": "",
    "news_summary": "",
    "market_sentiment": "",
    "hot_topics": "",
    "search_performed": true/false,
    "data_sources": ""
}
"""

    def __init__(self):
        self.config = get_config()
        self.model = None

    def analyze(self, context: Dict, news_context: Optional[List] = None) -> AnalysisResult:
        """
        执行AI分析
        """
        stock_code = context.get('code', '')
        stock_name = get_stock_name_multi_source(stock_code, context=context)

        try:
            # === 这里接入你的Gemini调用逻辑 ===
            # response = self.model.generate_content(...)
            # raw = response.text

            raw = "{}"  # 模拟返回
            fixed_json = repair_json(raw)
            data = json.loads(fixed_json)

            # 构造结果
            res = AnalysisResult(
                code=stock_code,
                name=stock_name,
                sentiment_score=data.get('sentiment_score', 50),
                trend_prediction=data.get('trend_prediction', '震荡'),
                operation_advice=data.get('operation_advice', '观望'),
                decision_type=data.get('decision_type', 'hold'),
                confidence_level=data.get('confidence_level', '中'),
                dashboard=data.get('dashboard'),
                analysis_summary=data.get('analysis_summary', ''),
                key_points=data.get('key_points', ''),
                risk_warning=data.get('risk_warning', ''),
                buy_reason=data.get('buy_reason', ''),
                raw_response=raw,
                search_performed=bool(news_context),
            )
            return res

        except Exception as e:
            logger.error(f"分析失败: {e}")
            return AnalysisResult(
                code=stock_code,
                name=stock_name,
                success=False,
                error_message=str(e)
            )

    def midday_stock_pick(self, stock_contexts: List[Dict]) -> List[MiddayStockPickResult]:
        """
        午盘选股（11:00前执行）
        基于技术面多头、乖离率、量能、筹码筛选
        """
        results = []
        for ctx in stock_contexts:
            code = ctx.get('code', '')
            name = get_stock_name_multi_source(code, context=ctx)
            realtime = ctx.get('realtime', {})
            tech = ctx.get('technical', {})

            # 简单打分示例
            score = 50
            reason = []
            buy_signal = "观望"

            # 多头排列
            ma5 = tech.get('ma5', 0)
            ma10 = tech.get('ma10', 0)
            ma20 = tech.get('ma20', 0)
            is_bullish = ma5 > ma10 > ma20

            if is_bullish:
                score += 25
                reason.append("均线多头排列")

            # 乖离率
            price = realtime.get('price', 0)
            if price and ma5:
                bias = (price - ma5) / ma5 * 100
                if bias < 2:
                    score += 15
                    reason.append("乖离率低，适合低吸")
                elif bias > 5:
                    score -= 20

            # 量能
            vr = tech.get('volume_ratio', 1)
            if 0.8 < vr < 1.5:
                score += 10
                reason.append("量能健康")

            # 最终信号
            if score >= 75:
                buy_signal = "强烈买入"
            elif score >= 60:
                buy_signal = "买入"
            elif score <= 30:
                buy_signal = "回避"

            results.append(MiddayStockPickResult(
                code=code,
                name=name,
                score=score,
                reason=reason,
                buy_signal=buy_signal,
                technical_data=tech
            ))

        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results
