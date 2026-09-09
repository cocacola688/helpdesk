"""
亮点：端到端意图识别

三路融合策略：
  1. LLM 语义理解（权重 70%）—— 主力，理解复杂语义和上下文
  2. Embedding 向量相似度（权重 20%）—— 快速匹配常见表达
  3. 关键词模式匹配（权重 10%）—— 零延迟兜底

三路结果通过加权投票合并，置信度低于阈值时降级为 OTHER。
LLM 和 Embedding 并行调用，不串行等待。
"""
import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    # 设备类
    DEVICE_FAILURE = "device_failure"      # 设备故障（电脑/手机/打印机等）
    DEVICE_SETUP = "device_setup"          # 设备配置
    SOFTWARE_INSTALL = "software_install"  # 软件安装/更新
    HARDWARE_REQUEST = "hardware_request"  # 硬件申请

    # 权限类
    PERMISSION_REQUEST = "permission_request"  # 权限申请
    ACCOUNT_ISSUE = "account_issue"        # 账号问题
    VPN_ACCESS = "vpn_access"              # VPN访问

    # 网络类
    NETWORK_ISSUE = "network_issue"        # 网络问题
    WIFI_PROBLEM = "wifi_problem"          # Wi-Fi问题
    NETWORK_SLOW = "network_slow"          # 网络慢

    # 通用类
    QUERY = "query"                        # 一般查询
    CONTACT_INFO = "contact_info"          # 联系方式
    GREETING = "greeting"                  # 问候
    COMPLAINT = "complaint"                # 投诉建议
    IT_ESCALATION = "it_escalation"        # 转IT工单
    FEEDBACK = "feedback"                  # 正面反馈
    OTHER = "other"


class UrgencyLevel(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    intent:     IntentCategory
    confidence: float
    urgency:    UrgencyLevel
    intent_group: str
    entities:   Dict[str, List[str]]   # 从消息中提取的实体
    reasoning:  str
    latency_ms: float
    source_scores: Dict[str, float] = field(default_factory=dict)


# ── Few-shot 模板（同时用于 LLM 示例和 Embedding 匹配）────────────────────────
_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.DEVICE_FAILURE: [
        "我的电脑开不了机",
        "打印机一直卡纸",
        "手机连不上公司Wi-Fi",
        "显示器黑屏了",
        "电脑蓝屏了显示错误码0x80070005",
        "系统崩溃了重启不了",
        "电脑死机了一直转圈",
        "笔记本键盘失灵",
    ],
    IntentCategory.DEVICE_SETUP: ["新电脑怎么配置邮箱？", "如何设置双显示器？", "打印机驱动怎么装？"],
    IntentCategory.SOFTWARE_INSTALL: ["怎么安装Office？", "需要升级系统吗？", "软件提示更新怎么办？"],
    IntentCategory.HARDWARE_REQUEST: ["我需要申请新电脑", "能给我配个鼠标吗？", "办公桌没有网线接口"],
    IntentCategory.PERMISSION_REQUEST: [
        "申请文件夹访问权限",
        "需要开通ERP系统",
        "怎么申请管理员权限？",
        "我需要申请VPN权限",
        "帮我开通系统访问权限",
        "申请数据库访问权限",
    ],
    IntentCategory.ACCOUNT_ISSUE: ["忘记密码了", "账号被锁定", "登录一直提示错误"],
    IntentCategory.VPN_ACCESS: [
        "VPN连不上",
        "在家怎么访问公司系统？",
        "VPN断开连接了",
        "远程登录失败",
        "VPN客户端报错",
    ],
    IntentCategory.NETWORK_ISSUE: ["网络断了", "无法访问内网", "网页打不开"],
    IntentCategory.WIFI_PROBLEM: ["Wi-Fi信号弱", "连上Wi-Fi但上不了网", "会议室Wi-Fi密码是多少？"],
    IntentCategory.NETWORK_SLOW: ["网速特别慢", "下载速度很慢", "视频会议卡顿"],
    IntentCategory.QUERY: ["请问一下", "我想了解", "能帮我查一下吗"],
    IntentCategory.CONTACT_INFO: ["IT部门电话是多少？", "怎么联系技术支持？", "谁负责网络问题？"],
    IntentCategory.GREETING: ["你好", "嗨，有人吗", "早上好"],
    IntentCategory.COMPLAINT: ["IT响应太慢了", "这个系统太难用", "为什么总是出问题"],
    IntentCategory.IT_ESCALATION: ["帮我转人工", "需要专家处理", "创建工单"],
    IntentCategory.FEEDBACK: ["问题解决了，谢谢！", "服务很好", "很满意"],
}

_SPECIFIC_INTENTS = {
    IntentCategory.DEVICE_FAILURE,
    IntentCategory.DEVICE_SETUP,
    IntentCategory.SOFTWARE_INSTALL,
    IntentCategory.HARDWARE_REQUEST,
    IntentCategory.PERMISSION_REQUEST,
    IntentCategory.ACCOUNT_ISSUE,
    IntentCategory.VPN_ACCESS,
    IntentCategory.NETWORK_ISSUE,
    IntentCategory.WIFI_PROBLEM,
    IntentCategory.NETWORK_SLOW,
    IntentCategory.CONTACT_INFO,
    IntentCategory.IT_ESCALATION,
}

_GENERIC_INTENTS = {
    IntentCategory.QUERY,
    IntentCategory.GREETING,
    IntentCategory.COMPLAINT,
    IntentCategory.FEEDBACK,
}

_INTENT_GROUPS: Dict[IntentCategory, IntentCategory] = {
    IntentCategory.DEVICE_FAILURE: IntentCategory.QUERY,
    IntentCategory.DEVICE_SETUP: IntentCategory.QUERY,
    IntentCategory.SOFTWARE_INSTALL: IntentCategory.QUERY,
    IntentCategory.HARDWARE_REQUEST: IntentCategory.QUERY,
    IntentCategory.PERMISSION_REQUEST: IntentCategory.QUERY,
    IntentCategory.ACCOUNT_ISSUE: IntentCategory.QUERY,
    IntentCategory.VPN_ACCESS: IntentCategory.QUERY,
    IntentCategory.NETWORK_ISSUE: IntentCategory.QUERY,
    IntentCategory.WIFI_PROBLEM: IntentCategory.QUERY,
    IntentCategory.NETWORK_SLOW: IntentCategory.QUERY,
    IntentCategory.CONTACT_INFO: IntentCategory.QUERY,
    IntentCategory.IT_ESCALATION: IntentCategory.IT_ESCALATION,
}

# 紧急关键词
_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: ["紧急", "急", "系统崩溃", "数据丢失", "无法工作", "业务中断", "生产故障"],
    UrgencyLevel.HIGH:     ["今天必须", "马上要", "尽快", "会议前", "客户演示"],
    UrgencyLevel.MEDIUM:   ["明天", "这周", "本周", "下周"],
}


def _cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，不依赖 numpy。"""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """
    端到端意图识别器

    核心功能：通过三路融合策略识别用户意图
    1. LLM 语义理解（70% 权重）- 处理复杂语义和上下文
    2. Embedding 向量匹配（20% 权重）- 快速匹配常见表达
    3. 关键词模式匹配（10% 权重）- 零延迟兜底策略

    设计亮点：
    - 无本地模型依赖，所有 AI 能力通过 Anthropic API
    - 懒加载 + 缓存机制，提升响应速度
    - 三路并行执行，不串行等待
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        confidence_threshold: float = 0.5,
    ):
        """
        初始化意图识别器

        参数:
            api_key: Anthropic API 密钥
            base_url: 可选的自定义 API 基础 URL（用于代理或自托管）
            model: LLM 模型名称
            confidence_threshold: 置信度阈值，低于此值返回 OTHER 意图
        """
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client    = AsyncAnthropic(**kwargs)
        self.model     = model
        self.threshold = confidence_threshold

        # Embedding 功能开关
        # 本地字符 n-gram 向量始终可用；如果未来客户端暴露 embeddings 资源，
        # _embed_text 会优先尝试远端向量，否则自动回退本地向量。
        self._embedding_enabled = True

        # 模板 Embedding 缓存：每个意图类别对应多个模板向量
        self._tpl_embeddings: Dict[IntentCategory, List[List[float]]] = {}

        # 识别结果缓存：避免重复识别相同或相似的消息
        self._cache: Dict[str, IntentResult] = {}

        # 缓存统计：用于监控缓存效果
        self.cache_hits   = 0
        self.cache_misses = 0

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        """
        识别用户意图（主入口）

        识别流程：
        1. 检查缓存，命中则直接返回
        2. 并行执行三路识别：
           - LLM 语义理解（异步）
           - Embedding 向量匹配（异步）
           - 关键词模式匹配（同步）
        3. 加权投票合并三路结果
        4. 提取实体和判断紧急度
        5. 缓存结果并返回

        参数:
            message: 用户消息文本
            history: 对话历史，格式 [{"role": "user"/"assistant", "content": "..."}]

        返回:
            IntentResult: 包含意图、置信度、紧急度、实体等完整信息
        """
        # 1. 缓存检查：避免重复识别
        key = self._cache_key(message, history)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        # 2. 三路并行识别
        # LLM 和 Embedding 异步并行执行，模式匹配同步执行（速度快）
        llm_task = asyncio.create_task(self._llm_recognize(message, history))
        emb_task = asyncio.create_task(self._embedding_recognize(message)) if self._embedding_enabled else None
        pat      = self._pattern_recognize(message)  # 同步执行，立即返回

        # 等待异步任务完成
        if emb_task:
            llm, emb = await asyncio.gather(llm_task, emb_task)
        else:
            llm = await llm_task
            emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}

        # 3. 加权投票：融合三路结果
        intent, confidence, source_scores = self._vote(llm, emb, pat)

        # 4. 提取结构化信息
        entities = self._extract_entities(message)  # 提取订单号、金额等实体
        urgency  = self._urgency(message, intent)   # 判断紧急程度

        # 5. 构建结果对象
        result = IntentResult(
            intent=intent,
            confidence=confidence,
            urgency=urgency,
            intent_group=self._intent_group(intent),  # 将细粒度意图归类到大类
            entities=entities,
            reasoning=llm.get("reasoning", ""),       # LLM 的推理说明
            latency_ms=(time.monotonic() - t0) * 1000,
            source_scores=source_scores,
        )

        # 6. LRU 缓存：满了就删除最旧的一半
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
        self._cache[key] = result
        return result

    def learn(self, message: str, correct: IntentCategory) -> None:
        """在线学习：将纠正样本加入模板，清除对应 Embedding 缓存。"""
        tpls = _TEMPLATES.setdefault(correct, [])
        if message not in tpls:
            tpls.append(message)
            self._tpl_embeddings.pop(correct, None)  # 下次重新计算
            self._cache.clear()  # 模板更新后旧缓存可能对应过时结果
            logger.info(f"学习新样本 → {correct.value}: {message[:40]}")

    # ── 三路识别策略 ──────────────────────────────────────────────────────────

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """
        策略 1：LLM 语义理解（Few-shot + 上下文）

        优点：理解复杂语义、处理模糊表达、考虑上下文
        缺点：延迟较高（~500ms）、成本较高

        通过 Few-shot 示例教会 LLM 意图分类规则，
        结合对话历史理解用户真实意图

        返回格式: {"intent": IntentCategory, "confidence": float, "reasoning": str}
        """
        message = self._clean_text(message)

        # 构建 Few-shot 示例：每类意图取 1 条示例，控制 prompt 长度
        examples = "\n".join(
            f'  消息: "{t}" → 意图: {cat.value}'
            for cat, tpls in _TEMPLATES.items()
            for t in tpls[:1]  # 每类取 1 条，控制 prompt 长度
        )

        # 构建对话上下文：只取最近 3 轮，避免 prompt 过长
        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        # 构建 prompt：强调优先返回细粒度意图
        prompt = f"""你是客服意图分析专家。根据示例判断用户意图，返回 JSON。
如果用户问题能匹配细粒度业务意图，请优先返回细粒度意图，而不是宽泛大类。
例如退款优先返回 refund，发票优先返回 invoice，登录故障优先返回 technical_login。

        {ctx}
        用户消息: "{message}"

返回格式（仅 JSON，不要其他文字）:
{{"intent": "<意图值>", "confidence": <0-1>, "reasoning": "<一句话说明>"}}

可选意图: {", ".join(c.value for c in IntentCategory)}"""
        prompt = self._clean_text(prompt)

        try:
            # 调用 LLM
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.1,  # 低温度确保稳定输出
                messages=[{"role": "user", "content": prompt}],
            )
            # 提取 JSON 响应
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])

            # 验证意图值是否合法
            try:
                data["intent"] = IntentCategory(data["intent"])
            except ValueError:
                data["intent"] = IntentCategory.OTHER
            return data
        except Exception as ex:
            logger.warning(f"LLM 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "reasoning": "LLM 失败", "failed": True}

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """
        策略 2：Embedding 向量相似度匹配

        优点：速度快（~100ms）、适合匹配常见表达
        缺点：无法理解复杂语义、依赖模板质量

        工作流程：
        1. 懒加载模板向量（首次调用时计算并缓存）
        2. 计算用户消息的向量
        3. 与每个意图类别的模板向量计算余弦相似度
        4. 返回相似度最高的意图

        返回格式: {"intent": IntentCategory, "confidence": float}
        """
        try:
            # 1. 确保模板向量已加载
            await self._load_template_embeddings()

            # 2. 计算用户消息的向量
            msg_vec = await self._embed_text(message)

            # 3. 与所有意图模板计算相似度，取最高分
            best_cat, best_score = IntentCategory.OTHER, 0.0
            for cat, vecs in self._tpl_embeddings.items():
                # 一个意图有多个模板，取最高相似度
                score = max(_cosine(msg_vec, v) for v in vecs)
                if score > best_score:
                    best_score, best_cat = score, cat

            return {"intent": best_cat, "confidence": best_score}
        except Exception as ex:
            logger.warning(f"Embedding 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0}

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        """
        策略 3：关键词模式匹配（同步，零延迟兜底）

        优点：零延迟（<1ms）、无 API 成本、100% 确定性
        缺点：只能匹配显式关键词、无法理解语义

        匹配策略：
        1. 优先匹配细粒度业务意图（如 course_selection、grade_query、emergency）
        2. 未命中则匹配通用大类意图（如 query、complaint）

        这样设计避免了细粒度意图被通用关键词覆盖
        例如："设备故障" 应该识别为 device_failure，而不是 query

        返回格式: {"intent": IntentCategory, "confidence": float}
        """
        msg = message.lower()

        # 细粒度业务意图模式（优先匹配）
        # 注意：按优先级排序，先匹配更具体的模式
        specific_patterns = {
            IntentCategory.IT_ESCALATION: ["转人工", "创建工单", "升级处理", "专家", "技术支持"],
            # 设备故障：增强蓝屏、错误码匹配
            IntentCategory.DEVICE_FAILURE: ["蓝屏", "死机", "崩溃", "0x", "错误码", "开不了机", "黑屏", "打印机", "卡纸", "键盘失灵", "鼠标坏了"],
            IntentCategory.SOFTWARE_INSTALL: ["安装软件", "安装office", "软件更新", "升级系统", "下载"],
            # 权限申请：明确是申请类动作
            IntentCategory.PERMISSION_REQUEST: ["申请权限", "开通权限", "申请访问", "开通系统", "申请管理员", "需要权限", "申请vpn权限", "开通vpn权限"],
            IntentCategory.ACCOUNT_ISSUE: ["忘记密码", "密码重置", "账号被锁", "无法登录", "登录失败"],
            # VPN访问：明确是连接问题
            IntentCategory.VPN_ACCESS: ["vpn连不上", "vpn断开", "vpn掉线", "vpn登录失败", "vpn报错", "远程连接失败"],
            IntentCategory.NETWORK_ISSUE: ["网络断了", "断网", "内网访问", "网页打不开"],
            IntentCategory.WIFI_PROBLEM: ["wifi", "wi-fi", "无线网", "信号弱"],
            IntentCategory.NETWORK_SLOW: ["网速慢", "网速", "卡顿", "延迟高"],
            IntentCategory.CONTACT_INFO: ["联系方式", "电话", "找谁", "负责人"],
        }

        # 通用大类意图模式（兜底匹配）
        generic_patterns = {
            IntentCategory.COMPLAINT:  ["投诉", "太差", "糟糕", "不满", "抱怨"],
            IntentCategory.QUERY:      ["?", "？", "怎么", "什么", "哪里", "如何"],
            IntentCategory.GREETING:   ["你好", "嗨", "hello", "hi", "您好"],
            IntentCategory.FEEDBACK:   ["很好", "不错", "满意", "感谢", "谢谢"],
        }

        # 1. 优先匹配细粒度意图
        best_cat, best_score = self._best_pattern_match(msg, specific_patterns)
        if best_cat != IntentCategory.OTHER:
            return {"intent": best_cat, "confidence": best_score}

        # 2. 未命中则匹配通用大类
        best_cat, best_score = self._best_pattern_match(msg, generic_patterns)
        return {"intent": best_cat, "confidence": best_score}

    # ── 投票合并 ──────────────────────────────────────────────────────────────

    def _vote(self, llm: Dict, emb: Dict, pat: Dict) -> tuple[IntentCategory, float, Dict[str, float]]:
        """
        三路识别结果加权投票合并

        投票策略：
        1. 正常情况：LLM 70% + Embedding 20% + Pattern 10%
        2. LLM 失败时：Embedding 或 Pattern 降级兜底
        3. 细粒度优化：Pattern 匹配到细粒度意图且置信度高时，覆盖通用大类

        设计思路：
        - LLM 理解力最强，权重最高
        - Embedding 速度快，作为辅助验证
        - Pattern 零延迟，用于兜底和细粒度意图提升

        返回: (最终意图, 融合置信度, 各路来源得分)
        """
        # 记录各路来源的原始得分
        source_scores = {
            "llm": float(llm.get("confidence", 0.0) or 0.0),
            "embedding": float(emb.get("confidence", 0.0) or 0.0),
            "pattern": float(pat.get("confidence", 0.0) or 0.0),
        }

        # LLM 失败时的降级逻辑
        if llm.get("failed"):
            # 优先使用 Embedding 结果
            if emb.get("intent") != IntentCategory.OTHER and emb.get("confidence", 0.0) > 0:
                return emb["intent"], source_scores["embedding"], source_scores
            # 次选 Pattern 结果
            if pat.get("intent") != IntentCategory.OTHER and pat.get("confidence", 0.0) > 0:
                return pat["intent"], source_scores["pattern"], source_scores
            # 都失败则返回 OTHER
            return IntentCategory.OTHER, 0.0, source_scores

        # 正常情况：加权投票
        # 优化策略：当Pattern匹配强时，增加其权重，降低LLM过度解释的影响
        if self._embedding_enabled:
            # 如果Pattern有高置信度匹配（>0.7），提高Pattern权重
            if source_scores["pattern"] > 0.7:
                weights = [(llm, 0.5), (emb, 0.2), (pat, 0.3)]
            else:
                weights = [(llm, 0.6), (emb, 0.25), (pat, 0.15)]
        else:
            # Embedding 不可用时调整权重
            if source_scores["pattern"] > 0.7:
                weights = [(llm, 0.6), (pat, 0.4)]
            else:
                weights = [(llm, 0.75), (pat, 0.25)]

        # 计算每个意图的加权得分
        scores: Dict[IntentCategory, float] = {}
        for result, w in weights:
            cat  = result.get("intent", IntentCategory.OTHER)
            conf = result.get("confidence", 0.0)
            scores[cat] = scores.get(cat, 0.0) + w * conf

        # 取得分最高的意图
        best = max(scores, key=scores.get)  # type: ignore
        best_score = scores[best]

        # 细粒度优化：Pattern 匹配到细粒度意图时，可能覆盖通用大类
        # 例如："我要退款" LLM 可能返回 billing（通用），Pattern 返回 refund（细粒度）
        pat_intent = pat.get("intent", IntentCategory.OTHER)
        pat_conf = float(pat.get("confidence", 0.0) or 0.0)

        # 如果当前最佳是通用大类，且 Pattern 匹配到细粒度意图，则使用细粒度
        if best in _GENERIC_INTENTS and pat_intent in _SPECIFIC_INTENTS and pat_conf >= 0.5 and best_score < 0.8:
            source_scores["refined_by_pattern"] = pat_conf
            return pat_intent, max(best_score, pat_conf), source_scores

        # 置信度低于阈值时返回 OTHER（表示不确定）
        if best_score < self.threshold:
            return IntentCategory.OTHER, best_score, source_scores

        return best, best_score, source_scores

    def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """
        使用正则表达式提取结构化实体

        提取目标：
        - employee_id: 工号（6-10位数字或字母数字组合）
        - device_id: 设备编号（如 PC-001、NB-2023-001）
        - error_code: 错误码（如 0x80070005、ERR_CONNECTION_REFUSED）
        - os_version: 操作系统版本（如 Windows 10、macOS 13.0）
        - ip_address: IP地址
        - email: 邮箱地址

        设计选择：
        使用规则提取而非 LLM，原因：
        1. 速度快（无 API 调用）
        2. 确定性强（不会幻觉）
        3. 成本低（无额外费用）
        4. 准确度高（针对结构化信息）

        返回: {"实体类型": ["提取值1", "提取值2", ...]}
        """
        message = self._clean_text(message)
        return {
            # 工号：匹配 "工号:E001234" 或 "员工编号: 2021001" 等格式
            "employee_id": self._unique(re.findall(r"(?:工号|员工编号|员工号)\s*[:：]?\s*([A-Z]?\d{6,10})", message, re.I)),

            # 设备编号：匹配 "PC-001"、"NB-2023-001"、"PRINTER-05" 等格式
            "device_id": self._unique(re.findall(r"\b([A-Z]{2,10}-\d{3,10}(?:-\d{3,10})?)\b", message, re.I)),

            # 错误码：匹配 "0x80070005"、"ERR_CONNECTION_REFUSED"、"Error 404" 等
            "error_code": self._unique(re.findall(r"(?:错误码|error\s*code|错误)\s*[:：]?\s*((?:0x[0-9A-F]{8}|ERR_[A-Z_]+|\d{3,4}))", message, re.I)),

            # 操作系统：匹配 "Windows 10"、"macOS 13"、"Ubuntu 22.04" 等
            "os_version": self._unique(re.findall(r"(Windows\s*\d+|macOS\s*\d+(?:\.\d+)?|Ubuntu\s*\d+(?:\.\d+)?|iOS\s*\d+|Android\s*\d+)", message, re.I)),

            # IP地址：匹配标准IPv4格式
            "ip_address": self._unique(re.findall(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b", message)),

            # 邮箱：匹配标准邮箱格式
            "email": self._unique(re.findall(r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b", message)),

            # 软件名称：常见办公软件
            "software": self._unique(re.findall(r"\b(Office|Word|Excel|PowerPoint|Outlook|Teams|Zoom|Chrome|Firefox|Photoshop|AutoCAD)\b", message, re.I)),

            # 日期：匹配 "今天"、"2024-01-01"、"2024/01/01" 等格式
            "date": self._unique(re.findall(r"(今天|明天|昨天|本周|这周|下周|\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)", message)),
        }

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    async def _load_template_embeddings(self) -> None:
        """懒加载所有模板的 Embedding（只在首次调用时执行）。"""
        missing = [cat for cat in _TEMPLATES if cat not in self._tpl_embeddings]
        if not missing:
            return

        all_texts = [t for cat in missing for t in _TEMPLATES[cat]]
        vecs = [await self._embed_text(text) for text in all_texts]
        idx = 0
        for cat in missing:
            n = len(_TEMPLATES[cat])
            self._tpl_embeddings[cat] = vecs[idx: idx + n]
            idx += n

    async def _embed_text(self, text: str) -> List[float]:
        """
        生成文本向量。

        如果未来接入的官方/兼容客户端提供 embeddings.create，会优先使用远端向量；
        当前 Anthropic SDK 没有该资源时，退化为字符 n-gram 哈希向量。这样不会因为
        Embedding 服务缺失导致三路融合中断。
        """
        embeddings = getattr(self.client, "embeddings", None)
        if embeddings is not None:
            try:
                resp = await embeddings.create(model="voyage-3-lite", input=[text])
                return list(resp.data[0].embedding)
            except Exception as ex:
                logger.warning(f"远端 Embedding 失败，使用本地向量兜底: {ex}")

        return self._local_embedding(text)

    @staticmethod
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        """稳定的字符 n-gram 哈希向量，用于无远端 Embedding 时的语义近似匹配。"""
        normalized = text.lower().strip()
        vec = [0.0] * dims
        tokens = set()
        for n in (1, 2, 3):
            if len(normalized) >= n:
                tokens.update(normalized[i:i + n] for i in range(len(normalized) - n + 1))
        if not tokens:
            tokens.add(normalized)

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        msg = message.lower()
        for level, kws in _URGENCY_KEYWORDS.items():
            if any(kw in msg for kw in kws):
                return level
        if intent == IntentCategory.IT_ESCALATION:
            return UrgencyLevel.HIGH
        if intent == IntentCategory.COMPLAINT:
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

    def _cache_key(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        payload = {"message": self._clean_text(message)[:200]}
        if history:
            payload["history"] = [
                {
                    "role": self._clean_text(item.get("role", ""))[:20],
                    "content": self._clean_text(item.get("content", ""))[:160],
                }
                for item in history[-3:]
            ]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _unique(values: List[str]) -> List[str]:
        return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))

    @staticmethod
    def _best_pattern_match(
        message: str,
        patterns: Dict[IntentCategory, List[str]],
    ) -> tuple[IntentCategory, float]:
        best_cat, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if kw in message)
            if not hits:
                continue
            # 单个明确业务关键词就给可用置信度；多个关键词命中时提高置信度。
            score = min(1.0, 0.5 + 0.25 * (hits - 1))
            if score > best_score:
                best_score, best_cat = score, cat
        return best_cat, best_score

    @staticmethod
    def _intent_group(intent: IntentCategory) -> str:
        return _INTENT_GROUPS.get(intent, intent).value

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 HTTP 客户端编码 prompt 时崩溃。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }
