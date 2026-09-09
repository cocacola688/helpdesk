"""
亮点：多 Agent 路由与编排

核心问题：多 Agent 情况下如何做 Routing？

路由策略（三层决策）：
  1. 意图路由 —— 根据 IntentCategory 直接映射到专属 Agent
  2. 性能路由 —— 同类 Agent 有多个时，选成功率最高、延迟最低的
  3. 降级路由 —— 专属 Agent 不可用时，自动降级到 GeneralAgent

并行协作：
  - 复杂问题（如"技术问题 + 账单问题"）可同时派发给多个 Agent
  - 结果由 Orchestrator 合并后返回

升级机制：
  - Agent 置信度低于阈值 → 自动升级到更高级 Agent 或转人工
"""
import asyncio
import inspect
import json
import logging
import os
import random
import time
import uuid
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from agents.tools import (
    AgentToolSpec,
    build_shared_rag_tools,
    device_tools,
    permission_tools,
    network_tools,
    escalation_tools,
    general_tools,
)
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class AgentType(Enum):
    GENERAL    = "general"        # 一线IT支持（通用接待）
    DEVICE     = "device"         # 设备故障处理
    PERMISSION = "permission"     # 权限申请处理
    NETWORK    = "network"        # 网络问题处理
    ESCALATION = "escalation"     # 升级处理/创建工单


@dataclass(frozen=True)
class AgentProfile:

    role: str
    mission: str
    workflow: Tuple[str, ...]
    input_contract: Tuple[str, ...]
    output_contract: Tuple[str, ...]
    handoff_conditions: Tuple[str, ...] = ()
    tool_scope: Tuple[str, ...] = ()
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024


def _env_float(name: str, default: float) -> float:
    """读取可选浮点配置；错误配置不应阻塞服务启动。"""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("忽略非法浮点配置 %s=%r", name, os.getenv(name))
        return default


def _env_int(name: str, default: int) -> int:
    """读取可选整数配置；错误配置不应阻塞服务启动。"""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("忽略非法整数配置 %s=%r", name, os.getenv(name))
        return default


@dataclass
class AgentStats:
    """Agent 运行时统计，供 Monitor 和路由决策使用。"""
    total:     int   = 0
    success:   int   = 0
    total_ms:  float = 0.0
    monitor_penalty: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    def routing_score(self) -> float:
        """路由评分：成功率高、延迟低的 Agent 得分高。"""
        latency_score = 1.0 / (1.0 + self.avg_ms / 1000)
        base_score = self.success_rate * 0.7 + latency_score * 0.3
        return base_score * max(0.0, 1.0 - self.monitor_penalty)


@dataclass
class AgentResponse:
    agent_type:  AgentType
    content:     str
    success:     bool
    confidence:  float = 1.0
    latency_ms:  float = 0.0
    escalate:    bool  = False   # 是否需要升级
    tools_used:  List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""        # 来自 MemoryManager 的格式化上下文
    history:     Optional[List[Dict[str, str]]] = None  # 对话历史，传给意图识别
    entities:    Dict[str, List[str]] = field(default_factory=dict)
    intent:      Optional[IntentCategory] = None
    intent_group: Optional[str] = None
    urgency:     Optional[UrgencyLevel]   = None
    intent_confidence: float = 1.0
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class OrchestratorResult:
    request_id:  str
    response:    str
    agent_type:  AgentType
    intent:      Optional[IntentCategory]
    escalated:   bool  = False
    latency_ms:  float = 0.0
    agent_types: List[AgentType] = field(default_factory=list)
    primary_agent: Optional[AgentType] = None
    supporting_agents: List[AgentType] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0


@dataclass
class RoutingDecision:
    """一次请求的结构化路由决策。"""
    primary_agent: AgentType
    supporting_agents: List[AgentType] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0

    @property
    def agent_types(self) -> List[AgentType]:
        return [self.primary_agent] + self.supporting_agents

    @property
    def multi_agent(self) -> bool:
        return bool(self.supporting_agents)


# ── 基础 Agent ────────────────────────────────────────────────────────────────

class BaseAgent:
    """
    所有 Agent 的基类，封装 LLM 调用、角色契约和统计

    核心职责：
    1. 统一 LLM 调用接口（支持工具调用的多轮对话）
    2. 管理 Agent 角色契约（输入输出规范、职责边界）
    3. 统计运行指标（成功率、延迟），供路由决策使用
    4. 工具白名单管理（每个 Agent 只能访问特定工具）

    设计亮点：
    - 工具调用自动循环：LLM 返回 tool_use 时自动执行并继续对话
    - 角色契约注入：将 AgentProfile 动态拼入 system prompt
    - 失败自动降级：Agent 失败时 Orchestrator 会降级到 GeneralAgent
    """

    agent_type: AgentType       # Agent 类型（GENERAL/TECHNICAL/BILLING/ESCALATION）
    system_prompt: str          # 基础系统提示词
    profile: AgentProfile       # 角色契约（职责、工作流、输入输出规范）

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        skill_manager: Optional[Any] = None,
        profile: Optional[AgentProfile] = None,
    ):
        """
        初始化 Agent

        参数:
            client: Anthropic API 客户端
            model: 默认使用的 LLM 模型
            skill_manager: Skills 管理器，用于动态注入业务规则
            profile: 角色契约，定义 Agent 的职责和能力边界
        """
        self._client = client
        self.profile = profile or self.profile
        self._model  = self.profile.model or model
        self._skill_manager = skill_manager

        # 运行时统计：用于路由评分和监控
        self.stats   = AgentStats()

        # 工具调用追踪：记录最近一次请求使用的工具
        self._last_tools_used: List[str] = []
        self._last_tool_traces: List[Dict[str, Any]] = []

        # 工具白名单：只有这里的工具才能被该 Agent 调用
        self._shared_tools: Dict[str, AgentToolSpec] = {}

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        """返回该角色真实可调用的工具白名单。"""
        return dict(self._shared_tools)

    def set_shared_tools(self, tools: Optional[Dict[str, AgentToolSpec]]) -> None:
        self._shared_tools = dict(tools or {})

    async def handle(self, req: Request) -> AgentResponse:
        """
        处理用户请求（Agent 主入口）

        执行流程：
        1. 调用 LLM 处理请求（可能涉及多轮工具调用）
        2. 检测是否需要升级到人工客服
        3. 记录统计数据（成功率、延迟）
        4. 返回响应结果

        统计数据的作用：
        - 成功率用于路由决策（低成功率的 Agent 会被降权）
        - 延迟用于监控和告警
        - 连续失败数用于熔断保护

        参数:
            req: 包含用户消息、上下文、历史等完整请求信息

        返回:
            AgentResponse: 包含回复内容、成功状态、延迟等信息
        """
        t0 = time.monotonic()
        self.stats.total += 1  # 总请求数 +1
        self._last_tools_used = []
        self._last_tool_traces = []

        try:
            # 调用 LLM 处理请求（核心逻辑）
            content = await self._call_llm(req)
            ms = (time.monotonic() - t0) * 1000

            # 成功统计
            self.stats.success += 1
            self.stats.total_ms += ms

            # 检测是否需要升级（通过关键词检测）
            escalate = self._needs_escalation(content)

            return AgentResponse(
                agent_type=self.agent_type,
                content=content,
                success=True,
                latency_ms=ms,
                escalate=escalate,
                tools_used=list(self._last_tools_used),
                tool_traces=list(self._last_tool_traces),
            )
        except Exception as ex:
            # 失败统计
            ms = (time.monotonic() - t0) * 1000
            self.stats.total_ms += ms
            logger.error(f"{self.agent_type.value} 处理失败: {ex}")

            # 返回友好的错误提示（不暴露技术细节）
            return AgentResponse(
                agent_type=self.agent_type,
                content="抱歉，处理您的请求时出现问题，请稍后重试。",
                success=False,
                latency_ms=ms,
                tool_traces=list(self._last_tool_traces),
            )

    async def handle_stream(self, req: Request):
        """
        流式处理用户请求（支持实时输出）

        相比 handle() 方法：
        - 实时 yield LLM 生成的文本
        - 降低首字延迟
        - 最终 yield 完整的 AgentResponse

        Yields:
            dict:
                - {"type": "content", "text": "..."}  # 流式文本片段
                - {"type": "done", "response": AgentResponse}  # 完整响应
        """
        t0 = time.monotonic()
        self.stats.total += 1
        self._last_tools_used = []
        self._last_tool_traces = []

        try:
            # 流式调用 LLM
            full_content = ""
            async for chunk in self._call_llm_stream(req):
                if chunk.get("type") == "content":
                    text = chunk.get("text", "")
                    full_content += text
                    yield {"type": "content", "text": text}

            ms = (time.monotonic() - t0) * 1000

            # 成功统计
            self.stats.success += 1
            self.stats.total_ms += ms

            # 检测是否需要升级
            escalate = self._needs_escalation(full_content)

            response = AgentResponse(
                agent_type=self.agent_type,
                content=full_content,
                success=True,
                latency_ms=ms,
                escalate=escalate,
                tools_used=list(self._last_tools_used),
                tool_traces=list(self._last_tool_traces),
            )
            yield {"type": "done", "response": response}

        except Exception as ex:
            ms = (time.monotonic() - t0) * 1000
            self.stats.total_ms += ms
            logger.error(f"{self.agent_type.value} 流式处理失败: {ex}")

            error_msg = "抱歉，处理您的请求时出现问题，请稍后重试。"
            yield {"type": "content", "text": error_msg}
            yield {
                "type": "done",
                "response": AgentResponse(
                    agent_type=self.agent_type,
                    content=error_msg,
                    success=False,
                    latency_ms=ms,
                    tool_traces=list(self._last_tool_traces),
                )
            }

    async def _call_llm(self, req: Request) -> str:
        """
        调用 LLM 处理请求，支持多轮工具调用

        这是 Agent 的核心方法，实现了完整的 Tool Use 循环：

        执行流程：
        1. 构建消息列表：
           - 先注入背景信息（记忆上下文）
           - 再注入结构化实体
           - 然后注入角色输入契约
           - 最后是用户消息

        2. 循环调用 LLM（最多 3 轮）：
           - 如果 LLM 返回纯文本 → 结束，返回文本
           - 如果 LLM 返回 tool_use → 执行工具 → 将结果反馈给 LLM → 继续循环

        3. 工具执行：
           - 验证工具是否在白名单中
           - 验证参数是否符合 Schema
           - 执行工具（支持同步和异步）
           - 记录工具调用追踪（用于调试和监控）

        工具调用循环的意义：
        - Agent 可以"思考-行动-观察"循环解决复杂问题
        - 例如：查询订单状态 → 发现需要更多信息 → 询问用户 → 继续处理

        参数:
            req: 用户请求，包含消息、上下文、实体等

        返回:
            LLM 最终回复的文本内容

        异常:
            RuntimeError: 工具调用超过最大轮数（3 轮）
        """
        # 清洗文本：移除非法 UTF-8 字符，避免 API 调用失败
        def _clean(s: str) -> str:
            return s.encode("utf-8", errors="ignore").decode("utf-8")

        # 1. 构建消息列表（注入上下文信息）
        messages = []

        # 注入记忆上下文（来自 MemoryManager）
        if req.context:
            messages.append({"role": "user", "content": f"[背景信息]\n{_clean(req.context)}"})
            messages.append({"role": "assistant", "content": "好的，我已了解背景信息。"})

        # 注入结构化实体（订单号、金额、错误码等）
        if req.entities:
            entities_text = json.dumps(req.entities, ensure_ascii=False)
            messages.append({"role": "user", "content": f"[结构化实体]\n{_clean(entities_text)}"})
            messages.append({"role": "assistant", "content": "好的，我会结合这些结构化实体处理。"})

        # 注入角色输入契约（告诉 Agent 当前有哪些可用信息）
        role_packet = self._build_role_packet(req)
        if role_packet:
            messages.append({"role": "user", "content": f"[角色输入契约]\n{_clean(role_packet)}"})
            messages.append({"role": "assistant", "content": "好的，我会按照该角色的输入和输出契约处理。"})

        # 注入用户消息
        messages.append({"role": "user", "content": _clean(req.message)})

        # 2. 工具调用循环（最多 3 轮）
        tools = self.get_tools()  # 获取该 Agent 的工具白名单
        tools_used: List[str] = []
        tool_traces: List[Dict[str, Any]] = []

        for _ in range(3):  # 限制最大轮数，防止无限循环
            # 构建 API 请求参数
            request_kwargs: Dict[str, Any] = {
                "model": self._model,
                "max_tokens": self.profile.max_tokens,
                "temperature": self.profile.temperature,
                "system": self._build_system_prompt(req),  # 动态构建系统提示词
                "messages": messages,
            }

            # 如果有可用工具，添加 tools 参数
            if tools:
                request_kwargs["tools"] = [
                    {
                        "name": spec.name,
                        "description": spec.description,
                        "input_schema": spec.input_schema,
                    }
                    for spec in tools.values()
                ]

            # 调用 LLM
            resp = await self._client.messages.create(**request_kwargs)

            # 提取 tool_use 块
            tool_uses = [block for block in (resp.content or []) if self._block_type(block) == "tool_use"]

            # 如果没有工具调用，说明 LLM 已经完成回复
            if not tool_uses:
                self._last_tools_used = tools_used
                return extract_text_content(resp.content)

            # 有工具调用，需要执行工具并继续对话
            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []

            # 执行所有工具调用
            for block in tool_uses:
                name = self._block_value(block, "name")
                tool_use_id = self._block_value(block, "id")
                args = self._block_value(block, "input") or {}
                spec = tools.get(name)

                tool_t0 = time.monotonic()
                call_success = True
                result_success: Optional[bool] = None
                error_text = ""

                # 检查工具是否在白名单中
                if spec is None:
                    call_success = False
                    result: Any = {"success": False, "error": f"工具不在 {self.agent_type.value} Agent 白名单中"}
                    error_text = result["error"]
                else:
                    try:
                        # 验证工具参数
                        self._validate_tool_input(spec, args)

                        # 执行工具
                        result = spec.handler(req, args)
                        if inspect.isawaitable(result):
                            result = await result

                        tools_used.append(name)

                        # 记录工具执行结果
                        if isinstance(result, dict) and "success" in result:
                            result_success = bool(result.get("success"))
                    except Exception as ex:
                        call_success = False
                        logger.warning("Agent 工具 %s 执行失败: %s", name, ex)
                        error_text = str(ex)
                        result = {"success": False, "error": error_text}

                tool_latency_ms = (time.monotonic() - tool_t0) * 1000

                # 提取错误信息
                if not error_text and isinstance(result, dict):
                    error_text = str(result.get("error", "") or "")

                # 记录工具调用追踪（用于调试和监控）
                tool_traces.append(
                    {
                        "agent_type": self.agent_type.value,
                        "tool_name": name,
                        "tool_use_id": tool_use_id,
                        "input": dict(args),
                        "success": call_success,
                        "result_success": result_success,
                        "latency_ms": round(tool_latency_ms, 1),
                        "cached": bool(result.get("cached")) if isinstance(result, dict) else False,
                        "reranked": bool(result.get("reranked")) if isinstance(result, dict) else False,
                        "error": error_text,
                    }
                )

                # 构建工具结果（反馈给 LLM）
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

            # 将工具结果添加到消息列表，继续下一轮对话
            messages.append({"role": "user", "content": tool_results})

        # 超过最大轮数仍未完成，抛出异常
        self._last_tools_used = tools_used
        self._last_tool_traces = tool_traces
        raise RuntimeError(f"{self.agent_type.value} 工具调用超过最大轮数")

    async def _call_llm_stream(self, req: Request):
        """
        流式调用 LLM，支持多轮工具调用

        与 _call_llm 的区别：
        - 使用 messages.stream() 替代 messages.create()
        - 实时 yield 文本内容
        - 工具调用暂不支持流式（工具执行完后再继续流式输出）

        Yields:
            dict: {"type": "content", "text": "..."}
        """
        def _clean(s: str) -> str:
            return s.encode("utf-8", errors="ignore").decode("utf-8")

        # 1. 构建消息列表（与 _call_llm 相同）
        messages = []

        # 注入背景信息
        if req.context:
            messages.append({"role": "user", "content": f"[背景信息]\n{_clean(req.context)}"})
            messages.append({"role": "assistant", "content": "好的，我已了解背景信息。"})

        # 注入结构化实体
        if req.entities:
            ent_lines = []
            for k, vals in req.entities.items():
                if vals:
                    ent_lines.append(f"- {k}: {', '.join(vals)}")
            if ent_lines:
                ent_text = "[结构化实体]\n" + "\n".join(ent_lines)
                messages.append({"role": "user", "content": ent_text})
                messages.append({"role": "assistant", "content": "好的，我会结合这些结构化实体处理。"})

        # 注入角色输入契约
        role_packet = self._build_role_packet(req)
        if role_packet:
            messages.append({"role": "user", "content": f"[角色输入契约]\n{_clean(role_packet)}"})
            messages.append({"role": "assistant", "content": "好的，我会按照该角色的输入和输出契约处理。"})

        # 注入用户消息
        messages.append({"role": "user", "content": _clean(req.message)})

        # 2. 工具调用循环（最多 3 轮）
        tools = self.get_tools()
        tools_used: List[str] = []
        tool_traces: List[Dict[str, Any]] = []

        for round_idx in range(3):
            # 构建 API 请求参数
            request_kwargs: Dict[str, Any] = {
                "model": self._model,
                "max_tokens": self.profile.max_tokens,
                "temperature": self.profile.temperature,
                "system": self._build_system_prompt(req),
                "messages": messages,
            }

            # 添加工具定义
            if tools:
                request_kwargs["tools"] = [
                    {
                        "name": spec.name,
                        "description": spec.description,
                        "input_schema": spec.input_schema,
                    }
                    for spec in tools.values()
                ]

            # 流式调用 LLM
            has_tool_use = False
            stream_content = []

            async with self._client.messages.stream(**request_kwargs) as stream:
                async for text in stream.text_stream:
                    yield {"type": "content", "text": text}

                # 获取完整消息（用于检查是否有工具调用）
                message = await stream.get_final_message()
                stream_content = message.content

            # 检查是否有工具调用
            tool_uses = [block for block in stream_content if self._block_type(block) == "tool_use"]

            # 没有工具调用，流式输出已完成
            if not tool_uses:
                self._last_tools_used = tools_used
                self._last_tool_traces = tool_traces
                return

            # 有工具调用：执行工具，继续下一轮（工具调用不支持流式）
            has_tool_use = True
            messages.append({"role": "assistant", "content": stream_content})

            tool_results = []
            for block in tool_uses:
                name = self._block_value(block, "name")
                tool_use_id = self._block_value(block, "id")
                args = self._block_value(block, "input") or {}
                spec = tools.get(name)

                tool_t0 = time.monotonic()
                call_success = True
                result_success: Optional[bool] = None
                error_text = ""

                if spec is None:
                    call_success = False
                    result: Any = {"success": False, "error": f"工具不在 {self.agent_type.value} Agent 白名单中"}
                    error_text = result["error"]
                else:
                    try:
                        self._validate_tool_input(spec, args)
                        result = spec.handler(req, args)
                        if inspect.isawaitable(result):
                            result = await result
                        tools_used.append(name)
                        if isinstance(result, dict) and "success" in result:
                            result_success = bool(result.get("success"))
                    except Exception as ex:
                        call_success = False
                        logger.warning("Agent 工具 %s 执行失败: %s", name, ex)
                        error_text = str(ex)
                        result = {"success": False, "error": error_text}

                tool_latency_ms = (time.monotonic() - tool_t0) * 1000

                if not error_text and isinstance(result, dict):
                    error_text = str(result.get("error", "") or "")

                tool_traces.append({
                    "agent_type": self.agent_type.value,
                    "tool_name": name,
                    "tool_use_id": tool_use_id,
                    "input": dict(args),
                    "success": call_success,
                    "result_success": result_success,
                    "latency_ms": round(tool_latency_ms, 1),
                    "cached": bool(result.get("cached")) if isinstance(result, dict) else False,
                    "reranked": bool(result.get("reranked")) if isinstance(result, dict) else False,
                    "error": error_text,
                })

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

            messages.append({"role": "user", "content": tool_results})

        # 超过最大轮数
        self._last_tools_used = tools_used
        self._last_tool_traces = tool_traces
        raise RuntimeError(f"{self.agent_type.value} 流式工具调用超过最大轮数")

    @staticmethod
    def _block_type(block: Any) -> Optional[str]:
        if isinstance(block, dict):
            return block.get("type")
        return getattr(block, "type", None)

    @staticmethod
    def _block_value(block: Any, key: str) -> Any:
        if isinstance(block, dict):
            return block.get(key)
        return getattr(block, key, None)

    @staticmethod
    def _validate_tool_input(spec: AgentToolSpec, args: Any) -> None:
        if not isinstance(args, dict):
            raise ValueError("工具参数必须是 JSON 对象")
        schema = spec.input_schema
        for field_name in schema.get("required", []):
            if field_name not in args:
                raise ValueError(f"缺少必需参数: {field_name}")
        properties = schema.get("properties", {})
        unknown = set(args) - set(properties)
        if unknown and schema.get("additionalProperties") is False:
            raise ValueError(f"不允许的工具参数: {', '.join(sorted(unknown))}")
        type_map = {"string": str, "number": (int, float), "integer": int, "boolean": bool}
        for key, value in args.items():
            expected = properties.get(key, {}).get("type")
            if expected in type_map and not isinstance(value, type_map[expected]):
                raise ValueError(f"参数 {key} 类型错误，期望 {expected}")

    def _build_system_prompt(self, req: Request) -> str:
        """把角色契约和动态 Skills 拼入 system prompt。"""
        profile_prompt = (
            f"\n\n[角色契约]\n"
            f"角色：{self.profile.role}\n"
            f"职责：{self.profile.mission}\n"
            f"处理流程：{' -> '.join(self.profile.workflow)}\n"
            f"可用输入：{'；'.join(self.profile.input_contract)}\n"
            f"输出要求：{'；'.join(self.profile.output_contract)}\n"
            f"升级条件：{'；'.join(self.profile.handoff_conditions) or '无，按通用客服规则处理'}\n"
            f"允许的数据/工具范围：{'、'.join(self.profile.tool_scope) or '仅使用当前请求上下文'}\n"
            "不要声称执行了未提供的查询、修改或退款操作；缺少证据时明确说明需要核验。"
        )
        base_prompt = f"{self.system_prompt}{profile_prompt}"
        if self._skill_manager is None:
            return base_prompt
        skill_prompt = self._skill_manager.prompt_for(req.message, self.agent_type.value)
        if not skill_prompt:
            return base_prompt
        return f"{base_prompt}\n\n[动态 Skills]\n{skill_prompt}"

    def _build_role_packet(self, req: Request) -> str:
        """给子 Agent 的确定性输入包；子类可补充领域字段。"""
        packet = {
            "agent_type": self.agent_type.value,
            "intent": req.intent.value if req.intent else None,
            "intent_group": req.intent_group,
            "urgency": req.urgency.name if req.urgency else None,
            "intent_confidence": round(req.intent_confidence, 4),
            "available_entities": req.entities or {},
        }
        return json.dumps(packet, ensure_ascii=False)

    def _needs_escalation(self, content: str) -> bool:
        """检测 Agent 是否建议升级（简单关键词检测）。"""
        keywords = ["转人工", "创建工单", "escalate", "specialist", "无法处理", "技术支持", "IT支持"]
        return any(kw in content for kw in keywords)


class GeneralAgent(BaseAgent):
    agent_type    = AgentType.GENERAL
    profile = AgentProfile(
        role="一线IT支持",
        mission="处理员工的IT日常问题，快速提供信息和指引，必要时引导到专业IT支持。",
        workflow=("理解员工问题", "检索IT知识库", "提供清晰答案", "必要时引导到专业IT支持"),
        input_contract=("对话历史", "员工问题", "IT知识库上下文"),
        output_contract=("先回应核心问题", "信息不足时只询问必要字段", "明确下一步和边界"),
        handoff_conditions=("涉及复杂设备故障、权限审批、网络架构调整", "员工明确要求人工处理或创建工单"),
        tool_scope=("search_knowledge_base", "inspect_request_context", "suggest_required_fields"),
        temperature=0.3,
        max_tokens=900,
    )
    system_prompt = (
        "你是企业IT支持助手。友好、简洁地回答员工的IT问题。"
        "如果问题超出你的能力范围，明确说明并建议联系专业IT支持或创建工单。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["triage_targets"] = ["device", "permission", "network", "escalation"]
        packet["response_mode"] = "answer_or_clarify"
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(general_tools())
        return tools


class DeviceAgent(BaseAgent):
    """设备故障处理 Agent

    专门处理电脑、手机、打印机等硬件设备的故障排查和配置问题。
    """
    agent_type    = AgentType.DEVICE
    profile = AgentProfile(
        role="设备故障处理专员",
        mission="处理电脑/手机/打印机等硬件故障排查，软件安装配置，提供详细的故障诊断和解决方案。",
        workflow=("确认设备型号和故障现象", "查询错误码和诊断方案", "提供排查步骤", "必要时升级到专家或创建工单"),
        input_contract=("设备信息", "错误码", "故障描述", "操作系统版本", "IT知识库上下文"),
        output_contract=("故障诊断", "排查步骤", "临时解决方案", "升级条件说明"),
        handoff_conditions=("硬件损坏需要维修", "需要IT资产管理审批", "超出远程支持能力"),
        tool_scope=("search_knowledge_base", "lookup_asset", "lookup_error_code", "build_diagnostic_plan"),
        temperature=0.1,
        max_tokens=1200,
    )
    system_prompt = (
        "你是设备故障处理专员。专注于：电脑故障、打印机问题、移动设备配置、软件安装。"
        "提供清晰的诊断步骤和解决方案。需要硬件维修或资产审批时，说明需要创建工单。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["device_fields"] = {
            "device_id": req.entities.get("device_id", []),
            "error_code": req.entities.get("error_code", []),
            "os_version": req.entities.get("os_version", []),
            "software": req.entities.get("software", []),
            "troubleshooting_boundary": "不得要求员工拆机、修改系统核心设置、执行高风险操作",
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(device_tools())
        return tools


class PermissionAgent(BaseAgent):
    """权限申请处理 Agent

    处理VPN/系统权限/账号开通等流程引导和申请事项。
    """
    agent_type    = AgentType.PERMISSION
    profile = AgentProfile(
        role="权限申请处理专员",
        mission="处理VPN/系统权限/账号开通流程引导，提供详细的申请条件、流程和所需材料。",
        workflow=("确认申请类型", "说明申请条件和流程", "列出所需材料", "引导提交申请或创建工单"),
        input_contract=("员工信息", "申请类型", "业务需求", "IT知识库上下文"),
        output_contract=("申请条件", "流程步骤", "所需材料", "审批时限", "注意事项"),
        handoff_conditions=("涉及高级权限审批", "跨部门协调", "特殊权限需求"),
        tool_scope=("search_knowledge_base", "check_permission_requirements", "get_approval_flow"),
        temperature=0.15,
        max_tokens=1000,
    )
    system_prompt = (
        "你是权限申请处理专员。负责VPN、系统权限、账号开通的流程引导。"
        "提供清晰的申请条件、流程步骤和材料清单。涉及审批的事项说明需要创建工单。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["permission_fields"] = {
            "employee_id": req.entities.get("employee_id", []),
            "email": req.entities.get("email", []),
            "permission_boundary": "不得承诺审批结果、权限生效时间",
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(permission_tools())
        return tools


class NetworkAgent(BaseAgent):
    """网络问题处理 Agent

    处理网络连接、Wi-Fi、VPN等网络相关问题的诊断。
    """
    agent_type    = AgentType.NETWORK
    profile = AgentProfile(
        role="网络问题处理专员",
        mission="处理网络连接、Wi-Fi、VPN问题诊断，提供详细的网络排查步骤和配置指导。",
        workflow=("确认网络问题类型", "执行基础诊断", "提供排查步骤", "必要时升级到网络工程师"),
        input_contract=("网络环境", "故障现象", "IP地址", "错误信息", "IT知识库上下文"),
        output_contract=("问题诊断", "排查步骤", "配置指导", "升级条件说明"),
        handoff_conditions=("涉及网络架构调整", "交换机路由器配置", "需要网络工程师现场处理"),
        tool_scope=("search_knowledge_base", "network_diagnostic", "check_vpn_status"),
        temperature=0.1,
        max_tokens=1000,
    )
    system_prompt = (
        "你是网络问题处理专员。专注于：网络连接、Wi-Fi问题、VPN诊断、网速优化。"
        "提供详细的诊断步骤和配置指导。涉及网络架构调整时，说明需要升级到网络工程师。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["network_fields"] = {
            "ip_address": req.entities.get("ip_address", []),
            "error_code": req.entities.get("error_code", []),
            "network_boundary": "不得要求员工修改路由器配置、执行网络扫描等敏感操作",
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(network_tools())
        return tools


class EscalationAgent(BaseAgent):
    """升级处理 Agent

    处理需要创建工单、转人工的场景，快速提供升级渠道。
    """

    agent_type = AgentType.ESCALATION
    profile = AgentProfile(
        role="IT工单升级处理",
        mission="处理需要创建工单或转人工的场景，快速提供升级渠道和工单创建指引。",
        workflow=("确认问题类型", "收集必要信息", "创建工单或提供联系方式", "说明后续处理流程"),
        input_contract=("问题描述", "紧急程度", "联系方式"),
        output_contract=("工单编号或联系方式", "预计处理时间", "注意事项"),
        handoff_conditions=("所有升级场景均会创建工单或转人工"),
        tool_scope=("create_ticket", "get_it_contacts"),
        temperature=0.0,
        max_tokens=500,
    )
    system_prompt = "你负责IT工单升级处理。快速创建工单或提供联系方式，不拖延。"

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(escalation_tools())
        return tools

    async def handle(self, req: Request) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        intent = req.intent.value if req.intent else "unknown"
        urgency = req.urgency.name if req.urgency else "UNKNOWN"
        content = (
            "📋 您的问题已记录，我将为您创建工单\n\n"
            "IT支持热线：400-XXX-XXXX（24小时）\n"
            "邮箱：itsupport@company.com\n"
            "企业微信：IT服务台\n\n"
            "工单创建后，专业技术人员将在以下时间内响应：\n"
            "- 紧急问题：30分钟内\n"
            "- 高优先级：2小时内\n"
            "- 普通问题：4小时内\n\n"
            f"问题类型：{intent}\n"
            f"紧急程度：{urgency}\n\n"
            "请保持联系方式畅通，如需补充信息，技术人员会主动联系您。"
        )
        ms = (time.monotonic() - t0) * 1000
        self.stats.success += 1
        self.stats.total_ms += ms
        return AgentResponse(
            agent_type=self.agent_type,
            content=content,
            success=True,
            latency_ms=ms,
            escalate=True,
            tools_used=[],
        )


class ResponseComposer:
    """多 Agent 汇总节点，统一主次、去重和输出边界。"""

    def __init__(self, client: AsyncAnthropic, model: str, skill_manager: Optional[Any] = None):
        self._client = client
        self._model = model
        self._skill_manager = skill_manager

    async def compose(self, req: Request, responses: List[AgentResponse]) -> str:
        successful = [response for response in responses if response.success and response.content.strip()]
        if not successful:
            return "抱歉，所有 Agent 均处理失败。"
        if len(successful) == 1:
            return successful[0].content

        evidence = "\n\n".join(
            f"[{response.agent_type.value} Agent 输出]\n{response.content}"
            for response in successful
        )
        prompt = (
            "你是IT支持 Response Composer，负责把多个专业 Agent 的结果合并成一条最终回复。\n"
            "要求：以主 Agent 的结论为主，按用户问题优先级组织内容；去掉重复和冲突表述；"
            "不能补造设备信息、权限审批结果、工单处理结果；如果结论冲突，明确说明需要核验；"
            "保留必要的诊断步骤、排查方法和升级边界。只输出给用户看的中文回复，不要提及 Agent。\n\n"
            f"主 Agent：{successful[0].agent_type.value}\n"
            f"用户问题：{req.message}\n"
            f"候选结果：\n{evidence}"
        )
        if self._skill_manager is not None:
            skill = self._skill_manager.prompt_for(req.message, "general")
            if skill:
                prompt += f"\n\n[IT支持输出边界]\n{skill}"
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=_env_int("HELPDESK_COMPOSER_MAX_TOKENS", 1000),
                temperature=_env_float("HELPDESK_COMPOSER_TEMPERATURE", 0.1),
                messages=[{"role": "user", "content": prompt}],
            )
            content = extract_text_content(response.content).strip()
            if content:
                return content
        except Exception as ex:
            logger.warning("Response Composer 失败，使用确定性合并: %s", ex)

        # 汇总节点不可用时保留主次标签，避免丢失某个专业 Agent 的结论。
        return "\n\n".join(
            f"{response.content}" if index == 0 else f"补充说明：\n{response.content}"
            for index, response in enumerate(successful)
        )


# ── 编排器 ────────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    多 Agent 编排器（系统核心调度中心）

    核心职责：
    1. 意图识别：理解用户想做什么
    2. 智能路由：选择最合适的 Agent 处理
    3. 并行协作：复杂问题派发给多个 Agent
    4. 降级保护：专属 Agent 失败时降级到 GeneralAgent
    5. 升级管理：识别需要人工介入的场景

    路由策略（三层决策）：
    1. 意图映射：根据 IntentCategory 直接映射到 AgentType
       例如：DEVICE_FAILURE → DeviceAgent, PERMISSION_REQUEST → PermissionAgent

    2. 性能路由：同类 Agent 有多个实例时，选成功率最高、延迟最低的
       通过 routing_score() = 成功率 * 0.7 + 延迟评分 * 0.3 计算

    3. 降级路由：专属 Agent 不可用或失败时，自动降级到 GeneralAgent
       确保系统永远有兜底方案

    设计亮点：
    - 静态路由表 + 动态评分：既有规则保证，又能自适应优化
    - 多 Agent 并行：技术+账单问题可同时处理，结果合并
    - 工具调用追踪：记录所有工具调用，便于调试和监控
    """

    # 意图 → Agent 类型的静态映射（路由表）
    # 这是路由的第一层决策：根据用户意图直接确定 Agent 类型
    _INTENT_ROUTING: Dict[IntentCategory, AgentType] = {
        # 设备类问题 → 设备支持
        IntentCategory.DEVICE_FAILURE: AgentType.DEVICE,
        IntentCategory.DEVICE_SETUP: AgentType.DEVICE,
        IntentCategory.SOFTWARE_INSTALL: AgentType.DEVICE,
        IntentCategory.HARDWARE_REQUEST: AgentType.DEVICE,

        # 权限类问题 → 权限处理
        IntentCategory.PERMISSION_REQUEST: AgentType.PERMISSION,
        IntentCategory.ACCOUNT_ISSUE: AgentType.PERMISSION,
        IntentCategory.VPN_ACCESS: AgentType.PERMISSION,

        # 网络类问题 → 网络支持
        IntentCategory.NETWORK_ISSUE: AgentType.NETWORK,
        IntentCategory.WIFI_PROBLEM: AgentType.NETWORK,
        IntentCategory.NETWORK_SLOW: AgentType.NETWORK,

        # 升级类问题 → 升级处理
        IntentCategory.IT_ESCALATION: AgentType.ESCALATION,

        # 其余意图 → GENERAL（默认，在 _route 方法中处理）
    }

    def __init__(
        self,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        skill_manager: Optional[Any] = None,
        rag_tool_manager: Optional[Any] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncAnthropic(**kwargs)

        self._intent_recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model)
        self._skill_manager = skill_manager
        self._composer = ResponseComposer(client, model, skill_manager)
        self._shared_tools: Dict[str, AgentToolSpec] = {}
        self._recent_tool_traces = deque(maxlen=_env_int("HELPDESK_TOOL_TRACE_MAX", 200))

        # Agent 池：每种类型可有多个实例（水平扩展）
        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.GENERAL: [self._make_agent(GeneralAgent, client, model, skill_manager)],
            AgentType.DEVICE: [self._make_agent(DeviceAgent, client, model, skill_manager)],
            AgentType.PERMISSION: [self._make_agent(PermissionAgent, client, model, skill_manager)],
            AgentType.NETWORK: [self._make_agent(NetworkAgent, client, model, skill_manager)],
            AgentType.ESCALATION: [self._make_agent(EscalationAgent, client, model, skill_manager)],
        }
        self.set_shared_tools(build_shared_rag_tools(rag_tool_manager))

    @staticmethod
    def _make_agent(
        agent_cls: type[BaseAgent],
        client: AsyncAnthropic,
        default_model: str,
        skill_manager: Optional[Any],
    ) -> BaseAgent:
        """按角色创建 Agent，并允许用环境变量覆盖该角色的模型。

        可使用更强模型，通用接待可使用更快模型，升级节点本身不需要调用 LLM。
        """
        profile = agent_cls.profile
        env_name = f"HELPDESK_{agent_cls.agent_type.value.upper()}_MODEL"
        model = os.getenv(env_name, "").strip() or profile.model
        configured_profile = replace(profile, model=model) if model else profile
        return agent_cls(client, default_model, skill_manager, profile=configured_profile)

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
        """更新 SkillManager 引用，供运行时重载或测试替换使用。"""
        self._skill_manager = skill_manager
        self._composer._skill_manager = skill_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._skill_manager = skill_manager

    def set_shared_tools(self, tools: Optional[Dict[str, AgentToolSpec]]) -> None:
        """更新所有 Agent 共享的工具白名单。"""
        self._shared_tools = dict(tools or {})
        for agents in self._pool.values():
            for agent in agents:
                agent.set_shared_tools(self._shared_tools)

    async def recognize_intent(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        """对外暴露意图识别，供 API 层先判断是否需要 RAG 等前置能力。"""
        return await self._intent_recognizer.recognize(message, history=history)

    def _record_tool_trace(self, result: OrchestratorResult) -> None:
        trace = {
            "request_id": result.request_id,
            "timestamp": datetime.now().isoformat(),
            "intent": result.intent.value if result.intent else None,
            "primary_agent": result.primary_agent.value if result.primary_agent else None,
            "supporting_agents": [agent.value for agent in result.supporting_agents],
            "tools_used": list(result.tools_used),
            "tool_calls": list(result.tool_traces),
            "escalated": result.escalated,
            "latency_ms": round(result.latency_ms, 1),
        }
        self._recent_tool_traces.append(trace)

    def get_tool_trace(self, request_id: str) -> Optional[Dict[str, Any]]:
        for trace in reversed(self._recent_tool_traces):
            if trace.get("request_id") == request_id:
                return trace
        return None

    def get_recent_tool_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self._recent_tool_traces:
            return []
        limit = max(1, min(int(limit or 20), len(self._recent_tool_traces)))
        return list(reversed(list(self._recent_tool_traces)[-limit:]))

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def run(self, req: Request) -> OrchestratorResult:
        """
        处理一次请求的完整流程（编排器主入口）

        执行流程：
        1. 意图识别（如果调用方未识别）
           - 分析用户想做什么
           - 提取结构化实体（订单号、金额等）
           - 判断紧急程度

        2. 路由决策
           - 根据意图选择 Agent
           - 检查是否需要多 Agent 协作
           - 评估是否需要先澄清需求

        3. 执行 Agent（含自动降级）
           - 调用选中的 Agent 处理
           - 失败时自动降级到 GeneralAgent

        4. 升级检查
           - Agent 建议升级
           - 或紧急度为 CRITICAL
           - 或意图为 ESCALATION/HUMAN_HANDOFF
           → 标记为需要人工介入

        5. 记录追踪
           - 工具调用明细
           - 路由决策理由
           - 用于调试和监控

        参数:
            req: 用户请求，包含消息、用户ID、会话ID、上下文等

        返回:
            OrchestratorResult: 包含回复、Agent类型、是否升级等完整信息
        """
        t0 = time.monotonic()

        # 1. 意图识别（如果调用方已识别则跳过）
        # API 层可能提前识别意图，这里做兜底处理
        if req.intent is None:
            intent_result = await self._intent_recognizer.recognize(req.message, history=req.history)
            req.intent  = intent_result.intent
            req.intent_group = intent_result.intent_group
            req.urgency = intent_result.urgency
            req.intent_confidence = intent_result.confidence

        # 2. 低置信度 OTHER 意图：先澄清需求，避免误路由
        if self._needs_clarification(req):
            result = OrchestratorResult(
                request_id=req.request_id,
                response="我还不能确定您要处理的是哪类问题。请补充一下是订单物流、退款账单、账户资料，还是技术故障？",
                agent_type=AgentType.GENERAL,
                intent=req.intent,
                escalated=False,
                latency_ms=(time.monotonic() - t0) * 1000,
                agent_types=[AgentType.GENERAL],
                primary_agent=AgentType.GENERAL,
                routing_reason="低置信度 OTHER 意图，先澄清用户需求",
                routing_confidence=req.intent_confidence,
            )
            self._record_tool_trace(result)
            return result

        # 3. 路由决策：选择最合适的 Agent（单个或多个）
        decision = self._route_decision(req)

        # 复杂问题自动并行协作
        # 例如："登录报错且被重复扣款" 同时派发给技术和账单 Agent
        if decision.multi_agent:
            return await self.run_parallel(req, decision)

        # 4. 执行主 Agent（含自动降级）
        response = await self._execute(req, decision.primary_agent)

        # 5. 升级检查：判断是否需要转人工
        escalated = False
        if response.escalate or req.urgency == UrgencyLevel.CRITICAL or req.intent == IntentCategory.IT_ESCALATION:
            escalated = True
            logger.warning(f"请求 {req.request_id} 触发升级: urgency={req.urgency}")
            # 生产环境：此处创建工单、通知IT支持团队

        # 6. 构建返回结果
        result = OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.agent_type,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            agent_types=[response.agent_type],
            primary_agent=decision.primary_agent,
            supporting_agents=[],
            tools_used=list(response.tools_used),
            tool_traces=list(response.tool_traces),
            routing_reason=decision.reason,
            routing_confidence=decision.confidence,
        )

        # 7. 记录工具调用追踪（用于 /trace/tool 接口查询）
        self._record_tool_trace(result)
        return result

    async def run_parallel(self, req: Request, decision: RoutingDecision) -> OrchestratorResult:
        """
        多 Agent 并行协作处理复杂问题

        应用场景：用户问题同时涉及多个专业领域
        例如："登录一直报错，而且还被重复扣款了"需要技术和账单 Agent 同时处理
        """
        # 记录开始时间，用于计算总延迟
        t0 = time.monotonic()

        # 获取所有需要协作的 Agent 类型（主 + 辅助）
        agent_types = decision.agent_types  # [primary, supporting1, supporting2, ...]

        # 为每个 Agent 创建异步任务，准备并行执行
        tasks = [self._execute(req, at) for at in agent_types]

        # 并行执行所有 Agent，return_exceptions=True 防止单个失败影响整体
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 过滤出有效响应，排除异常和失败的结果
        valid_responses = [r for r in responses if isinstance(r, AgentResponse)]

        # 使用 ResponseComposer 智能合并多个 Agent 的回复
        # 会以主 Agent 为主，去重，保持逻辑连贯
        combined = await self._composer.compose(req, valid_responses)

        # 升级检查：任意一个 Agent 建议升级就标记为需要升级
        escalated = any(isinstance(r, AgentResponse) and r.escalate for r in responses)

        # 合并所有 Agent 使用的工具列表，使用 dict.fromkeys 去重并保持顺序
        tools_used = list(dict.fromkeys(
            tool_name
            for response in valid_responses
            for tool_name in response.tools_used
        ))

        # 合并所有 Agent 的工具调用追踪记录（用于调试）
        tool_traces = [
            trace
            for response in valid_responses
            for trace in response.tool_traces
        ]

        # 构建最终结果对象
        result = OrchestratorResult(
            request_id=req.request_id,
            response=combined,                      # 合并后的回复
            agent_type=decision.primary_agent,      # 主 Agent 类型
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,  # 并行执行总延迟
            agent_types=[                           # 实际成功执行的 Agent 列表
                r.agent_type for r in responses
                if isinstance(r, AgentResponse) and r.success
            ] or agent_types,
            primary_agent=decision.primary_agent,
            supporting_agents=decision.supporting_agents,
            tools_used=tools_used,                  # 去重后的工具列表
            tool_traces=tool_traces,                # 完整的工具调用追踪
            routing_reason=decision.reason,
            routing_confidence=decision.confidence,
        )

        # 记录追踪信息到内存队列，供 /trace/tool 接口查询
        self._record_tool_trace(result)
        return result

    async def run_stream(self, req: Request):
        """
        流式处理请求（支持 Server-Sent Events）

        相比 run() 方法：
        - 实时 yield 生成的文本内容
        - 降低首字延迟（从 20s → 1-2s）
        - 最终 yield 完整的元数据

        Yields:
            dict: 包含 type 和 content 的字典
                - {"type": "content", "text": "..."}  # 流式文本
                - {"type": "metadata", "result": OrchestratorResult}  # 最终元数据
        """
        t0 = time.monotonic()

        # 1. 意图识别（如果调用方已识别则跳过）
        if req.intent is None:
            intent_result = await self._intent_recognizer.recognize(req.message, history=req.history)
            req.intent = intent_result.intent
            req.intent_group = intent_result.intent_group
            req.urgency = intent_result.urgency
            req.intent_confidence = intent_result.confidence

        # 2. 低置信度 OTHER 意图：先澄清需求
        if self._needs_clarification(req):
            clarification_text = "我还不能确定您要处理的是哪类问题。请补充一下是订单物流、退款账单、账户资料，还是技术故障？"
            yield {"type": "content", "text": clarification_text}

            result = OrchestratorResult(
                request_id=req.request_id,
                response=clarification_text,
                agent_type=AgentType.GENERAL,
                intent=req.intent,
                escalated=False,
                latency_ms=(time.monotonic() - t0) * 1000,
                agent_types=[AgentType.GENERAL],
                primary_agent=AgentType.GENERAL,
                routing_reason="低置信度 OTHER 意图，先澄清用户需求",
                routing_confidence=req.intent_confidence,
            )
            self._record_tool_trace(result)
            yield {"type": "metadata", "result": result}
            return

        # 3. 路由决策
        decision = self._route_decision(req)

        # 4. 多 Agent 并行协作暂不支持流式（复杂度较高）
        if decision.multi_agent:
            result = await self.run_parallel(req, decision)
            # 一次性返回完整内容
            yield {"type": "content", "text": result.response}
            yield {"type": "metadata", "result": result}
            return

        # 5. 流式执行主 Agent
        full_response = ""
        response = None

        async for chunk in self._execute_stream(req, decision.primary_agent):
            if chunk.get("type") == "content":
                text = chunk.get("text", "")
                full_response += text
                yield {"type": "content", "text": text}
            elif chunk.get("type") == "done":
                response = chunk.get("response")

        # 6. 升级检查
        escalated = False
        if response and (response.escalate or req.urgency == UrgencyLevel.CRITICAL or req.intent == IntentCategory.IT_ESCALATION):
            escalated = True
            logger.warning(f"请求 {req.request_id} 触发升级: urgency={req.urgency}")

        # 7. 构建返回结果
        if response:
            result = OrchestratorResult(
                request_id=req.request_id,
                response=full_response,
                agent_type=response.agent_type,
                intent=req.intent,
                escalated=escalated,
                latency_ms=(time.monotonic() - t0) * 1000,
                agent_types=[response.agent_type],
                primary_agent=decision.primary_agent,
                supporting_agents=[],
                tools_used=list(response.tools_used),
                tool_traces=list(response.tool_traces),
                routing_reason=decision.reason,
                routing_confidence=decision.confidence,
            )
            self._record_tool_trace(result)
            yield {"type": "metadata", "result": result}

    # ── 路由逻辑 ──────────────────────────────────────────────────────────────

    def _route(self, intent: Optional[IntentCategory], urgency: Optional[UrgencyLevel]) -> AgentType:
        """
        三层路由决策：
          1. 意图映射
          2. 紧急度覆盖（CRITICAL 直接升级）
          3. 默认 GENERAL
        """
        if urgency == UrgencyLevel.CRITICAL:
            return AgentType.ESCALATION

        if intent and intent in self._INTENT_ROUTING:
            target = self._INTENT_ROUTING[intent]
            # 如果目标类型有可用实例则使用，否则降级
            if target in self._pool and self._pool[target]:
                return target

        return AgentType.GENERAL

    def _route_decision(self, req: Request) -> RoutingDecision:
        """
        结构化路由决策（核心路由逻辑）

        决策流程：
        1. 紧急路由：CRITICAL 直接升级
        2. 明确升级：ESCALATION/HUMAN_HANDOFF 意图直接升级
        3. 领域评分：按意图、关键词、实体为各 Agent 打分
        4. 主次选择：
           - 主 Agent：得分最高的
           - 辅助 Agent：得分 ≥ 0.45 且 ≥ 主得分*0.55 的其他 Agent

        评分机制（_domain_scores）：
        - 意图匹配：直接命中 +0.75
        - 关键词匹配：每个关键词 +0.18（上限 0.45）
        - 实体匹配：error_code +0.2, amount +0.15, order_id +0.1

        设计思路：
        - 先处理确定性场景（紧急、升级）
        - 再用评分处理模糊场景（可能涉及多个领域）
        - 支持主辅协作（一个问题涉及多个专业领域）

        参数:
            req: 用户请求

        返回:
            RoutingDecision: 包含主 Agent、辅助 Agent 列表、路由理由、置信度
        """
        # 1. 紧急路由：CRITICAL 优先级最高，直接升级处理
        if req.urgency == UrgencyLevel.CRITICAL:
            return RoutingDecision(
                primary_agent=AgentType.ESCALATION,
                reason="紧急度为 CRITICAL，触发升级处理",
                confidence=1.0,
            )

        # 2. 明确升级：用户明确要求转人工或创建工单
        if req.intent == IntentCategory.IT_ESCALATION:
            return RoutingDecision(
                primary_agent=AgentType.ESCALATION,
                reason=f"意图为 {req.intent.value if req.intent else 'unknown'}，触发工单升级",
                confidence=max(req.intent_confidence, 0.9),
            )

        # 3. 领域评分：为每个 Agent 类型打分
        scores = self._domain_scores(req)

        # 只保留可用的 Agent（GENERAL 始终可用，其他需要检查是否有实例）
        available_scores = {
            agent_type: score
            for agent_type, score in scores.items()
            if agent_type == AgentType.GENERAL or self._pool.get(agent_type)
        }

        # 降级兜底：没有可用 Agent 时降级到 GeneralAgent
        if not available_scores:
            return RoutingDecision(
                primary_agent=AgentType.GENERAL,
                reason="无可用专属 Agent，降级到 GeneralAgent",
                confidence=0.1,
            )

        # 4. 按得分排序，选择主 Agent 和辅助 Agent
        ordered = sorted(available_scores.items(), key=lambda item: item[1], reverse=True)
        primary_agent, primary_score = ordered[0]

        # 辅助 Agent 筛选条件：
        # - 不能是 GENERAL（通用 Agent 不作为辅助）
        # - 得分 ≥ 0.45（有一定相关性）
        # - 得分 ≥ 主得分*0.55（与主 Agent 相关性不能太低）
        supporting_agents = [
            agent_type
            for agent_type, score in ordered[1:]
            if agent_type != AgentType.GENERAL and score >= 0.45 and score >= primary_score * 0.55
        ]

        # 5. 生成路由理由（用于调试和监控）
        reason = self._routing_reason(req, available_scores, primary_agent, supporting_agents)

        return RoutingDecision(
            primary_agent=primary_agent,
            supporting_agents=supporting_agents,
            reason=reason,
            confidence=round(min(primary_score, 1.0), 3),
        )

    def _domain_scores(self, req: Request) -> Dict[AgentType, float]:
        """
        为各领域 Agent 打分（路由评分核心算法）

        评分维度：
        1. 意图匹配（75%）：直接命中对应意图 +0.75
        2. 关键词匹配（18%）：每个关键词命中 +0.18（上限 0.45）
        3. 实体匹配（最高 20%）：
           - 有 device_id/error_code → Device +0.2
           - 有 employee_id → Permission +0.15
           - 有 ip_address → Network +0.15

        设计思路：
        - 意图权重最高：明确意图时直接选定 Agent
        - 关键词补充：意图模糊时用关键词辅助判断
        - 实体增强：结构化信息提供额外信号

        返回: {AgentType: 得分}, 得分范围 [0, ~1.2]
        """
        msg = req.message.lower()

        # 初始分数
        scores = {
            AgentType.GENERAL: 0.1,        # 通用 Agent 保底分
            AgentType.DEVICE: 0.0,
            AgentType.PERMISSION: 0.0,
            AgentType.NETWORK: 0.0,
        }

        # 1. 意图匹配评分
        # 设备类意图
        if req.intent in (
            IntentCategory.DEVICE_FAILURE,
            IntentCategory.DEVICE_SETUP,
            IntentCategory.SOFTWARE_INSTALL,
            IntentCategory.HARDWARE_REQUEST,
        ):
            scores[AgentType.DEVICE] += 0.75

        # 权限类意图
        if req.intent in (
            IntentCategory.PERMISSION_REQUEST,
            IntentCategory.ACCOUNT_ISSUE,
            IntentCategory.VPN_ACCESS,
        ):
            scores[AgentType.PERMISSION] += 0.75

        # 网络类意图
        if req.intent in (
            IntentCategory.NETWORK_ISSUE,
            IntentCategory.WIFI_PROBLEM,
            IntentCategory.NETWORK_SLOW,
        ):
            scores[AgentType.NETWORK] += 0.75

        # 通用类意图
        if req.intent in (
            IntentCategory.QUERY,
            IntentCategory.CONTACT_INFO,
            IntentCategory.GREETING,
            IntentCategory.FEEDBACK,
            IntentCategory.COMPLAINT,
            IntentCategory.OTHER,
        ):
            scores[AgentType.GENERAL] += 0.55

        # 2. 关键词匹配评分
        device_kws = ["电脑", "笔记本", "台式机", "打印机", "显示器", "黑屏", "蓝屏", "死机", "软件", "安装", "office"]
        permission_kws = ["权限", "申请", "开通", "账号", "密码", "登录", "vpn", "访问"]
        network_kws = ["网络", "wifi", "wi-fi", "断网", "网速", "连不上", "ip", "ping"]
        general_kws = ["怎么", "哪里", "电话", "联系", "咨询", "帮助"]

        # 统计关键词命中数
        device_hits = sum(1 for kw in device_kws if kw in msg)
        permission_hits = sum(1 for kw in permission_kws if kw in msg)
        network_hits = sum(1 for kw in network_kws if kw in msg)
        general_hits = sum(1 for kw in general_kws if kw in msg)

        # 每个关键词 +0.18 分，但有上限
        scores[AgentType.DEVICE] += min(0.45, device_hits * 0.18)
        scores[AgentType.PERMISSION] += min(0.45, permission_hits * 0.18)
        scores[AgentType.NETWORK] += min(0.45, network_hits * 0.18)
        scores[AgentType.GENERAL] += min(0.35, general_hits * 0.12)

        # 3. 实体匹配评分
        entities = req.entities or {}
        if entities.get("device_id") or entities.get("error_code") or entities.get("software"):
            scores[AgentType.DEVICE] += 0.2  # 有设备信息说明是设备问题
        if entities.get("employee_id") or entities.get("email"):
            scores[AgentType.PERMISSION] += 0.15   # 有员工信息说明可能是权限问题
        if entities.get("ip_address"):
            scores[AgentType.NETWORK] += 0.15   # 有IP地址说明是网络问题

        # 四舍五入到 3 位小数
        return {agent_type: round(score, 3) for agent_type, score in scores.items()}

    @staticmethod
    def _routing_reason(
        req: Request,
        scores: Dict[AgentType, float],
        primary_agent: AgentType,
        supporting_agents: List[AgentType],
    ) -> str:
        score_text = ", ".join(
            f"{agent_type.value}={score:.2f}"
            for agent_type, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        )
        support_text = ", ".join(agent.value for agent in supporting_agents) or "none"
        intent = req.intent.value if req.intent else "unknown"
        return (
            f"intent={intent}, group={req.intent_group or 'unknown'}, "
            f"primary={primary_agent.value}, supporting={support_text}, scores=[{score_text}]"
        )

    def _collaboration_targets(self, req: Request) -> List[AgentType]:
        """
        判断是否需要多个 Agent 并行协作。

        意图识别通常只返回一个主意图；这里用领域关键词补充检测复合问题，
        例如"电脑连不上网且VPN也登录不了"需要设备和网络 Agent 同时处理。
        """
        msg = req.message.lower()
        targets: List[AgentType] = []

        device_kws = ["电脑", "打印机", "软件", "安装", "故障"]
        permission_kws = ["权限", "账号", "密码", "vpn", "登录"]
        network_kws = ["网络", "wifi", "断网", "网速", "连不上"]

        if req.intent in (
            IntentCategory.DEVICE_FAILURE,
            IntentCategory.DEVICE_SETUP,
            IntentCategory.SOFTWARE_INSTALL,
        ) or any(kw in msg for kw in device_kws):
            targets.append(AgentType.DEVICE)
        if req.intent in (
            IntentCategory.PERMISSION_REQUEST,
            IntentCategory.ACCOUNT_ISSUE,
            IntentCategory.VPN_ACCESS,
        ) or any(kw in msg for kw in permission_kws):
            targets.append(AgentType.PERMISSION)
        if req.intent in (
            IntentCategory.NETWORK_ISSUE,
            IntentCategory.WIFI_PROBLEM,
            IntentCategory.NETWORK_SLOW,
        ) or any(kw in msg for kw in network_kws):
            targets.append(AgentType.NETWORK)

        # 保持顺序去重，并只返回当前有实例的 Agent 类型。
        deduped = list(dict.fromkeys(targets))
        return [agent_type for agent_type in deduped if self._pool.get(agent_type)]

    @staticmethod
    def _needs_clarification(req: Request) -> bool:
        """低置信度且无明确意图时，先追问，避免误路由。"""
        if req.intent != IntentCategory.OTHER:
            return False
        text = (req.message or "").strip()
        if len(text) <= 2:
            return False
        return req.intent_confidence < 0.5

    def _best_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """
        性能路由：从同类 Agent 中按 routing_score 加权随机选择。

        策略：
        - 单实例：直接返回
        - 多实例：按 routing_score 加权随机，兼顾性能和探索

        优势：
        - routing_score 高的实例获得更多请求
        - routing_score 低的实例仍有机会积累新统计数据
        - 可以检测故障恢复和性能变化
        """
        agents = self._pool.get(agent_type, [])
        if not agents:
            return None

        # 单实例直接返回
        if len(agents) == 1:
            return agents[0]

        # 多实例按 routing_score 加权随机
        weights = [max(0.01, a.stats.routing_score()) for a in agents]
        total_weight = sum(weights)

        # 所有实例都不可用时（极端情况），随机选择
        if total_weight < 0.1:
            return random.choice(agents)

        # 加权随机选择
        return random.choices(agents, weights=weights, k=1)[0]

    async def _execute(self, req: Request, agent_type: AgentType) -> AgentResponse:
        """执行 Agent，失败时降级到 GeneralAgent。"""
        agent = self._best_agent(agent_type)
        if agent is None:
            agent = self._best_agent(AgentType.GENERAL)
        if agent is None:
            return AgentResponse(
                agent_type=AgentType.GENERAL,
                content="服务暂时不可用，请稍后重试。",
                success=False,
            )

        response = await agent.handle(req)

        # 专属 Agent 失败时降级到 GeneralAgent
        if not response.success and agent_type not in (AgentType.GENERAL, AgentType.ESCALATION):
            logger.warning(f"{agent_type.value} 失败，降级到 GeneralAgent")
            fallback = self._best_agent(AgentType.GENERAL)
            if fallback:
                response = await fallback.handle(req)

        return response

    async def _execute_stream(self, req: Request, agent_type: AgentType):
        """
        流式执行 Agent，失败时降级到 GeneralAgent

        Yields:
            dict: {"type": "content", "text": "..."} 或 {"type": "done", "response": AgentResponse}
        """
        agent = self._best_agent(agent_type)
        if agent is None:
            agent = self._best_agent(AgentType.GENERAL)
        if agent is None:
            yield {"type": "content", "text": "服务暂时不可用，请稍后重试。"}
            yield {
                "type": "done",
                "response": AgentResponse(
                    agent_type=AgentType.GENERAL,
                    content="服务暂时不可用，请稍后重试。",
                    success=False,
                )
            }
            return

        # 流式处理 Agent
        full_content = ""
        response = None
        async for chunk in agent.handle_stream(req):
            if chunk.get("type") == "content":
                full_content += chunk.get("text", "")
                yield chunk
            elif chunk.get("type") == "done":
                response = chunk.get("response")

        # 专属 Agent 失败时降级到 GeneralAgent
        if response and not response.success and agent_type not in (AgentType.GENERAL, AgentType.ESCALATION):
            logger.warning(f"{agent_type.value} 失败，降级到 GeneralAgent")
            fallback = self._best_agent(AgentType.GENERAL)
            if fallback:
                full_content = ""
                async for chunk in fallback.handle_stream(req):
                    if chunk.get("type") == "content":
                        full_content += chunk.get("text", "")
                        yield chunk
                    elif chunk.get("type") == "done":
                        response = chunk.get("response")

        yield {"type": "done", "response": response}

    # ── 统计（供 Monitor 读取）────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                result[key] = {
                    "total":        agent.stats.total,
                    "success_rate": round(agent.stats.success_rate, 3),
                    "avg_ms":       round(agent.stats.avg_ms, 1),
                    "monitor_penalty": round(agent.stats.monitor_penalty, 3),
                    "routing_score": round(agent.stats.routing_score(), 3),
                    "role": agent.profile.role,
                    "workflow": list(agent.profile.workflow),
                    "tool_scope": list(agent.profile.tool_scope),
                    "available_tools": list(agent.get_tools()),
                    "model": agent._model,
                }
        return result

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
        """
        接收 Monitor 的在线表现反馈，动态调整路由惩罚项。

        penalties 的 key 使用 get_stats() 中的 agent key，例如 technical_0。
        """
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                penalty = penalties.get(key, 0.0)
                agent.stats.monitor_penalty = min(max(penalty, 0.0), 0.9)
