"""
IT HelpDesk 系统 — FastAPI 入口

启动时打印小熊饼干图案。
所有核心组件在 lifespan 中初始化，通过环境变量配置。
"""
import asyncio
import json
import logging
import os
import pathlib
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional


_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = r"""
    📚  🎓  🏫
   ╔══════════════════════╗
   ║  IT Helpdesk v1.0    ║
   ║   IT Helpdesk 系统    ║
   ╚══════════════════════╝
    📚  🎓  🏫
"""

# ── 全局组件（lifespan 中初始化）─────────────────────────────────────────────
_orchestrator = None
_memory       = None
_tool_manager = None
_monitor      = None
_evaluator    = None
_skill_manager = None

def _anthropic_cfg() -> Dict[str, Any]:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY")
    cfg: Dict[str, Any] = {
        "api_key":  key,
        "model":    os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip(),
    }
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if base_url:
        cfg["base_url"] = base_url
    return cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 应用生命周期管理

    启动时初始化所有核心组件：
    1. IntentRecognizer - 意图识别器
    2. SkillManager - 业务规则热加载器
    3. AgentOrchestrator - 多 Agent 编排器
    4. MemoryManager - 三级记忆管理（Redis + ChromaDB）
    5. MCPToolManager + KnowledgeBase - RAG 知识库
    6. PerformanceMonitor - 性能监控和告警
    7. EndToEndEvaluator - 端到端评测器

    关闭时清理资源：
    - 停止监控任务
    - 关闭 Redis 连接
    - 释放其他资源

    设计思路：
    - 所有组件都通过环境变量配置，便于容器化部署
    - 组件初始化失败会抛出异常，阻止服务启动（快速失败）
    - 使用全局变量存储组件实例，便于路由函数访问
    """
    global _orchestrator, _memory, _tool_manager, _monitor, _evaluator, _skill_manager

    # 打印启动横幅
    print(BANNER, flush=True)

    # 导入所有核心组件（延迟导入，避免循环依赖）
    from agents.agent_orchestrator import AgentOrchestrator, Request, build_shared_rag_tools
    from core.intent_recognizer import IntentRecognizer
    from evaluation.evaluator import EndToEndEvaluator
    from mcp.knowledge_base import KnowledgeBase
    from mcp.tool_manager import MCPToolManager, Tool
    from memory.conversation_memory import MemoryManager
    from monitor.performance_monitor import PerformanceMonitor
    from core.skill_loader import SkillManager

    # 读取 Anthropic API 配置
    cfg = _anthropic_cfg()
    logger.info(f"模型: {cfg['model']}  base_url: {cfg.get('base_url', '(官方)')}")

    # 1. 意图识别器（Orchestrator 内部也会创建，这里单独暴露给 Evaluator）
    recognizer = IntentRecognizer(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # 2. Skills 管理器：启动时从目录加载业务能力说明
    # Skills 会在 Agent 调用 LLM 时动态注入到 system prompt
    skills_dir = os.getenv("HELPDESK_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills"))
    _skill_manager = SkillManager(
        root_dir=skills_dir,
        max_prompt_chars=int(os.getenv("HELPDESK_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    _skill_manager.load()

    # 3. Agent 编排器（系统核心）
    _orchestrator = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=_skill_manager,
    )

    # 4. 记忆管理器（三级架构）
    # - 工作记忆：Redis（最近对话）
    # - 情景记忆：ChromaDB（历史对话语义检索）
    # - 用户画像：ChromaDB（长期偏好和实体）
    _memory = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # 5. MCP 工具管理器 + RAG 知识库
    _tool_manager = MCPToolManager(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # 初始化知识库（基于 ChromaDB 的向量检索）
    kb = KnowledgeBase(
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
    )
    logger.info(f"知识库已加载: {await kb.doc_count_async()} 个文档片段")

    # 定义知识库降级策略（知识库不可用时的友好提示）
    def knowledge_fallback(params: Dict[str, Any], context: Optional[Dict[str, Any]], error: str):
        query = params.get("query", "")
        return [{
            "title": "知识库降级结果",
            "content": f"知识库暂时不可用，未能完成对\"{query}\"的语义检索。请稍后重试，或转人工客服确认。",
            "score": 0.0,
            "fallback": True,
            "error": error,
        }]

    # 注册知识库工具（支持缓存、重排、降级、分类过滤）
    _tool_manager.register(Tool(
        name="knowledge_search",
        description="搜索知识库（基于 ChromaDB 向量检索，支持按分类过滤）",
        handler=kb.search_handler,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "查询文本"},
                "top_k": {"type": "integer", "description": "返回数量，默认 5"},
                "category": {
                    "type": "string",
                    "description": "分类过滤（可选）：device（设备）/permission（权限）/network（网络）/general（通用）/support（支持）",
                    "enum": ["device", "permission", "network", "general", "support"]
                },
                "domain": {"type": "string", "description": "领域过滤（可选）：device_failure/network_issue/permission_request/vpn_access等"},
            },
            "required": ["query"],
        },
        cache_ttl=300.0,           # 缓存 5 分钟
        supports_rerank=True,      # 支持结果重排
        fallback=knowledge_fallback,
    ))

    # 将知识库工具注入到所有 Agent
    if _orchestrator is not None:
        _orchestrator.set_shared_tools(build_shared_rag_tools(_tool_manager))

    # 6. 性能监控器（可选启动 Prometheus）
    prom_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    _monitor = PerformanceMonitor(
        orchestrator=_orchestrator,
        tool_manager=_tool_manager,
        interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),
        webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
        prometheus_port=prom_port,
    )
    await _monitor.start()

    # 7. 评测器（用于回归测试和质量保证）
    _evaluator = EndToEndEvaluator(
        orchestrator=_orchestrator,
        recognizer=recognizer,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        baseline_path=os.getenv("EVAL_BASELINE_PATH", "/app/data/eval/baseline.json"),
    )

    logger.info("IT Helpdesk 已就绪")

    # yield 之后的代码在应用关闭时执行
    yield

    # 清理资源
    await _monitor.stop()
    if _memory is not None:
        await _memory.close()
    logger.info("IT Helpdesk 已关闭")


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="IT Helpdesk 智能IT支持系统",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:     str
    user_id:     str = "anonymous"
    conv_id:     Optional[str] = None


class ChatResponse(BaseModel):
    conv_id:     str
    request_id:  str = ""
    response:    str
    intent:      str
    intent_group: str = "other"
    agent_type:  str
    agent_types: List[str] = Field(default_factory=list)
    primary_agent: str = ""
    supporting_agents: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0
    escalated:   bool
    latency_ms:  float
    knowledge_used: bool = False
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    intent_confidence: float = 0.0
    intent_source_scores: Dict[str, float] = Field(default_factory=dict)


class ToolTraceResponse(BaseModel):
    request_id: str
    found: bool
    trace: Dict[str, Any] = Field(default_factory=dict)


class RecentToolTracesResponse(BaseModel):
    items: List[Dict[str, Any]] = Field(default_factory=list)


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return {"status": "ok", "agents": _orchestrator.get_stats()}


@app.get("/skills", tags=["Skills"])
async def skills_summary():
    """查看当前已加载的 Skills，便于确认热加载结果和排查解析错误。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return _skill_manager.summary()


@app.post("/skills/reload", tags=["Skills"])
async def reload_skills():
    """运行时重新扫描 Skill 目录，不需要重启服务。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    _skill_manager.reload()
    if _orchestrator is not None:
        _orchestrator.set_skill_manager(_skill_manager)
    return _skill_manager.summary()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    主对话接口（系统核心 API）

    完整处理流程：
    1. 记忆读取
       - 从 Redis 读取工作记忆（最近对话）
       - 从 ChromaDB 检索情景记忆（语义相关历史）
       - 获取用户画像（偏好和常用实体）

    2. 意图识别
       - 三路融合识别用户意图（LLM + Embedding + Pattern）
       - 提取结构化实体（订单号、金额、错误码等）
       - 判断紧急程度

    3. Agent 路由与执行
       - 根据意图选择最合适的 Agent
       - 执行 Agent（可能涉及工具调用）
       - 自动降级和升级检查

    4. 记忆写入
       - 保存用户消息和 Agent 回复
       - 异步更新用户画像（不阻塞响应）

    5. 返回结果
       - 包含回复内容、意图、Agent 类型、工具使用等完整信息

    设计亮点：
    - 会话ID（conv_id）自动生成，支持多轮对话
    - 记忆管理全自动，开发者无需手动维护
    - 异步更新用户画像，不影响响应速度
    - 完整的可观测性（工具追踪、路由原因）

    参数:
        req: ChatRequest，包含 message（用户消息）、user_id、conv_id（可选）

    返回:
        ChatResponse: 包含回复、意图、Agent 类型、是否升级等完整信息
    """
    if _orchestrator is None or _memory is None:
        raise HTTPException(503, "服务未就绪")

    from agents.agent_orchestrator import Request as OrcReq
    from memory.conversation_memory import MsgRole

    # 生成或复用会话 ID
    conv_id = req.conv_id or str(uuid.uuid4())

    # 1. 读取记忆上下文（三级记忆融合）
    mem_ctx = await _memory.get_context(req.user_id, conv_id, query=req.message)

    # 2. 构建对话历史（用于意图识别的上下文）
    # 只取最近 5 轮，避免 prompt 过长
    history = [
        {"role": m.role.value, "content": m.content}
        for m in mem_ctx.recent_messages[-5:]
    ] if mem_ctx.recent_messages else None

    # 3. 意图识别（提前识别，便于 API 层做预处理）
    intent_result = await _orchestrator.recognize_intent(req.message, history=history)
    full_context = mem_ctx.to_prompt_text()

    # 4. 构建编排请求
    orch_req = OrcReq(
        message=req.message,
        user_id=req.user_id,
        conv_id=conv_id,
        context=full_context,           # 格式化的记忆上下文
        history=history,                # 原始对话历史
        entities=intent_result.entities, # 从消息中提取的实体
        intent=intent_result.intent,
        intent_group=intent_result.intent_group,
        urgency=intent_result.urgency,
        intent_confidence=intent_result.confidence,
    )

    # 5. 执行编排（路由 + Agent 处理）
    result = await _orchestrator.run(orch_req)

    # 6. 写入记忆（保存本轮对话）
    await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
    await _memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, result.response)

    # 7. 异步更新用户画像（后台任务，不阻塞响应）
    # 从对话中提炼偏好和实体，存入 ChromaDB
    asyncio.create_task(_memory.update_profile(req.user_id, conv_id))

    # 8. 返回响应
    return ChatResponse(
        conv_id=conv_id,
        request_id=result.request_id,
        response=result.response,
        intent=result.intent.value if result.intent else "other",
        intent_group=intent_result.intent_group,
        agent_type=result.agent_type.value,
        agent_types=[agent_type.value for agent_type in result.agent_types],
        primary_agent=result.primary_agent.value if result.primary_agent else result.agent_type.value,
        supporting_agents=[agent_type.value for agent_type in result.supporting_agents],
        tools_used=result.tools_used,
        routing_reason=result.routing_reason,
        routing_confidence=result.routing_confidence,
        escalated=result.escalated,
        latency_ms=round(result.latency_ms, 1),
        knowledge_used="search_knowledge_base" in result.tools_used,  # 是否使用了知识库
        entities=intent_result.entities,
        intent_confidence=round(intent_result.confidence, 4),
        intent_source_scores=intent_result.source_scores,  # 三路识别的各路得分
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    流式对话接口（支持 Server-Sent Events）

    相比 /chat 接口：
    - 实时返回生成内容，降低首字延迟（从 20s → 1-2s）
    - 使用 SSE 格式，前端可以逐字显示
    - 最终返回完整的元数据（intent、agent_type、tools_used 等）

    事件类型：
    - data: 流式文本内容
    - metadata: 完整的响应元数据（最后发送）
    - error: 错误信息
    """
    if _orchestrator is None or _memory is None:
        raise HTTPException(503, "服务未就绪")

    from agents.agent_orchestrator import Request as OrcReq
    from memory.conversation_memory import MsgRole

    async def generate():
        try:
            # 生成或复用会话 ID
            conv_id = req.conv_id or str(uuid.uuid4())

            # 1. 读取记忆上下文
            mem_ctx = await _memory.get_context(req.user_id, conv_id, query=req.message)

            # 2. 构建对话历史
            history = [
                {"role": m.role.value, "content": m.content}
                for m in mem_ctx.recent_messages[-5:]
            ] if mem_ctx.recent_messages else None

            # 3. 意图识别
            intent_result = await _orchestrator.recognize_intent(req.message, history=history)
            full_context = mem_ctx.to_prompt_text()

            # 4. 构建编排请求
            orch_req = OrcReq(
                message=req.message,
                user_id=req.user_id,
                conv_id=conv_id,
                context=full_context,
                history=history,
                entities=intent_result.entities,
                intent=intent_result.intent,
                intent_group=intent_result.intent_group,
                urgency=intent_result.urgency,
                intent_confidence=intent_result.confidence,
            )

            # 5. 流式执行编排
            full_response = ""
            result = None
            async for chunk in _orchestrator.run_stream(orch_req):
                if chunk.get("type") == "content":
                    text = chunk.get("text", "")
                    full_response += text
                    # 发送流式内容
                    yield f"data: {json.dumps({'type': 'content', 'text': text}, ensure_ascii=False)}\n\n"
                elif chunk.get("type") == "metadata":
                    # 收集最终的元数据
                    result = chunk.get("result")

            # 6. 写入记忆
            await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
            await _memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, full_response)

            # 7. 异步更新用户画像
            asyncio.create_task(_memory.update_profile(req.user_id, conv_id))

            # 8. 发送完整元数据
            if result:
                metadata_payload = {
                    "type": "metadata",
                    "conv_id": conv_id,
                    "request_id": result.request_id,
                    "intent": result.intent.value if result.intent else "other",
                    "intent_group": intent_result.intent_group,
                    "agent_type": result.agent_type.value,
                    "agent_types": [at.value for at in result.agent_types],
                    "primary_agent": result.primary_agent.value if result.primary_agent else result.agent_type.value,
                    "supporting_agents": [at.value for at in result.supporting_agents],
                    "tools_used": result.tools_used,
                    "routing_reason": result.routing_reason,
                    "routing_confidence": result.routing_confidence,
                    "escalated": result.escalated,
                    "latency_ms": round(result.latency_ms, 1),
                    "knowledge_used": "search_knowledge_base" in result.tools_used,
                    "entities": intent_result.entities,
                    "intent_confidence": round(intent_result.confidence, 4),
                    "intent_source_scores": intent_result.source_scores,
                }
                yield f"data: {json.dumps(metadata_payload, ensure_ascii=False)}\n\n"

            # 发送结束标记
            yield "data: {\"type\": \"done\"}\n\n"

        except Exception as e:
            logger.exception("流式对话失败")
            error_payload = {
                "type": "error",
                "error": str(e)
            }
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        }
    )


async def _build_knowledge_context(message: str, intent=None, top_k: int = 3) -> tuple[str, bool]:
    """
    为 /chat 主链路构建 RAG 知识上下文。

    智能路由：根据意图自动选择检索分类，提升检索精准度。
    这里复用 MCPToolManager 的查询改写、并行召回、重排、fallback 能力。
    """
    if _tool_manager is None:
        return "", False
    if not _should_use_knowledge(message, intent=intent):
        return "", False

    # 意图到分类的智能路由映射
    intent_to_category = {
        # 设备类 (device)
        "device_failure": "device",
        "device_setup": "device",
        "software_install": "device",
        "hardware_request": "device",

        # 权限类 (permission)
        "permission_request": "permission",
        "account_issue": "permission",
        "vpn_access": "permission",

        # 网络类 (network)
        "network_issue": "network",
        "wifi_problem": "network",
        "network_slow": "network",

        # 通用查询
        "query": "general",
        "contact_info": "support",
    }

    # 根据意图确定检索分类
    intent_value = getattr(intent, "value", intent)
    category = intent_to_category.get(intent_value)

    try:
        # 调用检索（带分类过滤）
        context_params = {"category": category} if category else {}
        result = await _tool_manager.search_with_rewrite(
            "knowledge_search",
            message,
            top_k=top_k,
            context=context_params  # 传递分类参数
        )

        if not result.success or not isinstance(result.data, list) or not result.data:
            return "", False

        parts = ["[知识库检索结果]"]
        if category:
            parts[0] += f" (已限定范围: {category})"

        used = False
        for i, item in enumerate(result.data[:top_k], start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "未命名文档"))
            content = str(item.get("content", "")).strip()
            score = item.get("score", "")
            if not content:
                continue
            used = True
            parts.append(f"{i}. 标题: {title}\n   相关度: {score}\n   内容: {content[:600]}")

        if not used:
            return "", False
        parts.append("请优先依据以上知识库内容回答；如果知识库内容不足，再结合通用客服能力说明。")
        return "\n".join(parts), True
    except Exception as ex:
        logger.warning(f"构建知识库上下文失败: {ex}")
        return "", False


def _should_use_knowledge(message: str, intent=None) -> bool:
    """跳过纯寒暄，业务类问题才检索知识库，避免无关 RAG 干扰回复。"""
    msg = (message or "").strip().lower()
    if not msg:
        return False
    intent_value = getattr(intent, "value", intent)
    if intent_value in {"greeting", "feedback", "it_escalation", "other"}:
        return False
    if intent_value in {
        "query", "device_failure", "device_setup", "software_install", "hardware_request",
        "permission_request", "account_issue", "vpn_access",
        "network_issue", "wifi_problem", "network_slow",
        "contact_info", "complaint",
    }:
        return True
    greetings = {"你好", "您好", "嗨", "hi", "hello", "hey", "早上好", "晚上好"}
    if msg in greetings:
        return False
    business_keywords = [
        "电脑", "笔记本", "打印机", "软件", "安装", "故障", "蓝屏", "黑屏",
        "权限", "账号", "密码", "vpn", "登录", "申请", "开通",
        "网络", "wifi", "断网", "网速", "连不上", "ip",
        "device", "permission", "network", "software", "install",
    ]
    return len(msg) >= 4 or any(kw in msg for kw in business_keywords)


@app.get("/monitor")
async def monitor_summary():
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议。"""
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    return _monitor.summary()


@app.get("/trace/tool/{request_id}", response_model=ToolTraceResponse)
async def get_tool_trace(request_id: str):
    """查看某次请求的工具调用明细。"""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    trace = _orchestrator.get_tool_trace(request_id)
    return ToolTraceResponse(
        request_id=request_id,
        found=trace is not None,
        trace=trace or {},
    )


@app.get("/trace/tools", response_model=RecentToolTracesResponse)
async def list_recent_tool_traces(limit: int = 20):
    """查看最近 N 次请求的工具调用明细。"""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return RecentToolTracesResponse(items=_orchestrator.get_recent_tool_traces(limit=limit))


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标入口。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search")
async def search(query: str, top_k: int = 5):
    """
    演示检索优化链路：查询改写 → 并行召回 → 重排 → Top-K。
    展示 MCP 工具调用的核心亮点。
    """
    if _tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    result = await _tool_manager.search_with_rewrite("knowledge_search", query, top_k=top_k)
    return {"query": query, "results": result.data, "reranked": result.reranked}


class DocInput(BaseModel):
    """单篇文档输入。"""
    title:   str
    content: str


class BatchDocInput(BaseModel):
    """批量文档导入请求体。"""
    documents: List[DocInput]


class EvalIntentInput(BaseModel):
    """意图识别评测用例。"""
    message: str
    expected_intent: str
    context: Optional[Dict[str, Any]] = None


class EvalDialogInput(BaseModel):
    """对话质量评测用例。question 单轮，turns 多轮。"""
    question: Optional[str] = None
    turns: Optional[List[str]] = None
    user_id: Optional[str] = None
    conv_id: Optional[str] = None


class EvalRunInput(BaseModel):
    """评测请求。为空时使用内置默认用例。"""
    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None


@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(body: BatchDocInput):
    """
    批量导入文档到知识库。

    文档会自动切片（每片 500 字）并存入 ChromaDB，ChromaDB 内置 Embedding 模型自动向量化。

    示例请求体：
    ```json
    {
      "documents": [
        {"title": "退款政策", "content": "用户在购买后 7 天内可以申请无理由退款..."},
        {"title": "配送说明", "content": "标准配送 3-5 个工作日..."}
      ]
    }
    ```
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    count = await kb.add_documents_async([{"title": d.title, "content": d.content} for d in body.documents])
    total = await kb.doc_count_async()
    return {"message": f"成功导入 {count} 个文档片段", "added_chunks": count, "total_chunks": total}


@app.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(file: UploadFile = File(...)):
    """
    上传文件导入知识库。

    支持格式：
    - `.txt` / `.md`：整个文件作为一篇文档，文件名作为标题
    - `.json`：JSON 数组格式 `[{"title": "...", "content": "..."}, ...]`

    文件大小限制：10MB
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    text = content.decode("utf-8", errors="ignore")
    filename = file.filename or "unknown"

    if filename.endswith(".json"):
        import json as _json
        try:
            docs = _json.loads(text)
            if not isinstance(docs, list):
                raise HTTPException(400, "JSON 文件应为数组格式: [{title, content}, ...]")
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
    else:
        # txt / md：整个文件作为一篇文档
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        docs = [{"title": title, "content": text}]

    count = await kb.add_documents_async(docs)
    total = await kb.doc_count_async()
    return {
        "message": f"文件 {filename} 导入成功",
        "added_chunks": count,
        "total_chunks": total,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """查看知识库统计信息（文档片段总数）。"""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    return {"total_chunks": await kb.doc_count_async()}


@app.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None):
    """运行内置评测用例，返回评测报告。"""
    if _evaluator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.evaluator import DEFAULT_DIALOG_CASES, DEFAULT_INTENT_CASES, IntentTestCase

    if body and body.intent_cases is not None:
        intent_cases = [
            IntentTestCase(
                message=c.message,
                expected_intent=c.expected_intent,
                context=c.context,
            )
            for c in body.intent_cases
        ]
    else:
        intent_cases = DEFAULT_INTENT_CASES

    if body and body.dialog_cases is not None:
        dialog_cases = [
            c.model_dump(exclude_none=True)
            for c in body.dialog_cases
        ]
    else:
        dialog_cases = DEFAULT_DIALOG_CASES

    report = await _evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
    )
    return {
        "pass_rate":       report.pass_rate,
        "total":           report.total,
        "passed":          report.passed,
        "avg_scores":      report.avg_scores,
        "regressions":     report.regressions,
        "recommendations": report.recommendations,
        "results": [
            {
                "test_id": r.test_id,
                "passed": r.passed,
                "scores": r.scores,
                "detail": r.detail,
                "metadata": r.metadata,
            }
            for r in report.results
        ],
    }


# ── 交互式 CLI ────────────────────────────────────────────────────────────────
async def _cli():
    print(BANNER)
    print("IT Helpdesk CLI — 输入 quit 退出\n")

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from memory.conversation_memory import MemoryManager, MsgRole
    from core.skill_loader import SkillManager

    cfg = _anthropic_cfg()
    skill_manager = SkillManager(
        root_dir=os.getenv("HELPDESK_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills")),
        max_prompt_chars=int(os.getenv("HELPDESK_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    skill_manager.load()
    orch = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=skill_manager,
    )
    mem  = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "localhost"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/tmp/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    user_id, conv_id = "cli_user", str(uuid.uuid4())

    while True:
        try:
            msg = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 ʕ•ᴥ•ʔ")
            break
        if not msg or msg.lower() in ("quit", "exit", "退出"):
            print("再见 ʕ•ᴥ•ʔ")
            break

        ctx = await mem.get_context(user_id, conv_id, query=msg)
        history = [
            {"role": m.role.value, "content": m.content}
            for m in ctx.recent_messages[-5:]
        ] if ctx.recent_messages else None
        req = Request(message=msg, user_id=user_id, conv_id=conv_id, context=ctx.to_prompt_text(), history=history)
        result = await orch.run(req)

        await mem.add_message(user_id, conv_id, MsgRole.USER, msg)
        await mem.add_message(user_id, conv_id, MsgRole.ASSISTANT, result.response)

        print(f"\nIT Helpdesk [{result.agent_type.value}]: {result.response}\n")

    await mem.close()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        asyncio.run(_cli())
    else:
        uvicorn.run(
            "api.main:app",
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("APP_ENV") == "development",
        )
