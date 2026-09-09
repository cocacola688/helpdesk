"""Agent 工具定义与实现。

所有 Agent 工具集中在这里，编排器只负责：
  1. 根据 Agent 类型暴露工具白名单
  2. 执行 LLM 返回的 tool_use
  3. 将工具结果回传给 LLM

工具本身保持确定性、可测试，并明确区分：
  - 当前请求分析
  - 设备资产查询
  - 权限申请指引
  - 网络诊断
  - 工单创建
  - 共享知识库 RAG

设备维修、权限审批、系统配置变更等需要真实IT系统授权的动作不在这里伪造。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from agents.agent_orchestrator import Request


AgentToolHandler = Callable[["Request", Dict[str, Any]], Union[Any, Awaitable[Any]]]


@dataclass(frozen=True)
class AgentToolSpec:
    """Agent 可见工具的定义和执行函数。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: AgentToolHandler


def make_tool(
    name: str,
    description: str,
    properties: Dict[str, Any],
    handler: AgentToolHandler,
    required: Optional[List[str]] = None,
) -> AgentToolSpec:
    """创建带 JSON Schema 的 Agent 工具。"""
    return AgentToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
        handler=handler,
    )


def inspect_request_context(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用助手工具：返回脱敏后的当前请求快照。"""
    return {
        "intent": req.intent.value if req.intent else None,
        "intent_group": req.intent_group,
        "urgency": req.urgency.name if req.urgency else None,
        "intent_confidence": round(req.intent_confidence, 4),
        "entities": req.entities or {},
        "context_available": bool(req.context),
        "requested_focus": str(args.get("focus", "general"))[:40],
    }


def suggest_required_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用助手工具：按业务类型计算下一轮只需询问的字段。"""
    intent = req.intent.value if req.intent else "other"
    fields: List[str] = []
    if intent in {"device_failure", "device_setup", "software_install"}:
        fields = ["设备型号或编号", "故障现象或错误信息"]
    elif intent in {"permission_request", "account_issue", "vpn_access"}:
        fields = ["工号", "申请的权限或系统"]
    elif intent in {"network_issue", "wifi_problem", "network_slow"}:
        fields = ["所在位置或网络环境", "具体故障现象"]
    elif intent == "other":
        fields = ["希望了解的具体IT问题"]
    return {
        "intent": intent,
        "required_fields": fields,
        "known_entities": req.entities or {},
    }


def lookup_asset(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """设备工具：查询IT资产信息（设备型号、配置、保修状态）。仅为演示，实际应连接IT资产管理系统。"""
    device_id = str(args.get("device_id", "")).upper().strip()
    # 模拟数据
    mock_data = {
        "PC-001": {
            "device_type": "台式电脑",
            "model": "Dell OptiPlex 7090",
            "cpu": "Intel Core i7-11700",
            "ram": "16GB",
            "os": "Windows 11 Pro",
            "purchase_date": "2023-03-15",
            "warranty_status": "在保",
            "assigned_to": "张三 (E001234)",
        },
        "NB-2023-001": {
            "device_type": "笔记本电脑",
            "model": "Lenovo ThinkPad X1 Carbon Gen 9",
            "cpu": "Intel Core i5-1135G7",
            "ram": "16GB",
            "os": "Windows 10 Pro",
            "purchase_date": "2023-01-20",
            "warranty_status": "在保",
            "assigned_to": "李四 (E001235)",
        },
        "PRINTER-05": {
            "device_type": "打印机",
            "model": "HP LaserJet Pro M428fdw",
            "location": "3楼办公区",
            "purchase_date": "2022-06-10",
            "warranty_status": "已过保",
            "last_maintenance": "2024-08-15",
        },
    }
    if device_id in mock_data:
        return {"success": True, "device_id": device_id, **mock_data[device_id]}
    return {
        "success": False,
        "device_id": device_id,
        "error": "未找到该设备信息，请确认设备编号或联系IT资产管理部门",
    }


def lookup_error_code(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """设备工具：查询错误码含义和解决方案。"""
    error_code = str(args.get("error_code", "")).upper().strip()
    # 常见错误码库
    error_db = {
        "0X80070005": {
            "description": "拒绝访问 - 权限不足",
            "common_causes": ["文件/文件夹权限不足", "注册表权限问题", "需要管理员权限"],
            "solutions": ["以管理员身份运行", "检查文件权限设置", "联系IT申请权限"],
        },
        "ERR_CONNECTION_REFUSED": {
            "description": "连接被拒绝 - 无法连接到服务器",
            "common_causes": ["服务器未运行", "防火墙阻止", "网络配置错误"],
            "solutions": ["检查网络连接", "确认服务器地址正确", "检查防火墙设置"],
        },
        "0X80004005": {
            "description": "未指定的错误",
            "common_causes": ["文件损坏", "系统文件缺失", "权限问题"],
            "solutions": ["运行系统文件检查 (sfc /scannow)", "检查磁盘错误", "重启电脑"],
        },
    }
    if error_code in error_db:
        return {"success": True, "error_code": error_code, **error_db[error_code]}
    return {
        "success": False,
        "error_code": error_code,
        "suggestion": "未找到该错误码的详细信息，建议记录完整错误信息并创建工单寻求技术支持",
    }


def build_diagnostic_plan(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """设备工具：根据故障类型生成诊断计划。"""
    issue_type = str(args.get("issue_type", "")).lower().strip()
    plans = {
        "无法开机": {
            "steps": [
                "1. 检查电源连接是否正常",
                "2. 尝试拔掉所有外接设备后开机",
                "3. 检查显示器连接和电源",
                "4. 听主机是否有异常声音（风扇声、报警声）",
                "5. 如仍无法开机，创建工单申请现场支持",
            ],
            "estimated_time": "5-10分钟",
        },
        "网络连接": {
            "steps": [
                "1. 检查网线是否插好或Wi-Fi是否已连接",
                "2. 尝试ping 内网网关 (通常是192.168.x.1)",
                "3. 运行 ipconfig /release 和 ipconfig /renew 重新获取IP",
                "4. 检查防火墙设置是否阻止了网络",
                "5. 尝试重启电脑和路由器",
            ],
            "estimated_time": "10-15分钟",
        },
        "打印机": {
            "steps": [
                "1. 检查打印机电源和网络连接",
                "2. 清除打印队列中的卡住任务",
                "3. 重启打印机和电脑",
                "4. 重新安装打印机驱动",
                "5. 检查是否有卡纸或墨粉不足",
            ],
            "estimated_time": "10-20分钟",
        },
    }
    plan = plans.get(issue_type)
    if plan:
        return {"success": True, "issue_type": issue_type, **plan}
    return {
        "success": False,
        "issue_type": issue_type,
        "message": "未找到该故障类型的诊断计划，建议描述具体故障现象并创建工单",
    }


def check_permission_requirements(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """权限工具：查询权限申请的条件和流程。"""
    permission_type = str(args.get("permission_type", "")).strip()
    mapping = {
        "VPN访问": {
            "conditions": ["正式员工", "有远程办公需求"],
            "required_materials": ["员工工号", "部门负责人审批邮件", "业务需求说明"],
            "approval_flow": "提交申请 → 部门主管审批 → IT安全审批 → 账号开通",
            "estimated_time": "1-2个工作日",
            "notes": "VPN账号有效期为6个月，到期需重新申请",
        },
        "文件共享权限": {
            "conditions": ["需要访问特定共享文件夹"],
            "required_materials": ["员工工号", "文件夹路径", "申请的权限类型（读/写）", "业务理由"],
            "approval_flow": "提交申请 → 文件夹管理员审批 → IT配置权限",
            "estimated_time": "4小时内",
            "notes": "敏感数据访问需要额外的安全审批",
        },
        "系统管理员权限": {
            "conditions": ["IT部门员工或经授权的系统管理员"],
            "required_materials": ["工号", "部门总监审批", "安全培训证明", "详细的权限需求说明"],
            "approval_flow": "提交申请 → 部门总监审批 → IT安全审批 → CISO审批 → 权限开通",
            "estimated_time": "3-5个工作日",
            "notes": "管理员权限需定期审计，每季度需重新确认",
        },
    }
    info = mapping.get(permission_type)
    if info:
        return {"success": True, "permission_type": permission_type, **info}
    return {
        "success": False,
        "permission_type": permission_type,
        "error": "未找到该权限类型的申请信息，请联系IT支持或创建工单咨询",
    }


def get_approval_flow(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """权限工具：获取审批流程详情。"""
    request_type = str(args.get("request_type", "")).strip()
    flows = {
        "普通权限": ["员工提交申请", "直属主管审批", "IT配置", "完成"],
        "敏感权限": ["员工提交申请", "直属主管审批", "IT安全审批", "信息安全官审批", "IT配置", "完成"],
        "系统变更": ["提交变更申请", "技术评审", "部门主管审批", "变更咨询委员会审批", "实施", "验证", "完成"],
    }
    flow = flows.get(request_type, ["提交申请", "相关部门审批", "IT处理", "完成"])
    return {
        "request_type": request_type,
        "approval_flow": flow,
        "note": "具体审批时间取决于审批人响应速度，紧急情况请标注优先级",
    }


def network_diagnostic(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """网络工具：执行基础网络诊断。"""
    diagnostic_type = str(args.get("diagnostic_type", "connectivity")).lower().strip()
    results = {
        "connectivity": {
            "test_name": "网络连通性测试",
            "checks": [
                {"item": "本地连接状态", "status": "正常", "details": "已连接到公司网络"},
                {"item": "DNS解析", "status": "正常", "details": "可以正常解析域名"},
                {"item": "网关连通性", "status": "正常", "details": "可以ping通默认网关"},
                {"item": "互联网访问", "status": "正常", "details": "可以访问外网"},
            ],
            "conclusion": "网络连接正常",
        },
        "vpn": {
            "test_name": "VPN连接诊断",
            "checks": [
                {"item": "VPN客户端状态", "status": "正常", "details": "客户端已安装且为最新版本"},
                {"item": "VPN服务器连通性", "status": "正常", "details": "可以连接到VPN服务器"},
                {"item": "认证状态", "status": "待检查", "details": "请确认用户名和密码正确"},
                {"item": "证书状态", "status": "正常", "details": "证书有效期至2025-12-31"},
            ],
            "conclusion": "VPN配置正常，如无法连接请检查账号密码",
        },
        "wifi": {
            "test_name": "Wi-Fi连接诊断",
            "checks": [
                {"item": "Wi-Fi适配器", "status": "正常", "details": "驱动正常，适配器已启用"},
                {"item": "信号强度", "status": "中等", "details": "信号强度 -65 dBm"},
                {"item": "SSID连接", "status": "已连接", "details": "已连接到 CompanyWiFi"},
                {"item": "IP地址", "status": "正常", "details": "已获取IP地址"},
            ],
            "conclusion": "Wi-Fi连接正常，信号中等，建议靠近接入点以获得更好性能",
        },
    }
    result = results.get(diagnostic_type)
    if result:
        return {"success": True, "diagnostic_type": diagnostic_type, **result}
    return {
        "success": False,
        "diagnostic_type": diagnostic_type,
        "error": "不支持的诊断类型，支持的类型：connectivity, vpn, wifi",
    }


def check_vpn_status(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """网络工具：检查VPN账号状态。"""
    employee_id = str(args.get("employee_id", "")).strip()
    if not employee_id:
        return {"success": False, "error": "请提供工号"}
    # 模拟数据
    return {
        "success": True,
        "employee_id": employee_id,
        "vpn_status": "正常",
        "account_active": True,
        "expiry_date": "2025-06-30",
        "last_login": "2024-05-10 14:30:25",
        "note": "VPN账号正常，如无法连接请检查网络和客户端版本",
    }


def create_ticket(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """升级工具：创建IT工单。"""
    import uuid
    ticket_id = f"IT-{uuid.uuid4().hex[:8].upper()}"
    title = str(args.get("title", "IT支持请求")).strip()
    description = str(args.get("description", req.message)).strip()
    priority = str(args.get("priority", "medium")).lower()

    priority_map = {
        "critical": "紧急",
        "high": "高",
        "medium": "中",
        "low": "低",
    }

    return {
        "success": True,
        "ticket_id": ticket_id,
        "title": title,
        "description": description[:200] + "..." if len(description) > 200 else description,
        "priority": priority_map.get(priority, "中"),
        "status": "已创建",
        "created_at": "2024-05-15 10:30:00",
        "assigned_to": "IT支持团队",
        "estimated_response": "根据优先级，预计在30分钟至4小时内响应",
        "tracking_url": f"https://itsupport.company.com/tickets/{ticket_id}",
        "note": "工单已创建，技术人员将尽快处理。您可以通过跟踪链接查看进度。",
    }


def get_it_contacts(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """升级工具：获取IT支持联系方式。"""
    return {
        "helpdesk_phone": "400-XXX-XXXX（24小时）",
        "helpdesk_email": "itsupport@company.com",
        "wechat": "IT服务台（企业微信）",
        "portal": "https://itsupport.company.com",
        "office_hours": "周一至周五 9:00-18:00（现场支持）",
        "after_hours": "非工作时间提供远程支持（电话/邮件）",
        "emergency_contact": "IT经理：138-XXXX-XXXX（紧急情况）",
        "note": "一般问题请通过服务台提交工单，紧急情况可直接致电",
    }


def build_shared_rag_tools(tool_manager: Any) -> Dict[str, AgentToolSpec]:
    """构建所有 Agent 可共享的 RAG 工具。"""

    async def search_knowledge_base(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or req.message or "").strip()
        top_k = int(args.get("top_k", 5) or 5)
        if not query:
            return {"success": False, "error": "query 不能为空", "results": []}
        if tool_manager is None:
            return {"success": False, "error": "RAG 工具未初始化", "results": []}

        result = await tool_manager.search_with_rewrite(
            "knowledge_search",
            query,
            top_k=top_k,
        )
        if not getattr(result, "success", False):
            return {
                "success": False,
                "query": query,
                "error": getattr(result, "error", "知识库检索失败"),
                "results": [],
                "reranked": False,
            }

        return {
            "success": True,
            "query": query,
            "top_k": top_k,
            "results": result.data,
            "reranked": bool(getattr(result, "reranked", False)),
        }

    return {
        "search_knowledge_base": make_tool(
            "search_knowledge_base",
            "检索IT知识库并返回最相关的文档片段；可用于设备故障、网络问题、权限申请等各类IT场景。",
            {
                "query": {"type": "string", "description": "员工问题或检索关键词"},
                "top_k": {"type": "integer", "description": "返回结果条数"},
            },
            search_knowledge_base,
            required=["query"],
        )
    }


def general_tools() -> Dict[str, AgentToolSpec]:
    return {
        "inspect_request_context": make_tool(
            "inspect_request_context",
            "查看当前请求的意图、紧急度、实体和上下文可用性；不查询外部业务系统。",
            {"focus": {"type": "string", "description": "希望关注的业务方向"}},
            inspect_request_context,
        ),
        "suggest_required_fields": make_tool(
            "suggest_required_fields",
            "根据当前意图建议下一轮只需向员工补充的字段。",
            {},
            suggest_required_fields,
        ),
    }


def device_tools() -> Dict[str, AgentToolSpec]:
    """设备支持工具。"""
    return {
        "lookup_asset": make_tool(
            "lookup_asset",
            "查询IT资产信息，包括设备型号、配置、保修状态。仅为演示，实际应连接IT资产管理系统。",
            {"device_id": {"type": "string", "description": "设备编号，例如 PC-001、NB-2023-001"}},
            lookup_asset,
            required=["device_id"],
        ),
        "lookup_error_code": make_tool(
            "lookup_error_code",
            "查询错误码含义和常见解决方案。",
            {"error_code": {"type": "string", "description": "错误码，例如 0x80070005、ERR_CONNECTION_REFUSED"}},
            lookup_error_code,
            required=["error_code"],
        ),
        "build_diagnostic_plan": make_tool(
            "build_diagnostic_plan",
            "根据故障类型生成详细的诊断步骤计划。",
            {"issue_type": {"type": "string", "description": "故障类型，例如 无法开机、网络连接、打印机"}},
            build_diagnostic_plan,
            required=["issue_type"],
        ),
    }


def permission_tools() -> Dict[str, AgentToolSpec]:
    """权限申请工具。"""
    return {
        "check_permission_requirements": make_tool(
            "check_permission_requirements",
            "查询权限申请的条件、材料、流程和审批时间。",
            {"permission_type": {"type": "string", "description": "权限类型，例如 VPN访问、文件共享权限、系统管理员权限"}},
            check_permission_requirements,
            required=["permission_type"],
        ),
        "get_approval_flow": make_tool(
            "get_approval_flow",
            "获取详细的审批流程步骤。",
            {"request_type": {"type": "string", "description": "申请类型，例如 普通权限、敏感权限、系统变更"}},
            get_approval_flow,
            required=["request_type"],
        ),
    }


def network_tools() -> Dict[str, AgentToolSpec]:
    """网络支持工具。"""
    return {
        "network_diagnostic": make_tool(
            "network_diagnostic",
            "执行网络诊断检查，包括连通性、VPN、Wi-Fi等。",
            {"diagnostic_type": {"type": "string", "description": "诊断类型：connectivity（连通性）、vpn（VPN）、wifi（Wi-Fi）"}},
            network_diagnostic,
            required=["diagnostic_type"],
        ),
        "check_vpn_status": make_tool(
            "check_vpn_status",
            "检查员工的VPN账号状态和有效期。",
            {"employee_id": {"type": "string", "description": "员工工号"}},
            check_vpn_status,
            required=["employee_id"],
        ),
    }


def escalation_tools() -> Dict[str, AgentToolSpec]:
    """升级处理工具。"""
    return {
        "create_ticket": make_tool(
            "create_ticket",
            "创建IT支持工单。",
            {
                "title": {"type": "string", "description": "工单标题"},
                "description": {"type": "string", "description": "问题详细描述"},
                "priority": {"type": "string", "description": "优先级：critical（紧急）、high（高）、medium（中）、low（低）"},
            },
            create_ticket,
            required=["title"],
        ),
        "get_it_contacts": make_tool(
            "get_it_contacts",
            "获取IT支持联系方式（服务台电话、邮箱、企业微信等）。",
            {},
            get_it_contacts,
        ),
    }
