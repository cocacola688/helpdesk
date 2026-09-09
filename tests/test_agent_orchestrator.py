import asyncio

from agents.agent_orchestrator import (
    AgentProfile,
    AgentResponse,
    AgentType,
    DeviceAgent,
    PermissionAgent,
    NetworkAgent,
    EscalationAgent,
    GeneralAgent,
    Request,
    ResponseComposer,
    RoutingDecision,
    build_shared_rag_tools,
)
from core.intent_recognizer import IntentCategory, UrgencyLevel


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

        class Messages:
            async def create(inner, **kwargs):
                self.calls.append(kwargs)
                if self.error:
                    raise self.error
                return self.response

        self.messages = Messages()


def make_request(**kwargs):
    values = {
        "message": "我的电脑蓝屏了，显示错误码0x80070005，同时VPN也连不上",
        "user_id": "u1",
        "conv_id": "c1",
        "intent": IntentCategory.DEVICE_FAILURE,
        "intent_group": "device",
        "urgency": UrgencyLevel.HIGH,
        "intent_confidence": 0.92,
        "entities": {"error_code": ["0x80070005"], "device_id": ["PC-001"]},
    }
    values.update(kwargs)
    return Request(**values)


def test_agent_profiles_have_distinct_contracts_and_generation_config():
    assert isinstance(GeneralAgent.profile, AgentProfile)
    assert GeneralAgent.profile.role != DeviceAgent.profile.role
    assert DeviceAgent.profile.workflow != PermissionAgent.profile.workflow
    assert DeviceAgent.profile.temperature < GeneralAgent.profile.temperature
    assert "search_knowledge_base" in GeneralAgent.profile.tool_scope
    assert "lookup_error_code" in DeviceAgent.profile.tool_scope
    assert "check_permission_requirements" in PermissionAgent.profile.tool_scope


def test_domain_agents_build_different_role_packets():
    req = make_request()
    general_packet = GeneralAgent(FakeClient(), "test-model")._build_role_packet(req)
    device_packet = DeviceAgent(FakeClient(), "test-model")._build_role_packet(req)
    permission_packet = PermissionAgent(FakeClient(), "test-model")._build_role_packet(req)

    assert "triage_targets" in general_packet
    assert "diagnostic_fields" in device_packet
    assert "verification_fields" in permission_packet
    assert general_packet != device_packet != permission_packet


def test_escalation_agent_is_a_real_non_llm_handoff_node():
    client = FakeClient()
    agent = EscalationAgent(client, "test-model")

    result = asyncio.run(agent.handle(make_request(
        intent=IntentCategory.IT_ESCALATION,
        urgency=UrgencyLevel.CRITICAL,
    )))

    assert result.success is True
    assert result.escalate is True
    assert "工单" in result.content or "升级" in result.content
    assert client.calls == []


def test_composer_fallback_preserves_primary_and_supporting_results():
    composer = ResponseComposer(FakeClient(error=RuntimeError("provider down")), "test-model")
    req = make_request()
    responses = [
        AgentResponse(AgentType.DEVICE, "先检查设备连接和电源，然后查看错误日志。", True),
        AgentResponse(AgentType.NETWORK, "请检查VPN客户端版本和网络连接状态。", True),
    ]

    content = asyncio.run(composer.compose(req, responses))

    assert "设备" in content or "检查" in content
    assert "补充说明" in content
    assert "VPN" in content or "网络" in content


def test_routing_decision_can_target_escalation_pool():
    # Keep this assertion close to the public data contract used by the API.
    decision = RoutingDecision(
        primary_agent=AgentType.ESCALATION,
        reason="critical request",
        confidence=1.0,
    )
    assert decision.agent_types == [AgentType.ESCALATION]
    assert not decision.multi_agent


def test_agent_tool_scopes_are_real_and_isolated():
    general_tools = set(GeneralAgent(FakeClient(), "test-model").get_tools())
    device_tools = set(DeviceAgent(FakeClient(), "test-model").get_tools())
    permission_tools = set(PermissionAgent(FakeClient(), "test-model").get_tools())
    network_tools = set(NetworkAgent(FakeClient(), "test-model").get_tools())
    escalation_tools = set(EscalationAgent(FakeClient(), "test-model").get_tools())

    assert general_tools == {"inspect_request_context", "suggest_required_fields"}
    assert device_tools == {"lookup_asset", "lookup_error_code", "build_diagnostic_plan"}
    assert permission_tools == {"check_permission_requirements", "get_approval_flow"}
    assert network_tools == {"network_diagnostic", "check_vpn_status"}
    assert escalation_tools == {"create_ticket", "get_it_contacts"}
    assert not general_tools & device_tools
    assert not device_tools & permission_tools
    assert not permission_tools & network_tools


def test_shared_rag_tool_is_available_to_all_agents():
    class RagManager:
        async def search_with_rewrite(self, tool_name, query, top_k=5, context=None):
            return type(
                "Result",
                (),
                {"success": True, "data": [{"title": "VPN连接故障", "content": "检查客户端版本和防火墙设置"}], "reranked": True},
            )()

    shared = build_shared_rag_tools(RagManager())

    general = GeneralAgent(FakeClient(), "test-model")
    device = DeviceAgent(FakeClient(), "test-model")
    permission = PermissionAgent(FakeClient(), "test-model")
    network = NetworkAgent(FakeClient(), "test-model")
    escalation = EscalationAgent(FakeClient(), "test-model")

    for agent in (general, device, permission, network, escalation):
        agent.set_shared_tools(shared)
        tools = agent.get_tools()
        assert "search_knowledge_base" in tools


def test_tool_input_validation_rejects_unknown_fields():
    agent = DeviceAgent(FakeClient(), "test-model")
    spec = agent.get_tools()["lookup_error_code"]

    try:
        agent._validate_tool_input(spec, {"error_code": "0x80070005", "secret": "nope"})
    except ValueError as exc:
        assert "不允许的工具参数" in str(exc)
    else:
        raise AssertionError("unknown tool fields should be rejected")


def test_tool_use_round_trip_executes_only_whitelisted_tool():
    class ToolUseBlock:
        type = "tool_use"
        id = "toolu_1"
        name = "lookup_error_code"
        input = {"error_code": "0x80070005"}

    class TextBlock:
        type = "text"
        text = "已根据错误码0x80070005给出排查建议。"

    class ToolClient:
        def __init__(self):
            self.calls = []
            self.responses = [
                type("Response", (), {"content": [ToolUseBlock()]})(),
                type("Response", (), {"content": [TextBlock()]})(),
            ]

        class Messages:
            def __init__(self, owner):
                self.owner = owner

            async def create(self, **kwargs):
                self.owner.calls.append(kwargs)
                return self.owner.responses.pop(0)

        @property
        def messages(self):
            return self.Messages(self)

    client = ToolClient()
    agent = DeviceAgent(client, "test-model")
    response = asyncio.run(agent.handle(make_request()))

    assert response.success is True
    assert response.tools_used == ["lookup_error_code"]
    assert len(client.calls) == 2
    assert {tool["name"] for tool in client.calls[0]["tools"]} == {
        "lookup_asset",
        "lookup_error_code",
        "build_diagnostic_plan",
    }
    assert "tool_result" in str(client.calls[1]["messages"])

