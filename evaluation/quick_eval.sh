#!/bin/bash
# IT Helpdesk 快速评测脚本
# 使用curl直接测试，无需Python依赖

echo "============================================================"
echo "🖥️  IT Helpdesk 意图识别评测"
echo "============================================================"
echo ""

# 测试用例
declare -a tests=(
    "我的电脑蓝屏了:device_failure"
    "需要安装Office软件:software_install"
    "申请VPN权限:vpn_access"
    "忘记密码了:account_issue"
    "网络连不上:network_issue"
    "Wi-Fi信号很弱:wifi_problem"
    "打印机卡纸:device_failure"
    "需要文件共享权限:permission_request"
    "帮我创建工单:it_escalation"
    "电脑开不了机:device_failure"
    "你好:greeting"
    "IT支持联系方式:contact_info"
)

correct=0
total=${#tests[@]}

for i in "${!tests[@]}"; do
    IFS=':' read -r message expected <<< "${tests[$i]}"

    # 调用API
    response=$(curl -s -X POST http://localhost:8000/chat \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"$message\", \"user_id\": \"eval\", \"conv_id\": \"eval_$i\"}")

    # 提取intent
    predicted=$(echo "$response" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('intent', 'error'))")

    # 对比
    if [ "$predicted" == "$expected" ]; then
        echo "  ✅ [$((i+1))/$total] $message"
        echo "      预测: $predicted"
        ((correct++))
    else
        echo "  ❌ [$((i+1))/$total] $message"
        echo "      预测: $predicted | 期望: $expected"
    fi

    sleep 0.2
done

echo ""
echo "============================================================"
echo "📊 评测结果"
echo "============================================================"
echo "  测试用例数: $total"
echo "  正确预测: $correct"
echo "  准确率: $(python3 -c "print(f'{$correct/$total*100:.1f}%')")"
echo ""
echo "💡 这是快速测试版，完整评测请运行: python3 evaluation/run_eval.py"
echo "============================================================"
