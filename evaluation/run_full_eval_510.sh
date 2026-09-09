#!/bin/bash
# 使用curl进行完整510条评测

echo "============================================================"
echo "📊 完整意图识别评测 (510条)"
echo "============================================================"
echo ""

# 检查服务
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ API服务未运行，请先启动: docker compose up -d"
    exit 1
fi

echo "✅ API服务正常"
echo ""

# 加载测试数据
TEST_FILE="data/eval/campus_intent_test_full_500.json"

if [ ! -f "$TEST_FILE" ]; then
    echo "❌ 测试文件不存在: $TEST_FILE"
    exit 1
fi

echo "📋 测试文件: $TEST_FILE"
echo ""

# 使用Python读取JSON并进行评测
python3 << 'PYEOF'
import json
import subprocess
import time
from collections import defaultdict, Counter
from datetime import datetime

print("🚀 开始评测...\n")

# 加载测试数据
with open('data/eval/campus_intent_test_full_500.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

total = len(test_data)
print(f"📊 测试总数: {total} 条")

# 统计
intent_dist = Counter(item['expected_intent'] for item in test_data)
print(f"🎯 意图类别: {len(intent_dist)} 个")
print(f"   平均每类: {total/len(intent_dist):.1f} 条\n")

# 评测
correct = 0
results = []
per_intent_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'tp': 0, 'fp': 0, 'fn': 0})

start_time = time.time()

for idx, item in enumerate(test_data, 1):
    message = item['message']
    expected = item['expected_intent']

    # 调用API
    try:
        cmd = [
            'curl', '-s', '-X', 'POST',
            'http://localhost:8000/chat',
            '-H', 'Content-Type: application/json',
            '-d', json.dumps({
                'message': message,
                'user_id': 'eval_user',
                'conv_id': f'eval_{idx}'
            })
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            predicted = data.get('intent', 'unknown')
            confidence = data.get('confidence', 0.0)

            is_correct = (predicted == expected)
            if is_correct:
                correct += 1

            results.append({
                'index': idx,
                'message': message,
                'expected': expected,
                'predicted': predicted,
                'correct': is_correct,
                'confidence': confidence
            })

            # 统计
            per_intent_stats[expected]['total'] += 1
            if is_correct:
                per_intent_stats[expected]['correct'] += 1
                per_intent_stats[expected]['tp'] += 1
            else:
                per_intent_stats[expected]['fn'] += 1
                per_intent_stats[predicted]['fp'] += 1

            # 进度
            if idx % 50 == 0:
                acc = correct / idx
                print(f"  进度: {idx}/{total} ({idx/total*100:.1f}%) | 当前准确率: {acc:.1%}")

        time.sleep(0.1)  # 控制请求频率

    except Exception as e:
        print(f"  ❌ [{idx}] 错误: {str(e)[:50]}")

elapsed = time.time() - start_time

# 计算指标
accuracy = correct / total if total > 0 else 0

# 计算每类指标
per_intent_metrics = {}
for intent in intent_dist.keys():
    stats = per_intent_stats[intent]
    tp = stats['tp']
    fp = stats['fp']
    fn = stats['fn']

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    per_intent_metrics[intent] = {
        'total': stats['total'],
        'correct': stats['correct'],
        'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 宏平均F1
macro_f1 = sum(m['f1'] for m in per_intent_metrics.values()) / len(per_intent_metrics) if per_intent_metrics else 0

# 输出结果
print(f"\n{'='*60}")
print(f"📊 评测完成")
print(f"{'='*60}\n")

print(f"⏱️  耗时: {elapsed:.1f}秒 ({elapsed/total:.2f}秒/条)")
print(f"📊 总数: {total} 条")
print(f"✅ 正确: {correct} 条")
print(f"❌ 错误: {total - correct} 条")
print(f"🎯 准确率: {accuracy:.2%}")
print(f"📈 宏平均F1: {macro_f1:.3f}\n")

print(f"{'='*60}")
print(f"📋 各意图详细指标")
print(f"{'='*60}\n")

# 排序显示
sorted_intents = sorted(per_intent_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)

print(f"{'意图':<25} {'样本':<6} {'准确率':<8} {'P':<8} {'R':<8} {'F1':<8}")
print(f"{'-'*70}")

for intent, metrics in sorted_intents:
    print(f"{intent:<25} {metrics['total']:<6} "
          f"{metrics['accuracy']:<8.1%} "
          f"{metrics['precision']:<8.3f} "
          f"{metrics['recall']:<8.3f} "
          f"{metrics['f1']:<8.3f}")

# 保存报告
report = {
    'timestamp': datetime.now().isoformat(),
    'test_file': 'data/eval/campus_intent_test_full_500.json',
    'total': total,
    'correct': correct,
    'accuracy': accuracy,
    'macro_f1': macro_f1,
    'elapsed': elapsed,
    'per_intent_metrics': per_intent_metrics,
    'results': results
}

report_file = f"data/eval/eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(report_file, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n✅ 详细报告已保存: {report_file}")

print(f"\n{'='*60}")
print(f"🎉 可用于简历的指标")
print(f"{'='*60}")
print(f"\n基于自建510条IT Helpdesk场景评测用例，")
print(f"意图识别准确率达到 {accuracy:.1%}，宏平均F1为 {macro_f1:.2f}，")
print(f"单类F1分布在 {min(m['f1'] for m in per_intent_metrics.values()):.2f}-{max(m['f1'] for m in per_intent_metrics.values()):.2f} 之间。")
print("")

PYEOF

echo ""
echo "============================================================"
echo "✅ 评测完成"
echo "============================================================"
