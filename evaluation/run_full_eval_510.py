#!/usr/bin/env python3
"""
完整评测脚本 - 使用510条意图测试集
运行完整的意图识别评测并生成详细报告
"""
import json
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("⚠️ 缺少 requests 库，将使用 curl 方式")


def eval_with_api(test_file: str, api_url: str = "http://localhost:8000"):
    """使用API评测"""
    if not HAS_REQUESTS:
        print("❌ 需要安装 requests: pip install requests")
        return None

    print(f"\n{'='*60}")
    print(f"📊 运行完整意图识别评测")
    print(f"{'='*60}\n")

    # 加载测试数据
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    print(f"📋 测试集: {test_file}")
    print(f"📊 测试数量: {len(test_data)} 条")

    # 统计意图分布
    intent_dist = Counter(item['expected_intent'] for item in test_data)
    print(f"🎯 意图类别: {len(intent_dist)} 个")
    print(f"   平均每类: {len(test_data)/len(intent_dist):.1f} 条\n")

    # 开始评测
    results = []
    correct = 0
    total = 0

    # 按意图统计
    per_intent_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'tp': 0, 'fp': 0, 'fn': 0})

    print("🚀 开始评测...\n")
    start_time = time.time()

    for idx, item in enumerate(test_data, 1):
        message = item['message']
        expected = item['expected_intent']
        total += 1

        try:
            # 调用API
            response = requests.post(
                f"{api_url}/chat",
                json={
                    'message': message,
                    'user_id': 'eval_user',
                    'conv_id': f'eval_{idx}'
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                predicted = data.get('intent', 'unknown')
                confidence = data.get('confidence', 0.0)

                # 判断是否正确
                is_correct = (predicted == expected)
                if is_correct:
                    correct += 1

                # 记录结果
                results.append({
                    'index': idx,
                    'message': message,
                    'expected': expected,
                    'predicted': predicted,
                    'correct': is_correct,
                    'confidence': confidence
                })

                # 更新per-intent统计
                per_intent_stats[expected]['total'] += 1
                if is_correct:
                    per_intent_stats[expected]['correct'] += 1
                    per_intent_stats[expected]['tp'] += 1
                else:
                    per_intent_stats[expected]['fn'] += 1
                    per_intent_stats[predicted]['fp'] += 1

                # 进度显示
                if idx % 50 == 0:
                    acc = correct / total
                    print(f"  进度: {idx}/{len(test_data)} ({idx/len(test_data)*100:.1f}%) | 当前准确率: {acc:.1%}")

            else:
                print(f"  ❌ [{idx}] API错误: {response.status_code}")

            # 控制请求频率
            time.sleep(0.1)

        except Exception as e:
            print(f"  ❌ [{idx}] 异常: {str(e)[:50]}")

    elapsed = time.time() - start_time

    # 计算指标
    accuracy = correct / total if total > 0 else 0

    # 计算每类的Precision, Recall, F1
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

    # 生成报告
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

    print(f"{'意图':<25} {'样本':<6} {'准确率':<8} {'Precision':<10} {'Recall':<8} {'F1':<8}")
    print(f"{'-'*80}")

    for intent, metrics in sorted_intents:
        print(f"{intent:<25} {metrics['total']:<6} "
              f"{metrics['accuracy']:<8.1%} "
              f"{metrics['precision']:<10.3f} "
              f"{metrics['recall']:<8.3f} "
              f"{metrics['f1']:<8.3f}")

    # 保存结果
    report = {
        'timestamp': datetime.now().isoformat(),
        'test_file': test_file,
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

    return report


def main():
    """主函数"""
    # 检查Docker服务
    if HAS_REQUESTS:
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code != 200:
                print("❌ API服务未就绪")
                return
        except:
            print("❌ 无法连接到API服务，请先启动: docker compose up -d")
            return

    # 运行评测
    test_file = "data/eval/campus_intent_test_full_500.json"
    if not Path(test_file).exists():
        print(f"❌ 测试文件不存在: {test_file}")
        return

    report = eval_with_api(test_file)

    if report:
        print(f"\n{'='*60}")
        print(f"🎉 评测完成！")
        print(f"{'='*60}")
        print(f"\n核心指标:")
        print(f"  准确率: {report['accuracy']:.2%}")
        print(f"  宏平均F1: {report['macro_f1']:.3f}")
        print(f"\n可用于简历:")
        print(f"  '基于自建510条IT Helpdesk场景评测用例，")
        print(f"   意图识别准确率{report['accuracy']:.1%}，宏平均F1 {report['macro_f1']:.2f}'")


if __name__ == "__main__":
    main()
