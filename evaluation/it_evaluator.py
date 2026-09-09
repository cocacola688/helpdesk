#!/usr/bin/env python3
"""
IT Helpdesk 完整评测脚本

评测内容：
1. 意图识别准确率（200+条测试用例）
2. RAG回答质量（50+条问答对）
3. 端到端对话质量（10组场景）
4. 生成详细报告和简历指标
"""
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.intent_recognizer import IntentRecognizer
from agents.agent_orchestrator import AgentOrchestrator
from evaluation.evaluator import LLMJudge


class ITHelpdeskEvaluator:
    """IT Helpdesk场景完整评测器"""

    def __init__(self, recognizer: IntentRecognizer, orchestrator: AgentOrchestrator, judge: LLMJudge):
        self.recognizer = recognizer
        self.orchestrator = orchestrator
        self.judge = judge

    async def eval_intent_recognition(self, test_file: Path) -> Dict[str, Any]:
        """评测意图识别准确率"""
        print("\n" + "="*60)
        print("📊 意图识别评测")
        print("="*60)

        with open(test_file, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)

        correct = 0
        total = len(test_cases)
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)
        errors = []

        for i, case in enumerate(test_cases, 1):
            try:
                result = await self.recognizer.recognize(case["message"])
                predicted = result["intent"]
                expected = case["expected_intent"]

                # 记录混淆矩阵
                confusion_matrix[expected][predicted] += 1
                per_class_total[expected] += 1

                if predicted == expected:
                    correct += 1
                    per_class_correct[expected] += 1
                    status = "✅"
                else:
                    status = "❌"
                    errors.append({
                        "message": case["message"],
                        "expected": expected,
                        "predicted": predicted,
                        "confidence": result.get("confidence", 0)
                    })

                print(f"  [{i}/{total}] {status} {case['message'][:30]:30s} | 预测: {predicted:15s} | 期望: {expected:15s}")

            except Exception as e:
                print(f"  [{i}/{total}] ⚠️  错误: {e}")
                errors.append({
                    "message": case["message"],
                    "error": str(e)
                })

        # 计算指标
        accuracy = correct / total if total > 0 else 0

        # 计算每类的Precision, Recall, F1
        per_class_metrics = {}
        all_f1_scores = []

        for intent in per_class_total.keys():
            # Recall = TP / (TP + FN)
            recall = per_class_correct[intent] / per_class_total[intent] if per_class_total[intent] > 0 else 0

            # Precision = TP / (TP + FP)
            tp = per_class_correct[intent]
            fp = sum(confusion_matrix[other][intent] for other in per_class_total.keys() if other != intent)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0

            # F1 = 2 * P * R / (P + R)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            per_class_metrics[intent] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": per_class_total[intent]
            }
            all_f1_scores.append(f1)

        # 宏平均F1
        macro_f1 = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0

        print(f"\n📈 意图识别结果：")
        print(f"  准确率: {accuracy*100:.1f}% ({correct}/{total})")
        print(f"  宏平均F1: {macro_f1:.3f}")
        print(f"  错误数: {len(errors)}")

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "correct": correct,
            "total": total,
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": dict(confusion_matrix),
            "errors": errors
        }

    async def eval_rag_quality(self, test_file: Path) -> Dict[str, Any]:
        """评测RAG回答质量"""
        print("\n" + "="*60)
        print("📚 RAG回答质量评测")
        print("="*60)

        with open(test_file, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)

        correct = 0
        total = len(test_cases)
        scores = []
        errors = []

        for i, case in enumerate(test_cases, 1):
            try:
                # 调用完整对话流程
                response = await self.orchestrator.run({
                    "message": case["question"],
                    "user_id": "eval_user",
                    "conv_id": f"eval_rag_{i}"
                })

                # 使用LLM-as-Judge评分
                judge_result = await self.judge.judge(
                    question=case["question"],
                    response=response.response,
                    expected_answer=case["expected_answer"]
                )

                score = judge_result.get("overall_score", 0)
                scores.append(score)

                if score >= 0.75:
                    correct += 1
                    status = "✅"
                else:
                    status = "❌"
                    errors.append({
                        "question": case["question"],
                        "expected": case["expected_answer"],
                        "actual": response.response[:200],
                        "score": score
                    })

                print(f"  [{i}/{total}] {status} {case['question'][:40]:40s} | 得分: {score:.2f}")

            except Exception as e:
                print(f"  [{i}/{total}] ⚠️  错误: {e}")
                errors.append({
                    "question": case["question"],
                    "error": str(e)
                })

        avg_score = sum(scores) / len(scores) if scores else 0
        accuracy = correct / total if total > 0 else 0

        print(f"\n📈 RAG质量结果：")
        print(f"  回答准确率: {accuracy*100:.1f}% ({correct}/{total}，阈值0.75)")
        print(f"  平均得分: {avg_score:.3f}")

        return {
            "accuracy": accuracy,
            "avg_score": avg_score,
            "correct": correct,
            "total": total,
            "scores": scores,
            "errors": errors
        }

    async def eval_e2e_scenarios(self, test_file: Path) -> Dict[str, Any]:
        """评测端到端对话场景"""
        print("\n" + "="*60)
        print("💬 端到端对话评测")
        print("="*60)

        with open(test_file, 'r', encoding='utf-8') as f:
            scenarios = json.load(f)

        scenario_results = []
        all_scores = []

        for i, scenario in enumerate(scenarios, 1):
            print(f"\n  场景 {i}: {scenario['scenario_name']}")
            print(f"  描述: {scenario['description']}")

            turn_results = []
            scenario_scores = []

            for j, turn in enumerate(scenario['turns'], 1):
                try:
                    response = await self.orchestrator.run({
                        "message": turn["user"],
                        "user_id": "eval_user",
                        "conv_id": f"eval_e2e_{i}"
                    })

                    # 检查预期条件
                    checks = []
                    if "expected_intent" in turn:
                        intent_match = response.intent == turn["expected_intent"]
                        checks.append(("intent", intent_match))

                    if "expected_agent" in turn:
                        agent_match = response.agent_type == turn["expected_agent"]
                        checks.append(("agent", agent_match))

                    if "check_contains" in turn:
                        contains_all = all(kw in response.response for kw in turn["check_contains"])
                        checks.append(("contains", contains_all))

                    if "expected_knowledge_used" in turn:
                        kb_match = response.knowledge_used == turn["expected_knowledge_used"]
                        checks.append(("knowledge", kb_match))

                    if "expected_escalated" in turn:
                        escalate_match = response.escalated == turn["expected_escalated"]
                        checks.append(("escalated", escalate_match))

                    all_passed = all(check[1] for check in checks)
                    status = "✅" if all_passed else "⚠️"

                    print(f"    轮次 {j}: {status} {turn['user'][:40]}")
                    for check_name, passed in checks:
                        print(f"      - {check_name}: {'✓' if passed else '✗'}")

                    turn_results.append({
                        "user": turn["user"],
                        "response": response.response[:200],
                        "checks": {name: passed for name, passed in checks},
                        "all_passed": all_passed
                    })

                except Exception as e:
                    print(f"    轮次 {j}: ⚠️  错误: {e}")
                    turn_results.append({
                        "user": turn["user"],
                        "error": str(e)
                    })

            # 场景整体评分
            passed_turns = sum(1 for t in turn_results if t.get("all_passed", False))
            scenario_score = passed_turns / len(turn_results) if turn_results else 0
            scenario_scores.append(scenario_score)
            all_scores.append(scenario_score)

            scenario_results.append({
                "name": scenario["scenario_name"],
                "score": scenario_score,
                "turns": turn_results
            })

            print(f"  场景得分: {scenario_score:.2%}")

        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

        print(f"\n📈 端到端评测结果：")
        print(f"  平均场景得分: {avg_score:.2%}")

        return {
            "avg_score": avg_score,
            "scenario_results": scenario_results,
            "total_scenarios": len(scenarios)
        }

    def generate_report(self, results: Dict[str, Any], output_file: Path):
        """生成评测报告"""
        print("\n" + "="*60)
        print("📝 生成评测报告")
        print("="*60)

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "intent_accuracy": results["intent"]["accuracy"],
                "intent_macro_f1": results["intent"]["macro_f1"],
                "rag_accuracy": results["rag"]["accuracy"],
                "rag_avg_score": results["rag"]["avg_score"],
                "e2e_avg_score": results["e2e"]["avg_score"]
            },
            "detailed_results": results
        }

        # 保存JSON报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"  ✅ 详细报告已保存: {output_file}")

        # 生成Markdown报告
        md_file = output_file.with_suffix('.md')
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"# IT Helpdesk 评测报告\n\n")
            f.write(f"**评测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(f"## 📊 核心指标摘要\n\n")
            f.write(f"| 指标 | 数值 |\n")
            f.write(f"|------|------|\n")
            f.write(f"| 意图识别准确率 | {results['intent']['accuracy']*100:.1f}% |\n")
            f.write(f"| 意图识别宏平均F1 | {results['intent']['macro_f1']:.3f} |\n")
            f.write(f"| RAG回答准确率 | {results['rag']['accuracy']*100:.1f}% |\n")
            f.write(f"| RAG平均得分 | {results['rag']['avg_score']:.3f} |\n")
            f.write(f"| 端到端场景得分 | {results['e2e']['avg_score']:.2%} |\n\n")

            f.write(f"## 📝 简历指标\n\n")
            f.write(f"```\n")
            f.write(f"意图识别准确率 {results['intent']['accuracy']*100:.1f}%、宏平均 F1 {results['intent']['macro_f1']:.2f}\n")
            f.write(f"RAG回答准确率 {results['rag']['accuracy']*100:.1f}%、平均得分 {results['rag']['avg_score']:.2f}\n")
            f.write(f"```\n\n")

            f.write(f"## 详细数据\n\n")
            f.write(f"### 意图识别\n\n")
            f.write(f"- 测试用例数: {results['intent']['total']}\n")
            f.write(f"- 正确预测: {results['intent']['correct']}\n")
            f.write(f"- 错误数: {len(results['intent']['errors'])}\n\n")

            f.write(f"### RAG质量\n\n")
            f.write(f"- 测试问题数: {results['rag']['total']}\n")
            f.write(f"- 达标数: {results['rag']['correct']}\n\n")

            f.write(f"### 端到端对话\n\n")
            f.write(f"- 测试场景数: {results['e2e']['total_scenarios']}\n")

        print(f"  ✅ Markdown报告已保存: {md_file}")

        return report


async def main():
    """运行完整评测"""
    print("="*60)
    print("🖥️  IT Helpdesk 完整评测")
    print("="*60)

    # 初始化组件
    print("\n⚙️  初始化评测组件...")

    # TODO: 需要实际的组件初始化
    # recognizer = IntentRecognizer(...)
    # orchestrator = AgentOrchestrator(...)
    # judge = LLMJudge(...)

    # evaluator = CampusEvaluator(recognizer, orchestrator, judge)

    # 评测路径
    data_dir = Path(__file__).parent.parent / "data" / "eval"

    # 运行评测
    # results = {}
    # results["intent"] = await evaluator.eval_intent_recognition(data_dir / "campus_intent_test.json")
    # results["rag"] = await evaluator.eval_rag_quality(data_dir / "campus_rag_test.json")
    # results["e2e"] = await evaluator.eval_e2e_scenarios(data_dir / "campus_e2e_test.json")

    # 生成报告
    # output_file = data_dir / f"campus_eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    # evaluator.generate_report(results, output_file)

    print("\n✅ 评测完成！")
    print("\n💡 提示: 此脚本需要在应用运行时执行，或者导入实际的组件")


if __name__ == "__main__":
    asyncio.run(main())
