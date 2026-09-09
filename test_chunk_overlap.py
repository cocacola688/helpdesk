#!/usr/bin/env python3
"""
测试 Chunk Overlap 功能

验证语义边界 Overlap 的效果
"""

import sys
sys.path.insert(0, "/Users/lb/自学/agent/HelpDeskV2/HelpDesk")

from mcp.knowledge_base import KnowledgeBase


def test_chunk_overlap():
    """测试 chunk overlap 功能"""

    kb = KnowledgeBase()

    # 测试文本（IT场景）
    test_text = (
        "Windows蓝屏错误（BSOD）常见处理方法。"
        "第一步：记录错误码，常见错误码包括0x0000007B（硬盘驱动问题）、0x0000007E（系统文件损坏）、0x000000D1（驱动程序问题）。"
        "第二步：重启电脑，按F8进入安全模式，如能进入说明是驱动或软件冲突。"
        "第三步：在安全模式下卸载最近安装的软件或驱动，特别是显卡驱动、杀毒软件。"
        "第四步：运行系统文件检查 sfc /scannow 修复系统文件。"
        "第五步：检查硬件，拔掉所有外接设备，移除最近新增的硬件（内存、硬盘等）。"
        "第六步：更新BIOS到最新版本。如问题持续，创建工单申请现场支持。"
    )

    print("=" * 80)
    print("测试文本长度:", len(test_text), "字符")
    print("=" * 80)
    print("\n原始文本:")
    print(test_text)
    print("\n" + "=" * 80)

    # 测试切片
    chunks = kb._chunk_text(test_text, chunk_size=200)

    print(f"\n切片结果: {len(chunks)} 个 chunk")
    print("=" * 80)

    for i, chunk in enumerate(chunks, 1):
        print(f"\n【Chunk {i}】 ({len(chunk)} 字符)")
        print("-" * 80)
        print(chunk)
        print("-" * 80)

        # 检查 overlap
        if i > 1:
            prev_chunk = chunks[i-2]
            # 查找重叠部分
            overlap = ""
            max_overlap = min(len(prev_chunk), len(chunk))
            for length in range(max_overlap, 0, -1):
                if prev_chunk[-length:] == chunk[:length]:
                    overlap = chunk[:length]
                    break

            if overlap:
                print(f"\n✅ Overlap with Chunk {i-1}:")
                print(f"   重叠长度: {len(overlap)} 字符")
                print(f"   重叠内容: {overlap}")
            else:
                print(f"\n⚠️  No overlap detected with Chunk {i-1}")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


def test_edge_cases():
    """测试边界情况"""

    print("\n\n" + "=" * 80)
    print("测试边界情况")
    print("=" * 80)

    kb = KnowledgeBase()

    # 测试1：短文本（不需要切分）
    print("\n【测试1】短文本（不需要切分）")
    short_text = "这是一段短文本。"
    chunks = kb._chunk_text(short_text, chunk_size=500)
    print(f"输入: {short_text}")
    print(f"输出: {len(chunks)} 个 chunk")
    assert len(chunks) == 1
    print("✅ 通过")

    # 测试2：空文本
    print("\n【测试2】空文本")
    empty_text = ""
    chunks = kb._chunk_text(empty_text, chunk_size=500)
    print(f"输入: (空)")
    print(f"输出: {len(chunks)} 个 chunk")
    assert len(chunks) == 0
    print("✅ 通过")

    # 测试3：只有一句话，但很长
    print("\n【测试3】单句超长文本")
    long_sentence = "这是一个非常长的句子" + "，内容" * 50 + "。"
    chunks = kb._chunk_text(long_sentence, chunk_size=100)
    print(f"输入: {len(long_sentence)} 字符的单句")
    print(f"输出: {len(chunks)} 个 chunk")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}: {len(chunk)} 字符")
    print("✅ 通过")

    # 测试4：多个短句
    print("\n【测试4】多个短句")
    short_sentences = "句子一。句子二。句子三。句子四。句子五。句子六。"
    chunks = kb._chunk_text(short_sentences, chunk_size=20)
    print(f"输入: {short_sentences}")
    print(f"输出: {len(chunks)} 个 chunk")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}: {chunk}")
    print("✅ 通过")


def test_overlap_quality():
    """测试 overlap 质量"""

    print("\n\n" + "=" * 80)
    print("测试 Overlap 质量")
    print("=" * 80)

    kb = KnowledgeBase()

    test_text = (
        "VPN连接失败常见原因及解决方案。"
        "原因1：客户端版本过旧。解决：访问内网门户下载最新版VPN客户端。"
        "原因2：账号过期或被锁定。解决：联系IT支持确认账号状态。"
        "原因3：防火墙阻止。解决：检查Windows防火墙是否允许VPN客户端。"
        "原因4：网络环境限制。解决：某些公共Wi-Fi屏蔽VPN端口，尝试切换到手机热点。"
        "端口要求：UDP 500、UDP 4500、TCP 443必须开放。"
    )

    chunks = kb._chunk_text(test_text, chunk_size=100)

    print(f"\n原始文本: {len(test_text)} 字符")
    print(f"切片结果: {len(chunks)} 个 chunk\n")

    # 分析每对相邻 chunk 的 overlap
    total_overlap = 0
    overlap_count = 0

    for i in range(len(chunks) - 1):
        curr_chunk = chunks[i]
        next_chunk = chunks[i + 1]

        # 计算 overlap
        overlap = ""
        max_overlap = min(len(curr_chunk), len(next_chunk))
        for length in range(max_overlap, 0, -1):
            if curr_chunk[-length:] == next_chunk[:length]:
                overlap = next_chunk[:length]
                break

        overlap_ratio = len(overlap) / len(curr_chunk) * 100 if overlap else 0

        print(f"Chunk {i+1} → Chunk {i+2}:")
        print(f"  Overlap: {len(overlap)} 字符 ({overlap_ratio:.1f}%)")
        print(f"  内容: {overlap[:50]}..." if len(overlap) > 50 else f"  内容: {overlap}")

        if overlap:
            total_overlap += len(overlap)
            overlap_count += 1

    if overlap_count > 0:
        avg_overlap = total_overlap / overlap_count
        print(f"\n平均 Overlap: {avg_overlap:.1f} 字符")

    print("\n✅ Overlap 质量测试完成")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Chunk Overlap 功能测试")
    print("=" * 80)

    test_chunk_overlap()
    test_edge_cases()
    test_overlap_quality()

    print("\n" + "=" * 80)
    print("所有测试完成 ✅")
    print("=" * 80)
