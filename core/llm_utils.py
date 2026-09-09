"""
LLM 响应辅助工具
用于处理 Anthropic API 返回的复杂响应格式
"""
from typing import Any, Iterable, List


def extract_text_content(content: Iterable[Any]) -> str:
    """
    从 Anthropic 风格的响应内容中提取纯文本块

    Anthropic API 返回的 content 可能包含多种类型的块：
    - text 块：纯文本内容
    - tool_use 块：工具调用请求
    - 其他类型块

    本函数只提取 text 类型的块，并将它们合并为完整文本

    参数:
        content: API 响应的 content 字段，可能是对象列表或字符串

    返回:
        合并后的纯文本内容，多个文本块用换行符连接
    """
    texts: List[str] = []

    # 遍历所有内容块
    for block in content or []:
        # 处理已经是字符串的情况（简单响应）
        if isinstance(block, str):
            texts.append(block)
            continue

        # 从对象中提取 type 和 text 属性
        # 兼容两种格式：对象属性和字典键
        block_type = getattr(block, "type", None)
        text = getattr(block, "text", None)
        if isinstance(block, dict):
            block_type = block.get("type", block_type)
            text = block.get("text", text)

        # 只收集 text 类型的块
        if isinstance(text, str) and (block_type in (None, "text")):
            texts.append(text)

    # 将所有文本块用换行符连接
    return "\n".join(t for t in texts if t)
