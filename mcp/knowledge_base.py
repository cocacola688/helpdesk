"""
RAG 知识库 —— 基于 ChromaDB 的真实检索实现。

功能：
  1. 文档导入：将文本切片后存入 ChromaDB（自动生成 Embedding）
  2. 语义检索：根据 query 从知识库中检索最相关的文档片段
  3. 与 MCP 工具框架集成：作为 knowledge_search 工具的真实 handler

ChromaDB 在这里的角色：
  - memory/ 中用于存储对话记忆（情景记忆 + 用户画像）
  - 这里用于存储知识库文档（RAG 检索）
  两者是不同的 collection，互不干扰。
"""
import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库。

    ChromaDB 内置了 Embedding 模型（all-MiniLM-L6-v2），
    调用 add() 时自动生成向量，query() 时自动做语义匹配。
    不需要额外调用 Anthropic Embeddings API。
    """

    COLLECTION_NAME = "knowledge_base"

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
    ):
        # 优先连接独立 ChromaDB 服务（服务端内置 embedding 模型，客户端无需下载）
        self._use_server = False
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 使用服务端时不传 embedding_function，让服务端处理
        # 本地模式时也不传，使用 ChromaDB 默认的（会触发模型下载）
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "HelpDesk RAG 知识库"},
        )

        # 如果知识库为空，导入默认文档
        if self._collection.count() == 0:
            self._load_default_docs()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [
            {
                "title": "...",
                "content": "...",
                "category": "device",           # 可选：分类（device/permission/network/general/support）
                "domain": "device_failure",     # 可选：具体领域
                "tags": ["蓝屏", "故障"],         # 可选：标签列表
            },
            ...
        ]
        长文档会自动切片（每片 500 字）。
        """
        ids, docs, metas = [], [], []

        for doc in documents:
            title    = doc.get("title", "")
            content  = doc.get("content", "")
            category = doc.get("category", "general")  # 默认 general
            domain   = doc.get("domain", "")
            tags     = doc.get("tags", [])
            chunks   = self._chunk_text(content, chunk_size=500)

            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{title}_{i}_{chunk[:50]}".encode()).hexdigest()
                ids.append(doc_id)
                docs.append(chunk)

                # 增强的 metadata
                meta = {
                    "title": title,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "category": category,  # 分类
                }

                # 可选字段
                if domain:
                    meta["domain"] = domain
                if tags:
                    meta["tags"] = ",".join(tags) if isinstance(tags, list) else tags

                metas.append(meta)

        if ids:
            # ChromaDB 会自动生成 Embedding
            self._collection.add(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"知识库导入 {len(ids)} 个文档片段")

        return len(ids)

    async def add_documents_async(self, documents: List[Dict[str, str]]) -> int:
        """异步导入文档；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.add_documents, documents)

    def search(self, query: str, top_k: int = 5, category: str = None, domain: str = None) -> List[Dict[str, Any]]:
        """
        语义检索：根据 query 返回最相关的文档片段。

        ChromaDB 内部自动将 query 转为向量，与存储的文档向量做余弦相似度匹配。

        参数:
            query: 查询文本
            top_k: 返回结果数量
            category: 可选，按分类过滤（device/permission/network/general/support）
            domain: 可选，按具体领域过滤（course_selection/library/dormitory等）

        返回:
            List[Dict]: 检索结果列表
        """
        # 构建过滤条件
        where = {}
        if category:
            where["category"] = category
        if domain:
            where["domain"] = domain

        # 调用 ChromaDB 检索（支持 metadata 过滤）
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where if where else None,  # 只有有过滤条件时才传入
        )

        items = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                items.append({
                    "title":    meta.get("title", ""),
                    "content":  doc,
                    "score":    round(1.0 - dist, 4),  # ChromaDB 返回距离，转为相似度
                    "chunk":    meta.get("chunk_index", 0),
                })

        return items

    async def search_async(self, query: str, top_k: int = 5, category: str = None, domain: str = None) -> List[Dict[str, Any]]:
        """异步检索；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.search, query, top_k, category, domain)

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    async def doc_count_async(self) -> int:
        """异步获取文档片段数量。"""
        return await asyncio.to_thread(self._collection.count)

    # ── MCP 工具 handler ─────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """
        作为 MCP 工具的 handler 注册。

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))

        支持参数：
        - query: 查询文本
        - top_k: 返回数量（默认 5）
        - category: 分类过滤（可选）
        - domain: 领域过滤（可选）
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        category = params.get("category")
        domain = params.get("domain")
        return await self.search_async(query, top_k=top_k, category=category, domain=domain)

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        将长文本按句子切片，带语义边界的重叠。

        优化策略：
        - 按句子切分，保证语义完整性
        - 相邻 chunk 重叠 2 个句子，保留上下文连续性
        - 解决边界信息丢失和跨 chunk 推理问题

        参数:
            text: 原始文本
            chunk_size: 每个 chunk 的目标大小（字符数）

        返回:
            List[str]: 切片后的文本列表

        示例:
            输入: "句子A。句子B。句子C。句子D。句子E。"
            输出: [
                "句子A。句子B。句子C。",
                "句子B。句子C。句子D。句子E。"  # 重叠了句子B和C
            ]
        """
        # 快速路径：文本短于 chunk_size，无需切分
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        overlap_sentences = 2  # 重叠句子数

        # 按句号切分成句子列表
        sentences = text.replace("\n", "。").split("。")
        sentences = [s.strip() for s in sentences if s.strip()]

        # 为每个句子添加句号（除了最后一个可能没有）
        sentences = [s if s.endswith("。") else s + "。" for s in sentences]

        i = 0  # 当前处理的句子索引
        while i < len(sentences):
            chunk_sentences = []
            current_length = 0

            # 添加 overlap（从前一个 chunk 的最后 overlap_sentences 句开始）
            if chunks and i >= overlap_sentences:
                overlap_start = i - overlap_sentences
                for j in range(overlap_start, i):
                    if j < len(sentences):
                        sent = sentences[j]
                        chunk_sentences.append(sent)
                        current_length += len(sent)

            # 继续添加新句子，直到达到 chunk_size
            start_i = i  # 记录本次开始位置
            while i < len(sentences):
                sent = sentences[i]

                # 检查是否会超过 chunk_size
                if current_length + len(sent) <= chunk_size:
                    chunk_sentences.append(sent)
                    current_length += len(sent)
                    i += 1
                else:
                    # 已达到 chunk_size，结束当前 chunk
                    break

            # 如果本次没有添加任何新句子（全是 overlap），强制至少添加一个句子
            if i == start_i and i < len(sentences):
                chunk_sentences.append(sentences[i])
                i += 1

            # 保存当前 chunk
            if chunk_sentences:
                chunk_text = "".join(chunk_sentences)
                chunks.append(chunk_text)

        return chunks

    def _load_default_docs(self) -> None:
        """导入默认知识库文档（IT支持场景常见问题）。"""
        default_docs = [
            {
                "title": "Windows蓝屏错误处理",
                "category": "device",
                "domain": "device_failure",
                "tags": ["蓝屏", "Windows", "故障排查"],
                "content": (
                    "Windows蓝屏错误（BSOD）常见处理方法。"
                    "第一步：记录错误码，常见错误码包括0x0000007B（硬盘驱动问题）、0x0000007E（系统文件损坏）、0x000000D1（驱动程序问题）。"
                    "第二步：重启电脑，按F8进入安全模式，如能进入说明是驱动或软件冲突。"
                    "第三步：在安全模式下卸载最近安装的软件或驱动，特别是显卡驱动、杀毒软件。"
                    "第四步：运行系统文件检查 sfc /scannow 修复系统文件。"
                    "第五步：检查硬件，拔掉所有外接设备，移除最近新增的硬件（内存、硬盘等）。"
                    "第六步：更新BIOS到最新版本。如问题持续，创建工单申请现场支持。"
                ),
            },
            {
                "title": "VPN连接故障排查",
                "category": "network",
                "domain": "vpn_access",
                "tags": ["VPN", "连接", "远程访问"],
                "content": (
                    "VPN连接失败常见原因及解决方案。"
                    "原因1：客户端版本过旧。解决：访问内网门户下载最新版VPN客户端（当前版本v3.2.1）。"
                    "原因2：账号过期或被锁定。解决：联系IT支持确认账号状态，VPN账号有效期为6个月。"
                    "原因3：防火墙阻止。解决：检查Windows防火墙是否允许VPN客户端，添加例外规则。"
                    "原因4：网络环境限制。解决：某些公共Wi-Fi屏蔽VPN端口，尝试切换到手机热点。"
                    "原因5：证书过期。解决：删除旧证书，重新登录自动下载新证书。"
                    "端口要求：UDP 500、UDP 4500、TCP 443必须开放。"
                    "如以上方法无效，记录错误日志（客户端-设置-导出日志）并创建工单。"
                ),
            },
            {
                "title": "打印机常见故障处理",
                "category": "device",
                "domain": "device_failure",
                "tags": ["打印机", "卡纸", "驱动"],
                "content": (
                    "打印机故障快速处理指南。"
                    "故障1：卡纸。处理：关闭打印机电源，打开后盖取出卡纸，注意沿出纸方向取出避免撕裂，清理纸屑，检查纸张规格是否正确（A4 70-80g）。"
                    "故障2：打印模糊或有条纹。处理：打开打印机属性-维护-清洗打印头，如无改善更换墨盒或碳粉盒。"
                    "故障3：无法识别打印机。处理：检查USB线或网线连接，重启打印机和电脑，重新安装驱动程序（从公司内网下载对应型号驱动）。"
                    "故障4：打印队列卡住。处理：控制面板-设备和打印机-右键打印机-查看打印队列-取消所有文档，重启Print Spooler服务。"
                    "故障5：提示墨粉不足但刚换过。处理：取出墨盒左右摇晃使墨粉分布均匀，检查芯片触点是否清洁。"
                    "公司打印机型号：HP LaserJet Pro M428、Canon imageCLASS MF445。驱动下载地址：内网-IT资源-打印机驱动。"
                ),
            },
            {
                "title": "文件共享权限申请流程",
                "category": "permission",
                "domain": "permission_request",
                "tags": ["权限", "文件共享", "申请"],
                "content": (
                    "文件共享权限申请流程说明。"
                    "申请条件：正式员工，有明确业务需求。"
                    "所需信息：工号、部门、申请的文件夹路径（如\\\\fileserver\\projects\\ProjectA）、权限类型（只读/读写/完全控制）、业务理由、预计使用时长。"
                    "申请流程：第一步，登录IT服务门户（https://itsupport.company.com）-权限申请-文件共享权限。"
                    "第二步，填写申请表单，上传部门主管邮件审批（必需）。"
                    "第三步，提交后系统自动转发给文件夹管理员审批。"
                    "第四步，管理员审批通过后，IT自动配置权限，邮件通知申请人。"
                    "处理时效：普通权限4小时内，敏感数据文件夹需额外安全审批，1-2个工作日。"
                    "注意事项：权限遵循最小化原则，只申请必需的权限；临时项目权限到期自动回收；离职时所有权限自动清除。"
                    "联系人：文件服务器问题联系存储团队 storage@company.com。"
                ),
            },
            {
                "title": "Office软件安装指南",
                "category": "device",
                "domain": "software_install",
                "tags": ["Office", "安装", "激活"],
                "content": (
                    "公司Office软件安装与激活指南。"
                    "授权版本：Microsoft Office 2021 Professional Plus（企业批量授权版）。"
                    "安装步骤：第一步，访问内网软件中心（http://software.internal.company.com）。"
                    "第二步，下载Office 2021安装包（约3GB，建议使用有线网络）。"
                    "第三步，卸载旧版本Office（重要！保留Outlook邮箱数据）。"
                    "第四步，以管理员身份运行安装程序，选择完整安装（包含Word、Excel、PowerPoint、Outlook、OneNote、Teams）。"
                    "第五步，安装完成后打开任意Office应用，输入公司邮箱激活（使用工作账号登录Microsoft 365）。"
                    "激活方式：自动激活（连接公司内网时），首次激活需联网。"
                    "常见问题：提示激活失败-检查是否连接VPN或公司内网；提示版本冲突-确认已完全卸载旧版本；提示权限不足-右键以管理员身份运行。"
                    "技术支持：Office激活问题联系 office-support@company.com，附上错误截图和工号。"
                ),
            },
            {
                "title": "Wi-Fi连接问题解决",
                "category": "network",
                "domain": "wifi_problem",
                "tags": ["Wi-Fi", "无线网络", "连接"],
                "content": (
                    "公司Wi-Fi连接问题排查与解决。"
                    "公司Wi-Fi网络：CompanyWiFi（2.4GHz+5GHz双频，覆盖所有办公区）、CompanyGuest（访客网络，需验证码）。"
                    "连接步骤：搜索CompanyWiFi-输入域账号（工号@company.com）和密码-接受证书-连接成功。"
                    "问题1：无法搜索到Wi-Fi。排查：检查笔记本Wi-Fi开关（Fn+F2），更新无线网卡驱动，检查是否在信号覆盖范围。"
                    "问题2：连接后无法上网。排查：打开浏览器测试，如自动跳转到认证页面输入工号密码；检查IP地址是否自动获取（192.168.x.x）；ping 网关测试（192.168.1.1）。"
                    "问题3：频繁断线。排查：检查信号强度（建议>-70dBm），切换到5GHz频段（信号更稳定），远离微波炉等干扰源，更新Wi-Fi驱动。"
                    "问题4：速度慢。排查：测速（http://speedtest.company.com），5GHz理论速度300Mbps，2.4GHz理论速度100Mbps；高峰期（9:00-10:00）可能拥堵；清除DNS缓存（ipconfig /flushdns）。"
                    "Wi-Fi覆盖问题或AP故障请创建工单，标注具体位置（楼层+工位号）。"
                ),
            },
            {
                "title": "IT支持联系方式",
                "category": "support",
                "domain": "contact_info",
                "tags": ["联系", "支持", "服务台"],
                "content": (
                    "IT支持服务联系方式。"
                    "服务台热线：400-XXX-XXXX（7×24小时，按1设备支持，按2网络支持，按3权限申请，按0紧急问题）。"
                    "邮箱：itsupport@company.com（非紧急问题建议使用邮件，附上问题截图和设备信息）。"
                    "企业微信：IT服务台（工作日9:00-18:00响应）。"
                    "服务门户：https://itsupport.company.com（创建工单、查询进度、下载软件、查看知识库）。"
                    "现场支持：总部-3楼IT办公室（周一至周五 9:00-12:00, 14:00-18:00）；分公司-联系当地IT对接人。"
                    "响应时效：紧急问题（业务中断）30分钟内响应；高优先级（影响工作）2小时内响应；普通问题4小时内响应；咨询类1个工作日内回复。"
                    "紧急联系人：IT经理 138-XXXX-XXXX（非工作时间紧急情况）；网络工程师 139-XXXX-XXXX（网络中断）；安全事件 security@company.com（病毒、数据泄露）。"
                    "常见问题可先查阅知识库（服务门户-知识中心）或FAQ，90%的问题都有标准解决方案。"
                ),
            },
        ]
        self.add_documents(default_docs)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
