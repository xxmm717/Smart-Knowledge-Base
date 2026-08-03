import json
import os
import sys
from typing import Tuple, Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from processor.common.logger import logger
from processor.import_process.core.state import ImportGraphState
from processor.utils.core.task_utils import add_running_task
from processor.utils.format_utils import format_state
import re

# --- 配置参数 (Configuration) ---
# 单个Chunk最大字符长度：超过则触发二次切分（适配大模型上下文窗口）
DEFAULT_MAX_CONTENT_LENGTH = 2000
# 短Chunk合并阈值：同父标题的短Chunk会被合并，减少碎片化
MIN_CONTENT_LENGTH = 500

def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分
    将长文档切分成小的 Chunks (切片) 以便检索。

    【核心节点】文档切分主节点（node_document_split）
    整体流程：加载输入→按MD标题初切→无标题兜底→长切短合→统计输出→结果备份
    核心目的：将长MD文档切分为长度适中的Chunk，适配大模型上下文窗口和向量检索
    后续扩展点：可在各步骤间新增Chunk元信息补充、自定义切分规则、向量入库前置处理等
    :param state: 项目状态字典（ImportGraphState），必须包含md_content/task_id；可选local_dir/max_content_length/file_title
    :return: 更新后的状态字典，新增chunks键（存储最终处理后的Chunk列表，每个Chunk为含title/content/parent_title的字典）
    """
    # 动态获取函数名避免硬编码
    name = sys._getframe().f_code.co_name

    # 节点启动日志，打印当前工作流状态
    logger.debug(f"【{name}】节点启动，\n当前工作流状态：{format_state(state)}")

    # 开始：记录节点运行状态
    add_running_task(state["task_id"], name)

    try:
            # ============1.加载并标准化输入数据================
            # 作用：从状态字典提取md内容/文件标题/chunk最大长度，统一换行消除系统差异，做空值兜底
            # 输出：标准化后的md_content、文件标题、单个chunk最大长度，有无效md内容直接终止节点执行

            content,file_title,max_len = step_1_get_input(state)
            if content is None:
                logger.info(f">>>节点执行终止：{name}（无有效MD内容）")
                return state

            # ============2.按md文件标题做初次切割==============
            # 作用： 基于md文件的标题（#/##/###）切分为独立章节，自动跳过代码块内的伪标题，保证文章语义完整
            # 输出： 初切后的章节列表、识别到的有效标题数量、MD原始文本总行数（为后续统计、日志使用）

            sections,title_count,lines_count = step_2_split_by_titles(content,file_title)

            # ============3.无标题场景兜底处理==================
            # 作用： 解决md文档无任何标题的边界情况，避免后续切分逻辑异常
            # 输出：有标题则返回步骤2的章节列表，无标题则将全文封装为单个【无标题】章节，保证数据格式统一

            sections = step_3_handle_no_title(content,sections,title_count,file_title)

            # ============4.chunk精细化处理===================
            # 作用：核心切分逻辑，先将超长章节按【段落+句子】二次切分，再合并父类标题的过短章节，减少碎片化
            # 额外处理：对所有chunk做parent-title兜底，适配milvus向量库必填字段要求
            # 输出：长度适中、语义完整、低碎片化的最终chunk列表（可直接用于向量入库、大模型调用

            sections = step_4_refine_chunks(sections,max_len)

            # ============5.输出文档切分统计信息================
            # 作用：打印核心统计数据，便于监控切分效果，调试问题（原始行数、最终chunk数、首个chunk预览）
            # 输出：无返回值，仅通过logger输出标准化统计日志

            step_5_print_starts(lines_count,sections)

            # ============6.chunk结果本地备份+状态更新==========
            # 作用1：将最终chunk列表备份到chunk.json，便于后续问题排查、数据复用
            # 作用2：将chunk列表写入状态字典，传递给下一个节点（如向量入库、大模型摘要等）

            state["chunks"] = sections
            step_6_backup(state,sections)

            # 节点执行完成日志
            logger.info(
                f">>> 核心节点执行完成：【文档切分】{name}，已生成{len(sections)}个有效Chunk，结果已写入状态字典")

    except Exception as e:
            # 全局异常捕获，保证节点执行失败不崩溃整个流程，记录错误日志便于排查
            logger.error(f">>>核心节点执行失败：【文档切分】{name}，错误信息：{str(e)}",exc_info=True)

    return state

def step_1_get_input(state:ImportGraphState)->Tuple[Any, str, int]:
    """
    【步骤1】获取并预处理输入数据
    功能：从状态字典中提取MD内容/文件标题/最大长度，做基础标准化
    :param state: 项目状态字典（ImportGraphState），包含md_content等核心键
    :return: 标准化后的MD内容/文件标题/单个Chunk最大长度（无内容则返回None,None,None）
    """
    md_content = state.get("md_content")
    if md_content is None:
        logger.warning("状态字典中无有效MD内容，终止文档切分")
        return None,None,None

    # 基础标准化，统一换行符，避免windows/linux换行差异导致的后续异常处理
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

    # 提取文件标题
    file_title = state.get("file_title","unknown file")

    # chunk最大长度
    max_len = DEFAULT_MAX_CONTENT_LENGTH

    logger.info(f"步骤1：输入数据加载完成，文件标题：{file_title}，最大Chunk长度：{max_len}")

    return md_content,file_title,max_len

def step_2_split_by_titles(content: str, file_title: str) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    【步骤2】按Markdown标题初次切分（核心：按#分级切分，跳过代码块内标题）
    LangChain前置预处理：将整份MD按标题拆分为独立章节，为后续精细化切分做基础
    :param content: 标准化后的MD完整内容（字符串）
    :param file_title: 所属文件标题，用于标记章节归属
    :return: 切分后的章节列表/有效标题数量/原始文本总行数
    """
    # 正则匹配Markdown 1-6级标题（核心规则，适配缩进/标准格式）
    # ^\s*：行首允许0/多个空格/Tab（兼容缩进的标题）
    # #{1,6}：匹配1-6个#（对应MD1-6级标题）
    # \s+：#后必须有至少1个空格（区分#是标题还是普通文本）
    # .+：标题文字至少1个字符（避免空标题）

    title_pattern = r'^\s*#{1,6}\s+.+'

    # 将md内容按换行符拆分为行列表，逐行处理
    lines = content.split("\n")
    sections = [] #最终切分的章节列表
    current_title = "" #当前章节标题
    current_lines = [] #当前章节的行缓存
    title_count = 0 #有效标题数量（非代码块内）
    in_code_block = False #代码块标记：避免误判代码块内的#伪标题

    def _flush_section():
        """内部辅助函数，将当前缓存的章节写入sections。空缓存则跳过"""
        if not current_lines:
            return
        sections.append({
            "title": current_title,
            # 每段时间使用 \n换行区分
            "content": "\n".join(current_lines),
            "file_title": file_title
        })

    # 逐行遍历，识别标题并切分章节
    for line in lines:
        stripped_line = line.strip()
        # 识别代码块边界（```/~~~）：进入退出代码块时翻转状态
        if stripped_line.startswith("```") or stripped_line.startswith("~~~"):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue

        #判断是否为有效标题：非代码块内 + 匹配标题正则
        is_valid_title = (not in_code_block) and re.match(title_pattern, line)
        if is_valid_title:
            #遇到新标题：先将上一个章节写入结果，再初始化新章节
            _flush_section()
            current_title = line.strip() #清理标题前后空格
            current_lines = [current_title]
            title_count += 1
            logger.debug(f"识别到MD标题：{current_title}")
        else:
            # 普通行：追加到当前章节的行缓存
            current_lines.append(line)

    #处理最后一个章节：循环结束后，将最后一个缓存章节写入结构
    _flush_section()
    logger.info(f"步骤2：MD标题切分完成，识别到{title_count}个有效标题，原始文本共{len(lines)}行")
    return sections,title_count,len(lines)

def step_3_handle_no_title(content: str, sections: List[Dict[str,Any]], title_count: int, file_title: str) -> List[Dict[str, Any]]:
    """
    【步骤3】无标题兜底处理
    功能：若MD中未识别到任何标题，将全文作为一个整体处理，避免后续逻辑异常
    :param content: 标准化后的MD完整内容
    :param sections: 步骤2切分后的章节列表
    :param title_count: 步骤2识别的有效标题数量
    :param file_title: 所属文件标题
    :return: 兜底后的章节列表
    """
    if title_count == 0:
        # 无标题情况：替换为单章节，标题为"无标题"
        logger.warning(f"步骤3：未识别到任何MD标题，将全文作为单个章节处理，文件：{file_title}")
        return [{"title": "无标题","content": content, "file_title": file_title}]
    logger.debug(f"步骤3：检测到{title_count}个有效标题，无需兜底处理")
    return sections

def step_4_refine_chunks(sections: List[Dict[str,Any]], max_len: int) -> List[Dict[str, Any]]:
    """
    【步骤4】Chunk精细化处理（核心：长切短合，适配大模型/检索）
    执行流程：1.切分超长章节 2.合并过短章节 3.父标题兜底（适配Milvus向量库schema）
    :param sections: 步骤3处理后的章节列表
    :param max_len: 单个Chunk最大字符长度
    :return: 长度适中、低碎片化的最终Chunk列表
    """
    # 边界处理：最大长度无效（空/≤0）,直接返回原章节，避免切分异常
    if not max_len or max_len <= 0:
        logger.warning(f"步骤4：Chunk最大长度配置无效（{max_len}），跳过精细化处理")
        return sections

    # 阶段1、切分超长章节 -> 所有章节控制在max_len之内
    refined_split = []
    for section in sections:
        # 对每个章节执行超长切分，结果平铺加入列表（避免嵌套）
        # extend的作用就是： 把另外一个列表（或可迭代对象）的元素，一个个拆出来，直接追加到当前列表的尾部
        refined_split.extend(_split_long_section(section, max_len))
    logger.info(f"步骤4-1：超长章节切分完成，共生成{len(refined_split)}个初始子Chunk")

    # 阶段2、合并过短章节 -> 减少碎片化，提升后续检索/大模型调用效果
    final_sections = _merge_short_sections(refined_split, max_len=max_len)
    logger.info(f"步骤4-2：过短章节合并完成，最终得到{len(final_sections)}个Chunk")

    # 为所有Chunk补齐父标题，保证后续向量入库和来源追溯字段稳定。
    for section in final_sections:
        if not section.get("parent_title"):
            section["parent_title"] = section.get("title") or "无标题"

    return final_sections

def step_5_print_starts(lines_count, sections):
    """
    【步骤5】输出文档切分统计信息（日志记录，便于监控/调试）
    :param lines_count: MD原始文本总行数
    :param sections: 最终处理后的Chunk列表
    """
    chunk_num = len(sections)
    # 输出核心统计信息：原始行数/最终Chunk数/首个Chunk预览
    logger.info("-" * 50 + " 文档切分统计信息 " + "-" * 50)
    logger.info(f"MD原始文本总行数：{lines_count}")
    logger.info(f"最终生成Chunk数量：{chunk_num}")
    if sections:
        first_title = sections[0].get("title", "无标题")
        logger.info(f"首个Chunk标题预览：{first_title}")
    logger.info("-" * 110)
    pass

def step_6_backup(state: ImportGraphState, sections: List[Dict[str, Any]]) -> None:
    """
    【步骤6】Chunk结果本地JSON备份（便于调试/问题排查，保留处理结果）
    备份结构：Markdown所在目录/chunks/chunk.json
    :param state: 项目状态字典，需包含md_path（当前Markdown文件路径）
    :param sections: 最终处理后的Chunk列表
    """
    # 获取当前Markdown路径。备份目录基于md_path计算，不改变前面节点已有的文件路径设计。
    md_path = (state.get("md_path") or "").strip()
    if not md_path:
        logger.warning("步骤6：未配置Markdown路径（md_path），跳过Chunk结果备份")
        return

    try:
        # 找到Markdown所在目录，例如：output/test/test_new.md -> output/test。
        document_dir = os.path.dirname(os.path.abspath(md_path))

        # 在Markdown目录下创建专用chunks目录；目录已存在时不会报错。
        backup_dir = os.path.join(document_dir, "chunks")
        os.makedirs(backup_dir, exist_ok=True)

        # 每个文档目录固定使用chunk.json，最终路径为：output/test/chunks/chunk.json。
        backup_path = os.path.join(backup_dir, "chunk.json")

        # 使用w模式写入：重新处理同一文档时，覆盖该文档自己的旧备份。
        with open(backup_path, "w", encoding="utf-8") as f:
            """
            sections是Python 嵌套数据结构（List[Dict[str, Any]]，列表里装字典，字典里可能嵌套字符串 / 数字等），而普通文件写入
            （如f.write(sections)）仅支持写入字符串，直接写 Python 数据结构会报错。
            json.dump的核心作用就是：将 Python 原生数据结构（列表、字典、字符串、数字等）直接序列化并写入 JSON 文件，无需手动转换为字符串，
            同时保证数据格式规范、可跨语言 / 跨场景读取，完美适配「Chunk 列表备份」的需求。
            """
            json.dump(
                sections,
                f,
                # 开启 True："title": "\u4e00\u7ea7\u6807\u9898"（乱码，无法直接看）；
                # 开启 False："title": "一级标题"（正常中文，人工可直接阅读）。
                ensure_ascii=False, # 保留中文，不转义为\u编码
                indent=2  # 格式化缩进，便于阅读
            )
        logger.info(f"步骤6：Chunk结果备份成功，备份文件路径：{backup_path}")
    except Exception as e:
        # 备份失败仅记录日志，不终止主流程
        logger.error(f"步骤6：Chunk结果备份失败，错误信息：{str(e)}", exc_info=False)

def _split_long_section(section: Dict[str,Any], max_len: int = DEFAULT_MAX_CONTENT_LENGTH) -> List[Dict[str, Any]]:
    """
    【辅助函数】超长章节二次切分（核心适配LangChain分割器）
    功能：单个章节内容超限时，按「段落→句子→空格」从粗到细切分，保留语义
    切分规则：1.先按空行(段落) 2.再按换行 3.最后按中英文标点/空格
    :param section: 原始章节字典，必须包含content键，可选title/file_title等
    :param max_length: 单个Chunk最大字符长度，默认使用全局配置
    :return: 切分后的子章节列表，每个子章节带父标题/序号等元信息
    """
    # 内容空值兜底：无内容直接返回原章节
    content = section.get("content","") or ""
    # 长度未超限，无需切分，直接返回原章节（列表格式保持统一）
    if len(content) <= max_len:
        unchanged_section = section.copy()
        # 未二次切分的章节也要标记自己的父标题，避免不同章节因parent_title都为None而被误合并。
        unchanged_section.setdefault("parent_title", unchanged_section.get("title") or "无标题")
        return [unchanged_section]

    # 标准化预处理：统一换行符，避免不同系统（\r\n/\n)导致的切分异常
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    # 提取章节标题，用于组装子chunk前缀（保留上下文）
    title = section.get("title","") or ""
    # 标题前缀：带空行分隔符，与正文区分开
    prefix = f"{title}\n\n" if title else ""
    # 计算正文可用长度： 总长度 - 标题前缀长度（避免标题占满chunk额度）
    available_len = max_len - len(prefix)
    # 极端情况： 标题长度超过阈值，无法切分，返回原章节
    if available_len <= 0:
        logger.warning(f"章节标题过长，无法切分：{title[:20]}...")
        return [section]
    # 清理正文重复标题：避免原章节中正文开头重复标题，导致子类chunk冗余
    body = content
    if title and body.lstrip().startswith(title):
        body = body[body.find(title) + len(title):].lstrip()

    # 初始化LangChain递归分割器（核心工具：按优先级分隔符切分，保留语义）
    # separators： 分隔符优先级（从粗到细），优先按大语义单元切分，最后才硬拆
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=available_len, #正文最大长度（已扣除标题）
        chunk_overlap=0, #无重叠度，按标题切分后语义完整，无需重叠
        # 分割符优先级：空行(段落)→换行→中文标点→英文标点→空格，最后硬拆
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "]
    )

    # 切分正文并组装子章节（带完整元信息，便于溯源）
    sub_sections = []
    for idx,chunk in enumerate(splitter.split_text(body),start=1):
        # 清理空内容，跳过切分后的空字符串
        text = chunk.lstrip()
        if not text:
            continue

        # 组装子chunk完整内容：标题前缀 + 切分后的正文
        full_text = (prefix + text).strip()
        # 子章节元信息：保留父级关联，添加序号，便于后续检索/溯源
        sub_sections.append({
            "title": f"{title}-{idx}" if title else "chunk-{idx}", # 子chunk标题（带序号）
            "content": full_text,
            "parent_title": title,
            "part": idx,
            "file_title": section.get("file_title","")
        })
    logger.info(f"超长章节切分完成：{title} → 生成{len(sub_sections)}个子Chunk")

    return sub_sections

def _merge_short_sections(
        sections: List[Dict[str, Any]],
        min_len: int = MIN_CONTENT_LENGTH,
        max_len: int = DEFAULT_MAX_CONTENT_LENGTH
) -> List[Dict[str, Any]]:
    """
    【辅助函数】过短章节合并（减少碎片化，提升检索效果）
    核心规则：仅合并「同父标题」且「当前块长度不足阈值」的相邻Chunk，避免跨章节合并
    :param sections: 待合并的Chunk列表（通常是_split_long_section切分后的结果）
    :param min_length: 最小长度阈值，低于此值的Chunk会被合并
    :return: 合并后的Chunk列表，长度适中，保留元信息
    """

    # 边界处理：空列表直接返回，避免后续索引报错
    if not sections:
        logger.debug("待合并Chunk列表为空，直接返回")
        return []

    merged_sections = [] #最终合并结果
    current_chunk = None #迭代累加器：保存当前待合并的chunk

    for section in sections:
        # 初始化： 第一个chunk直接作为当前待合并块
        if current_chunk is None:
            current_chunk = section
            continue

        #合并条件： 1.当前块长度不足阈值 2. 当前遍历到的 section 与 current_chunk 属于同一个原章节
        is_current_short = len(current_chunk["content"]) < min_len
        is_same_parent = current_chunk.get("parent_title") == section.get("parent_title")

        # 合并前清理：去掉下一块开头重复的父标题，避免内容冗余。
        parent_title = section.get("parent_title", "")
        next_content = section.get("content", "") or ""
        if parent_title and next_content.startswith(parent_title):
            next_content = next_content[len(parent_title):].lstrip()
        merged_content = current_chunk["content"] + "\n\n" + next_content
        is_within_max_len = len(merged_content) <= max_len

        if is_current_short and is_same_parent and is_within_max_len:
            # 合并内容，空行分隔，并确保合并后仍不超过单个Chunk最大长度。
            current_chunk["content"] = merged_content
            #更新子chunk序号：保留最新序号，便于溯源
            if "part" in section:
                current_chunk["part"] = section["part"]
            logger.debug(f"合并短Chunk：{current_chunk.get('parent_title')} → 累计长度{len(current_chunk['content'])}")
        else:
            #不满足合并条件：将当前块加入结果，切换为新的待合并块
            merged_sections.append(current_chunk)
            current_chunk = section

    #循环结束后，将最后一个待合并块加入结果
    if current_chunk is not None:
        merged_sections.append(current_chunk)
    logger.debug(f"短Chunk合并完成：原{len(sections)}个 → 合并后{len(merged_sections)}个")

    return merged_sections



#测试 阶段1-3 文本精细切割
# if __name__ == "__main__":
#     """仅测试阶段1：读取真实 Markdown，并输出按标题切分的结果。"""
#     from pathlib import Path
#
#     if hasattr(sys.stdout, "reconfigure"):
#         sys.stdout.reconfigure(encoding="utf-8", errors="replace")
#
#     md_path = Path(__file__).resolve().parents[3] / "output" / "test" / "test_new.md"
#     if not md_path.exists():
#         raise FileNotFoundError(f"测试文件不存在：{md_path}")
#
#     content = md_path.read_text(encoding="utf-8")
#     file_title = md_path.stem
#
#     state = {
#         "md_content": content,
#         "file_title": file_title,
#     }
#     normalized_content, normalized_title, _ = step_1_get_input(state)
#     sections, title_count, lines_count = step_2_split_by_titles(
#         normalized_content,
#         normalized_title,
#     )
#     sections = step_3_handle_no_title(
#         normalized_content,
#         sections,
#         title_count,
#         normalized_title,
#     )
#
#     print("=" * 80)
#     print("步骤1-3：Markdown 标题初步切分测试")
#     print(f"测试文件：{md_path}")
#     print(f"原始行数：{lines_count}")
#     print(f"识别标题数：{title_count}")
#     print(f"步骤3处理后章节数：{len(sections)}")
#     print("说明：本次测试未执行步骤4的最大长度二次切分")
#     print("=" * 80)
#
#     for index, section in enumerate(sections, start=1):
#         preview = section["content"][:300].replace("\n", "\\n")
#         print(f"\n[{index}] title: {section['title'] or '(无标题前置内容)'}")
#         print(f"content_length: {len(section['content'])}")
#         print(f"content_preview: {preview}")
