import json
from pathlib import Path
from typing import List, Dict, Any

from processor.common.logger import logger
from processor.import_process.core import state
from processor.import_process.core.base import BaseNode
from processor.import_process.core.state import ImportGraphState, create_default_state
from processor.utils.lm.embedding_utils import get_bge_m3_ef, generate_embedding


# ==========================================
# BGE-M3向量化核心节点
# 核心能力：将文本切片转换为稠密/稀疏双向量，为Milvus向量检索提供数据基础
# 依赖模型：BAAI/bge-m3（多语言、多粒度，同时支持语义/关键词检索）
# 向量说明：
#   1. 稠密向量：1024维固定长度，记录文本深层语义信息，用于语义相似度匹配
#   2. 稀疏向量：变长键值对，记录文本关键词/特征位置，用于关键词精准匹配
# 核心设计：
#   - 单例模型：避免重复加载模型，节省显存/时间
#   - 批量处理：分批生成向量，防止大批次导致的显存溢出
#   - 文本增强：拼接商品名+切片内容，强化核心特征，提升检索准确性
# ==========================================# ==========================================
# # BGE-M3向量化核心节点
# # 核心能力：将文本切片转换为稠密/稀疏双向量，为Milvus向量检索提供数据基础
# # 依赖模型：BAAI/bge-m3（多语言、多粒度，同时支持语义/关键词检索）
# # 向量说明：
# #   1. 稠密向量：1024维固定长度，记录文本深层语义信息，用于语义相似度匹配
# #   2. 稀疏向量：变长键值对，记录文本关键词/特征位置，用于关键词精准匹配
# # 核心设计：
# #   - 单例模型：避免重复加载模型，节省显存/时间
# #   - 批量处理：分批生成向量，防止大批次导致的显存溢出
# #   - 文本增强：拼接商品名+切片内容，强化核心特征，提升检索准确性
# # ==========================================

class NodeBgeEmbedding(BaseNode):
    """
    节点: 向量化
    使用 BGE-M3 模型将文本转换为向量。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_bge_embedding"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        LangGraph核心节点：BGE-M3文本向量化处理
        主流程（串行执行，全流程异常隔离）：
            1. 输入校验：验证chunks有效性，核心数据缺失则终止当前节点
            2. 模型初始化：获取BGE-M3单例模型实例，避免重复加载
            3. 批量向量化：分批拼接文本、生成双向量，为切片绑定向量字段
            4. 状态更新：将带向量的chunks更新回全局状态，供下游Milvus入库节点使用
        参数：
            state: ImportGraphState - 流程全局状态对象，包含上游传入的chunks、task_id等数据
        返回：
            ImportGraphState - 更新后的状态对象，chunks字段新增dense_vector/sparse_vector
        异常处理：
            节点内所有异常均捕获，不终止整体LangGraph流程，仅记录错误日志
        """
        try:
            # 步骤1：输入数据校验，核心chunks无效则抛出异常
            text_to_embedding = step_1_validate_input(state)

            # 步骤2：初始化BGE-M3模型（单例模式，仅加载一次）
            bge_m3_ef = step_2_init_model()

            # 步骤3：批量生成双向量，为切片绑定向量字段
            output_data = step_3_generate_embeddings(text_to_embedding, bge_m3_ef)

            # 步骤4：更新全局状态，将带向量的chunks回传下游
            state['chunks'] = output_data
            logger.info(f"--- BGE-M3 向量化处理完成，共处理 {len(output_data)} 条文本切片 ---")

        except Exception as e:
            # 捕获节点所有异常，记录错误堆栈，不中断整体流程
            logger.error(f"BGE-M3向量化节点执行失败：{str(e)}", exc_info=True)

        # 返回更新后的状态对象，传递至下游节点
        return state

def step_1_validate_input(state:ImportGraphState)->List[Dict[str, Any]]:
    """
    向量化前置步骤1：输入数据有效性校验
    核心作用：
        1. 从全局状态提取待向量化的chunks切片列表
        2. 严格校验chunks类型和非空性，无有效数据则终止向量化
    参数：
        state: ImportGraphState - 流程全局状态对象
    返回：
        List[Dict[str, Any]] - 校验通过的文本切片列表
    异常：
        若chunks非列表/为空，抛出ValueError，终止当前向量化流程
    """
    # 从状态字典中提取切片数据
    texts_to_embedding = state.get('chunks')

    #校验，必须是非空列表，否则无法进行向量化
    if not isinstance(texts_to_embedding, list) or not texts_to_embedding:
        logger.error("向量化输入校验失败：chunks字段为空或非有效列表")
        raise ValueError("错误：无有效文本切片数据，无法执行向量化处理")

    logger.info(f"向量化输入校验通过，待处理文本切片数量：{len(texts_to_embedding)}")
    return texts_to_embedding

def step_2_init_model():
    """
    向量化步骤2：初始化BGE-M3模型实例（单例模式）
    核心作用：
        1. 调用单例函数get_bge_m3_ef，确保模型全局仅加载一次
        2. 校验模型实例有效性，加载失败则抛出明确异常
    返回：
        Any - 有效BGE-M3模型实例（embedding function）
    异常：
        模型加载失败（路径错误/显存不足/依赖缺失）时，抛出ValueError并提示配置问题
    """
    try:
        # 获取单例模型实例，避免重复加载浪费资源
        ef = get_bge_m3_ef()
        # 校验模型是否有效
        if ef is None:
            raise ValueError("BGE-M3模型实例为None：pymilvus.model模块未找到或模型加载失败")

        logger.info("BGE-M3模型实例初始化成功（单例模式）")
        return ef

    except Exception as e:
        # 包装异常信息，明确错误原因和排查方向
        error_msg = f"BGE-M3模型初始化失败：{e}，请检查模型路径/环境变量配置是否正确"
        logger.error(error_msg)
        raise ValueError(error_msg)

def step_3_generate_embeddings(text_to_embedding: List[Dict[str,Any]], bge_m3_ef: Any)->List[Dict[str, Any]]:
    """
    向量化核心步骤3：批量生成稠密/稀疏双向量
    核心逻辑（分批执行，每批独立异常处理）：
        1. 文本拼接：item_name（商品名）+ 换行 + content（切片内容），强化核心特征
        2. 批量调用：传入拼接后的文本，生成批量双向量
        3. 向量绑定：为每个切片复制原数据，新增dense_vector/sparse_vector字段
        4. 异常兜底：单批次失败则保留原切片数据，继续处理下一批次
    参数：
        texts_to_embed: List[Dict[str, Any]] - 校验通过的文本切片列表，含item_name/content字段
        bge_m3_ef: Any - 步骤2初始化的BGE-M3模型实例
    返回：
        List[Dict[str, Any]] - 带向量字段的文本切片列表，异常批次保留原数据
    关键配置：
        batch_size: 每批处理5条，可根据服务器显存大小调整（显存大则调大，反之调小）
    """
    # 初始化结果列表，存储带向量的切片数据
    output_data = []
    # 初始化大小配置：平衡显存占用和处理效率，建议根据实际环境调整
    batch_size = 5

    # 按批次遍历，避免一次性处理过多数据导致显存溢出（OOM）
    total = len(text_to_embedding)
    for i in range (0, total, batch_size):
        # 截取当前批次的切片，最后一批自动适配剩余数量【每次获取5个】
        batch_texts = text_to_embedding[i:i + batch_size]
        # 计算当前批次的起止索引，用于日志展示（方便从1开始，也不获取下标，没有影响）
        start_idx,end_idx = i+1,min(i+len(batch_texts),total)

        try:
            # 构造模型输入文本：拼接商品名+切片内容，增强核心特征
            input_texts = []
            for doc in batch_texts:
                item_name = doc['item_name']
                content = doc['content']
                # 有商品名则拼接（换行分隔提升模型识别效率），无则直接使用内容
                # 几乎所有的 Embedding 模型（尤其是基于 BERT 架构的），对前 128 个 token 的注意力是最集中的。越往后的词，对最终向量方向的拉扯力越弱。
                # **“核心词前置”**的原则
                # 方案 1：用强标点代替换行（最简单、最推荐）
                # 优化前：苹果手机\n性能很好...
                # 优化后：苹果手机。性能很好...
                # 方案2：加一点“微量”的语义胶水（适合属性明确的场景）
                text = f"商品：{item_name}:介绍：{content}" if item_name else content
                # Embedding 模型是个强迫症，你给它喂中文，就用全套中文标点伺候；给它喂英文，就用全套英文标点。保持 语境纯粹 ，生成的向量质量最高！
                input_texts.append(text)

            # 调用封装函数生成批量向量，返回格式：{"dense": [稠密向量列表], "sparse": [稀疏向量列表]}
            docs_embeddings = generate_embedding(input_texts)
            if not docs_embeddings:
                logger.warning(f"第{start_idx}-{end_idx}条切片：向量生成返回空，保留原数据")
                output_data.extend(batch_texts)
                continue

            # 为当前批次每个切片绑定对应向量，复制原数据避免修改上游源数据
            for j,doc in enumerate(batch_texts):
                item = doc.copy()
                item["dense_vector"] = docs_embeddings["dense"][j]
                item["sparse_vector"] = docs_embeddings["sparse"][j]

                output_data.append(item)

            logger.info(f"第{start_idx}-{end_idx}条切片：双向量生成成功")

        except Exception as e:
            # 捕获单批次所有异常，记录错误堆栈，不终止整体批量处理
            logger.error(
                f"第{start_idx}-{end_idx}条切片：向量生成失败，保留原数据 | 错误原因：{str(e)}",exc_info=True)

            # 异常批次保留原切片数据，保证数据完整性，后续可人工排查
            output_data.extend(batch_texts)
            continue
    return output_data


def run_hak180_debug() -> ImportGraphState:
    """使用 HAK180 真实切片运行节点，并输出便于人工检查的调试结果。"""
    project_root = Path(__file__).resolve().parents[3]
    chunks_path = (
        project_root
        / "output"
        / "hak180产品安全手册"
        / "chunks"
        / "chunks.json"
    )
    if not chunks_path.exists():
        raise FileNotFoundError(f"HAK180 切片文件不存在：{chunks_path}")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"HAK180 切片文件不是非空列表：{chunks_path}")

    item_name = "BrotherHAK180烫金机"
    for chunk in chunks:
        chunk["item_name"] = item_name

    initial_state = create_default_state(
        task_id="debug_hak180_bge_embedding",
        file_title="hak180产品安全手册",
        item_name=item_name,
        chunks=chunks,
    )
    final_state = NodeBgeEmbedding()(initial_state)
    output_chunks = final_state.get("chunks", [])

    print("\n===== HAK180 BGE-M3 调试结果 =====")
    print(f"输入切片：{len(chunks)} 条")
    print(f"输出切片：{len(output_chunks)} 条")
    for index, chunk in enumerate(output_chunks, start=1):
        dense_vector = chunk.get("dense_vector") or []
        sparse_vector = chunk.get("sparse_vector") or {}
        dense_preview = [round(value, 6) for value in dense_vector[:5]]
        print(
            f"[{index}] title={chunk.get('title', '')!r}, "
            f"dense_dim={len(dense_vector)}, "
            f"sparse_nonzero={len(sparse_vector)}, "
            f"dense_preview={dense_preview}"
        )

    debug_output_path = chunks_path.with_name("chunks_with_embeddings_debug.json")
    debug_output_path.write_text(
        json.dumps(output_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"完整向量结果：{debug_output_path}")
    print("===================================")
    return final_state


if __name__ == "__main__":
    run_hak180_debug()
