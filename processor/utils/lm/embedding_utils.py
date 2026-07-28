#模型单例对象，避免重复初始化
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from processor.common.logger import logger
from processor.config.embedding_config import embeddings_config

# 模型单例对象，避免重复初始化
_bge_m3_ef = None

def get_bge_m3_ef():
    """
    获取BGE-M3模型单例对象，自动加载环境变量配置
    :return: 初始化完成的BGEM3EmbeddingFunction实例
    """
    global _bge_m3_ef

    # 单例模式：已初始化则直接返回，避免重复加载模型
    if _bge_m3_ef is not None:
        logger.debug("BGE-M3模型单例已存在，直接返回实例")
        return _bge_m3_ef

    # 从环境变量加载配置，无配置则使用默认值
    # 本地有可以使用本地址！ 没有则使用"BAA/bge-m3"会自动下载！ 如果云端部署也可以使用url地址
    model_name = embeddings_config.beg_m3_path or "BAAI/bge-m3"
    device = embeddings_config.bge_device or "cpu"
    use_fp16 = embeddings_config.bge_fp16 or False

    # 打印模型初始化配置，便于问题排查
    logger.info(
        "开始初始化BGE-M3模型",
        extra={
            "model_name": model_name,
            "device": device,
            "use_fp16": use_fp16,
            "normalize_embeddings": True
        }
    )

    try:
        # 初始化BGE-M3模型，开启原生L2归一（适配Milvus IP内积检索）
        _bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
            normalize_embeddings=True # 模型原生对稠密+稀疏向量做L2归一化
        )
        logger.success("BGE-M3模型初始化成功，已开启原生L2归一化") #它把所有向量拉伸到同一长度（模长1）,让我们可以在数据库中放心使用最快的IP内积检索，既提速又不丢精度
        return _bge_m3_ef
    except Exception as e:
        logger.error(f"BGE-M3模型初始化失败：{str(e)}", exc_info=True)
        raise #向上抛出异常，由调用方处理

def generate_embedding(texts):
    """
    为文本列表生成稠密+稀疏混合向量嵌入（模型原生L2归一化）
    :param texts: 要生成嵌入的文本列表，单文本也需封装为列表
    :return: 字典格式的向量结果，key为dense/sparse，对应嵌套列表/字典列表
    :raise: 向量生成过程中的异常，由调用方捕获处理
    """

    # 合法参数校验
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("生成向量入参不合法，texts必须为非空列表")
        raise ValueError("参数texts必须是包含文本的非空列表")

    logger.info(f"开始为{len(texts)}条文本生成混合向量嵌入")
    try:
        # 加载BGE-M3模型单例
        model = get_bge_m3_ef()
        # 模型编码生成向量，返回dense（稠密向量）+ sparse（CSR格式稀疏向量）
        embeddings = model.encode_documents(texts)
        logger.debug(f"模型编码完成，开始解析稀疏向量格式，共{len(texts)}条")

        # sparse 是一个批量 CSR 稀疏矩阵：所有文本的非零数据放在同一组
        # indices/data 数组中，indptr 用来记录每条文本的数据边界。
        sparse_matrix = embeddings["sparse"]
        processed_sparse = []

        for i in range(len(texts)):
            # i 是当前文本在输入列表中的位置。例如 i=0 表示第一段文本。
            # CSR 的 indptr 长度比文本数多 1：
            # - indptr[i]     是当前文本非零数据的起点（包含该位置）
            # - indptr[i + 1] 是当前文本非零数据的终点（不包含该位置）
            # 因此 [start:end] 恰好只取到第 i 条文本对应的数据。
            start = sparse_matrix.indptr[i]
            end = sparse_matrix.indptr[i + 1]

            # indices 保存“非零权重位于哪个向量维度”。
            # 例如 [12, 48] 表示当前文本只在第 12、48 维有非零权重。
            # tolist() 将 NumPy 数组和其中的 NumPy 整数转换为普通 Python 列表/整数，
            # 便于 JSON 序列化和后续写入 Milvus。
            sparse_indices = sparse_matrix.indices[start:end].tolist()

            # data 与 indices 按位置一一对应，保存同一个维度的权重。
            # 例如 indices=[12, 48]、data=[0.63, 0.21] 表示：
            # 第 12 维权重为 0.63，第 48 维权重为 0.21。
            # 如果该文本没有任何非零项，则 start == end，两个列表都会为空。
            sparse_data = sparse_matrix.data[start:end].tolist()

            # zip() 将两个列表逐项配对：(12, 0.63)、(48, 0.21)。
            # 字典最终表示为 {特征维度索引: 权重}，只保存非零项，
            # 相比保存完整的长向量会节省大量内存和数据库空间。
            sparse_dict = {index: weight for index, weight in zip(sparse_indices, sparse_data)}
            processed_sparse.append(sparse_dict)

        # dense 是每条文本对应的完整稠密向量。列表推导式逐条调用 tolist()，
        # 将 NumPy 数组转换为普通的 list[float]，顺序与 texts 保持一一对应。
        result = {
            "dense": [embedding.tolist() for embedding in embeddings["dense"]],
            "sparse": processed_sparse,
        }

        logger.success(f"{len(texts)}条文本向量生成完成，格式已适配工业级使用")
        return result

    except Exception as e:
        logger.error(f"文本向量生成失败：{str(e)}", exc_info=True)
        raise  # 不吞异常，向上传递让调用方做重试/降级处理

"""
核心设计亮点&适配说明：
1. 模型原生归一化：开启normalize_embeddings = True，自动对稠密+稀疏向量做L2归一化，完美适配Milvus IP内积检索（单位化后IP等价于余弦，计算更快）；
2. 彻底解决NumPy类型做key问题：sparse_indices加.tolist()，将np.int64转为Python原生int，满足字典key的可哈希要求，无报错风险；
3. 稀疏值适配序列化：sparse_data加.tolist()，将np.float32转为Python原生float，支持JSON写入/接口返回/Milvus入库等所有场景；
4. 单例模式优化：模型仅初始化一次，避免重复加载耗时耗资源，提升批量处理效率；
5. 格式匹配业务调用：返回dense嵌套列表、sparse字典列表，与vector_result["dense"][0]/sparse_vector["sparse"][0]取值逻辑完美契合；
6. 分级日志覆盖：从模型初始化、向量生成到异常报错，全流程日志记录，便于生产环境问题排查；
7. 入参合法性校验：防止空列表/非列表入参导致的内部报错，提升工具类健壮性。
"""
