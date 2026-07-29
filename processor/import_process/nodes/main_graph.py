from pathlib import Path
from uuid import uuid4

from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.common.logger import logger
from processor.config.milvus_config import milvus_config
from processor.import_process.nodes.node_bge_embedding import NodeBgeEmbedding
from processor.import_process.nodes.node_document_split import NodeDocumentSplit
from processor.import_process.nodes.node_entry import NodeEntry
from processor.import_process.nodes.node_import_milvus import NodeImportMilvus
from processor.import_process.nodes.node_item_name_recognition import NodeItemNameRecognition
from processor.import_process.nodes.node_md_img import NodeMdImg
from processor.import_process.nodes.node_pdf_to_md import NodePdfToMd
from processor.import_process.core.state import ImportGraphState,create_default_state
from processor.utils.client.milvus_client import get_milvus_client
from processor.utils.escape_milvus_string_utils import escape_milvus_string

# 1.创建状态图
workflow = StateGraph(ImportGraphState)

# 2.注册所有节点
# 2.1 创建节点
node_entry = NodeEntry()
node_pdf_to_md = NodePdfToMd()
node_md_img = NodeMdImg()
node_document_split = NodeDocumentSplit()
node_item_name_recognition = NodeItemNameRecognition()
node_bge_embedding = NodeBgeEmbedding()
node_import_milvus = NodeImportMilvus()

# 2.2 注册节点
workflow.add_node("node_entry", node_entry)
workflow.add_node("node_pdf_to_md", node_pdf_to_md)
workflow.add_node("node_md_img", node_md_img)
workflow.add_node("node_document_split", node_document_split)
workflow.add_node("node_item_name_recognition", node_item_name_recognition)
workflow.add_node("node_bge_embedding", node_bge_embedding)
workflow.add_node("node_import_milvus", node_import_milvus)

# 3.设置入口节点
workflow.set_entry_point("node_entry")

# 4.定义条件边
# 4.1 创建条件路由
def route_after_entry(state:ImportGraphState)->str:
    if state.get("is_pdf_read_enabled"):
        return "node_pdf_to_md"
    elif state.get("is_md_read_enabled"):
        return "node_md_img"
    else:
        return END

# 4.2 注册条件边
workflow.add_conditional_edges(
    "node_entry",
    route_after_entry,
    # 如果不执行后面的 print_ascii()的话，这句话可以省略掉
    {
        "node_pdf_to_md": "node_pdf_to_md",
        "node_md_img": "node_md_img",
        END: END
    }
)

# 5.注册顺序边
workflow.add_edge("node_pdf_to_md","node_md_img")
workflow.add_edge("node_md_img","node_document_split")
workflow.add_edge("node_document_split","node_item_name_recognition")
workflow.add_edge("node_item_name_recognition","node_bge_embedding")
workflow.add_edge("node_bge_embedding","node_import_milvus")
workflow.add_edge("node_import_milvus",END)

# 6.编译工作流
kb_import_app = workflow.compile()

# 真实联调测试
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    pdf_path = (
        project_root
        / "doc"
        / "H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf"
    )
    output_dir = project_root / "output"

    if not pdf_path.is_file():
        raise FileNotFoundError(f"真实联调 PDF 不存在：{pdf_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    init_state = create_default_state(
        task_id=f"main_graph_real_{uuid4().hex}",
        import_file_path=str(pdf_path),
        local_dir=str(output_dir),
    )

    logger.info("===== main_graph 七节点真实联调开始 =====")
    logger.info(f"输入 PDF：{pdf_path}")
    logger.info(f"本地产物目录：{output_dir}")

    final_state = kb_import_app.invoke(init_state)
    chunks = final_state.get("chunks", [])
    item_name = final_state.get("item_name", "")
    md_path = Path(final_state.get("md_path", ""))

    if not md_path.is_file():
        raise RuntimeError(f"联调失败：最终 Markdown 不存在：{md_path}")
    if not chunks:
        raise RuntimeError("联调失败：未生成任何 Chunk")

    missing_vectors = [
        index
        for index, chunk in enumerate(chunks, start=1)
        if not chunk.get("dense_vector") or not chunk.get("sparse_vector")
    ]
    if missing_vectors:
        raise RuntimeError(f"联调失败：第 {missing_vectors} 个 Chunk 缺少双向量")

    missing_ids = [
        index
        for index, chunk in enumerate(chunks, start=1)
        if not str(chunk.get("chunk_id", "")).isdigit()
    ]
    if missing_ids:
        raise RuntimeError(f"联调失败：第 {missing_ids} 个 Chunk 未回填 Milvus 主键")

    client = get_milvus_client()
    if client is None:
        raise RuntimeError("联调失败：无法连接 Milvus 进行结果查询")

    collection_name = milvus_config.chunks_collection
    filter_expr = f'item_name == "{escape_milvus_string(item_name)}"'
    client.flush(collection_name=collection_name)
    client.load_collection(collection_name=collection_name)
    persisted_rows = client.query(
        collection_name=collection_name,
        filter=filter_expr,
        output_fields=["chunk_id", "item_name"],
    )
    if len(persisted_rows) != len(chunks):
        raise RuntimeError(
            "联调失败：Milvus 实际查询数量与状态 Chunk 数量不一致，"
            f"state={len(chunks)}, milvus={len(persisted_rows)}"
        )

    logger.info("===== main_graph 七节点真实联调成功 =====")
    logger.info(f"任务 ID：{final_state['task_id']}")
    logger.info(f"识别主体：{item_name}")
    logger.info(f"Markdown：{md_path}")
    logger.info(f"Chunk 数量：{len(chunks)}")
    logger.info(f"Milvus 集合：{collection_name}")
    logger.info(f"Milvus 查询数量：{len(persisted_rows)}")

