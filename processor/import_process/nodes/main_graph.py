from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.import_process.nodes.node_bge_embedding import NodeBgeEmbedding
from processor.import_process.nodes.node_document_split import NodeDocumentSplit
from processor.import_process.nodes.node_entry import NodeEntry
from processor.import_process.nodes.node_import_milvus import NodeImportMilvus
from processor.import_process.nodes.node_item_name_recognition import NodeItemNameRecognition
from processor.import_process.nodes.node_md_img import NodeMdImg
from processor.import_process.nodes.node_pdf_to_md import NodePdfToMd
from processor.import_process.core.state import ImportGraphState,create_default_state

# 1.创建状态图
workflow = StateGraph(ImportGraphState)

# 2.注册所有节点
# 2.1 创建节点
node_enrty = NodeEntry()
node_pdf_to_md = NodePdfToMd()
node_md_img = NodeMdImg()
node_document_split = NodeDocumentSplit()
node_item_name_recognition = NodeItemNameRecognition()
node_bge_embedding = NodeBgeEmbedding()
node_import_milvus = NodeImportMilvus()

# 2.2 注册节点
workflow.add_node("node_enrty", node_enrty)
workflow.add_node("node_pdf_to_md", node_pdf_to_md)
workflow.add_node("node_md_img", node_md_img)
workflow.add_node("node_document_split", node_document_split)
workflow.add_node("node_item_name_recognition", node_item_name_recognition)
workflow.add_node("node_bge_embedding", node_bge_embedding)
workflow.add_node("node_import_milvus", node_import_milvus)

# 3.设置入口节点
workflow.set_entry_point("node_enrty")

# 4.定义条件边
# 4.1 创建条件路由
def route_after_entry(state:ImportGraphState)->str:
    if state['is_md_read_enabled']:
        return 'node_pdf_to_md'
    elif state['is_md_read_enabled']:
        return 'node_md_img'
    else:
        return END

# 4.2 注册条件边
workflow.add_conditional_edges(
    "node_enrty",
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

# 测试
if __name__ == "__main__":
    from processor.import_process import logger

    # 1 初始化状态信息
    init_state = create_default_state(
        task_id="task_001",
        local_file_path="d:/abc.cmd"
    )

    # 2（invoke） 运行工作流
    # 在图的外部，仅在整个图的执行过程都结束之后，我们才能够拿到最终的状态
    # final_state = kb_import_app.invoke(init_state)
    # print(format_state(final_state))

    # 2（stream）运行工作流
    # 在图的外部，每执行完一个节点，就可以输出当前的state
    for chunk in kb_import_app.stream(init_state):
        # chunk：字典
        logger.info(chunk.keys())
        logger.info(chunk.items())

    logger.info("输出图结构:")
    # 以下代码需要 uv add grandalf
    kb_import_app.get_graph().print_ascii()

