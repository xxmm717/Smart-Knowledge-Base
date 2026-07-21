from typing import Optional

from dotenv import load_dotenv
from langgraph.constants import END
from langgraph.graph import StateGraph
from minio.datatypes import Object

from processor.import_process import logger
from processor.import_process.nodes.node_bge_embedding import NodeBgeEmbedding
from processor.import_process.nodes.node_document_split import NodeDocumentSplit
from processor.import_process.nodes.node_entry import NodeEntry
from processor.import_process.nodes.node_import_milvus import NodeImportMilvus
from processor.import_process.nodes.node_item_name_recognition import NodeItemNameRecognition
from processor.import_process.nodes.node_md_img import NodeMdImg
from processor.import_process.nodes.node_pdf_to_md import NodePdfToMd
from processor.import_process.state import ImportGraphState, create_default_state

load_dotenv()

class KBImportWorkflow:
    """
       知识库导入工作流类
       封装LangGraph工作流的构建、编译、执行逻辑，支持自定义配置和多实例运行
    """
    def __init__(self):
        self.workflow = StateGraph(ImportGraphState)
        self._init_nodes()
        self._register_nodes()
        self._setup_routers()
        self._compiled_app:Optional[Object] = None

    def _init_nodes(self):
        """初始化所有业务节点（私有方法，封装节点创建逻辑）"""
        self.node_entry = NodeEntry()
        self.node_pdf_to_md = NodePdfToMd()
        self.node_md_img = NodeMdImg()
        self.node_document_split = NodeDocumentSplit()
        self.node_item_name_recognition = NodeItemNameRecognition()
        self.node_bge_embedding = NodeBgeEmbedding()
        self.node_import_milvus = NodeImportMilvus()

    def _register_nodes(self):
        """注册所有节点到工作流（私有方法，统一管理节点注册）"""
        # 节点标识与实例属性名保持一致，便于维护
        self.workflow.add_node("node_entry", self.node_entry)
        self.workflow.add_node("node_pdf_to_md", self.node_pdf_to_md)
        self.workflow.add_node("node_md_img", self.node_md_img)
        self.workflow.add_node("node_document_split", self.node_document_split)
        self.workflow.add_node("node_item_name_recognition", self.node_item_name_recognition)
        self.workflow.add_node("node_bge_embedding", self.node_bge_embedding)
        self.workflow.add_node("node_import_milvus", self.node_import_milvus)

    def _route_after_entry(self, state: ImportGraphState) -> str:
        """入口节点后的条件路由函数（私有方法，封装路由逻辑）"""
        if state.get("is_md_read_enabled"):
            return "node_md_img"
        elif state.get("is_pdf_read_enabled"):
            return "node_pdf_to_md"
        else:
            return END

    def _setup_routers(self):
        """设置工作流路由规则（私有方法，封装边的定义）"""
        # 设置入口节点
        self.workflow.set_entry_point("node_entry")
        # 注册条件路由边
        self.workflow.add_conditional_edges(
            "node_entry",
            self._route_after_entry,
            {
                "node_md_img": "node_md_img",
                "node_pdf_to_md": "node_pdf_to_md",
                END: END
            }
        )
        # 注册静态顺序边
        self.workflow.add_edge("node_pdf_to_md", "node_md_img")
        self.workflow.add_edge("node_md_img", "node_document_split")
        self.workflow.add_edge("node_document_split", "node_item_name_recognition")
        self.workflow.add_edge("node_item_name_recognition", "node_bge_embedding")
        self.workflow.add_edge("node_bge_embedding", "node_import_milvus")
        self.workflow.add_edge("node_import_milvus", END)

    def compile(self):
        """编译工作流（如果已编译则直接获取结果）"""
        if not self._compiled_app:
            self._compiled_app = self.workflow.compile()
        return self._compiled_app

    def run(self, initial_state: ImportGraphState, stream: bool = False) -> ImportGraphState:
        """
        统一执行入口，支持切换invoke/stream
        :param initial_state:  初始状态对象
        :param stream: 是否是流式输出
        :return: 执行完成后的状态对象
        """
        """"""
        if not self._compiled_app:
            self.compile()
        if stream:
            return self._compiled_app.stream(initial_state)
        else:
            return self._compiled_app.invoke(initial_state)

    @classmethod
    def create_and_run(cls, initial_state: ImportGraphState, stream: bool = False) -> ImportGraphState:
        """
        快捷方法：创建工作流实例并立即执行（兼容原有函数式调用习惯）
        :param initial_state: 初始状态对象
        :param stream: 是否是流式输出
        :return: 执行完成后的状态对象
        """
        workflow = cls()
        return workflow.run(initial_state, stream)

 # ===================== 用法示例 =====================

if __name__ == "__main__":

    # 定义初始状态
    init_state = create_default_state(
        task_id="task_demo",
        local_file_path="万用表的使用.pdf"
    )

    # 用法1：标准类用法（推荐，支持多实例）
    # 创建工作流实例
    kb_import_app = KBImportWorkflow()
    # 执行工作流
    final_state = kb_import_app.run(init_state)
    logger.info(f"工作流执行完成！最终状态: {final_state}")

    # 用法2：快捷调用
    for chunk in KBImportWorkflow.create_and_run(init_state, stream=True):
        # chunk：字典类型
        logger.info(chunk.keys())
        logger.info(chunk.items())
