from processor.import_process.core.base import BaseNode
from processor.import_process.core.state import ImportGraphState


class NodeBgeEmbedding(BaseNode):
    """
    节点: 向量化
    使用 BGE-M3 模型将文本转换为向量。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_bge_embedding"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # TODO
        logger.info(f"【{self.name}】节点逻辑")

        return state