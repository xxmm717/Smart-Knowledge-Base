from processor.import_process.base import BaseNode
from processor.import_process.state import ImportGraphState


class NodeDocumentSplit(BaseNode):
    """
    节点: 文档切分
    将长文档切分成小的 Chunks (切片) 以便检索。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_document_split"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # TODO
        self.logger.info(f"【{self.name}】节点逻辑")

        return state