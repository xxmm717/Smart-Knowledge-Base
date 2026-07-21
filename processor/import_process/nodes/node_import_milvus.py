from processor.import_process.base import BaseNode
from processor.import_process.state import ImportGraphState


class NodeImportMilvus(BaseNode):
    """
    节点: 导入向量库
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_import_milvus"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # TODO
        logger.info(f"【{self.name}】节点逻辑")

        return state