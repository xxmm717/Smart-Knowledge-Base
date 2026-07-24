from processor.import_process.core.base import BaseNode
from processor.import_process.core.state import ImportGraphState


class NodeItemNameRecognition(BaseNode):
    """
    节点: 主体识别
    识别文档核心描述的物品/商品名称
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_item_name_recognition"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # TODO
        logger.info(f"【{self.name}】节点逻辑")

        return state