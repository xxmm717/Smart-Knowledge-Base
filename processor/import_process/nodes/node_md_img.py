from processor.import_process.base import BaseNode
from processor.import_process.state import ImportGraphState


class NodeMdImg(BaseNode):

    def process(self,state:ImportGraphState)->ImportGraphState:
        """
           节点: 图片处理
           处理 Markdown 中的图片资源。
           """

        # 覆盖基类的 name 属性，标识节点名称
        name:str = "node_md_img"
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # TODO
        self.logger.info(f"【{self.name}】节点逻辑")

        return state