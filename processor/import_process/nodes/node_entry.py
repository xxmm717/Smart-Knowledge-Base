import os
import sys
from os.path import splitext

from processor.common.logger import logger
from processor.import_process.core.exceptions import FileProcessingError
from processor.import_process.core.state import ImportGraphState, create_default_state
from processor.utils.core.task_utils import add_running_task
from processor.utils.format_utils import format_state


def node_entry(state:ImportGraphState)->ImportGraphState:
    """
    必要参数：
    - 必须包含 task_id(任务ID)
    - import_file_path(原始文件路径)
    - local_dir(输出文件的放置路径)
    更新参数：
    - is_pdf_read_enabled/is_md_read_enabled
    - pdf_path/md_path
    - file_title
    节点逻辑
    :param state: 工作流状态对象
    :return: 更新后的状态对象
    """
    # 动态获取函数名避免硬编码
    name = sys._getframe().f_code.co_name

    # 节点启动日志，打印当前工作流状态
    logger.debug(f"【{name}】节点启动，\n当前工作流状态：{format_state(state)}")

    # 开始：记录节点运行状态
    add_running_task(state["task_id"],name)

    # 1.获取文件路径与非空校验
    import_file_path = state.get("import_file_path","").strip()
    local_dir = state.get("local_dir","").strip()
    if not import_file_path:
        raise ValueError("缺失参数：import_file_path")
    if not local_dir:
        raise ValueError("缺失参数：local_dir")
    # 新增：检查文件是否存在
    if not os.path.isfile(import_file_path):
        logger.error(f"文件不存在：{import_file_path}")
        raise FileProcessingError(
            message=f"文件不存在：{import_file_path}",
            node_name=name
        )
    # 2.判断文件后缀
    if import_file_path.endswith(".pdf"):
        logger.info(f"文件类型校验通过：{import_file_path} ->PDF格式，开启PDF解析流程")
        state["is_pdf_read_enabled"] = True
        state["pdf_path"] = import_file_path
    elif import_file_path.endswith(".md"):
        logger.info(f"文件类型校验通过：{import_file_path} → MD格式，开启MD解析流程")
        state["is_md_read_enabled"] = True
        state["md_path"] = import_file_path
    else:
        logger.warning(f"文件类型校验失败：{import_file_path} → 不支持的格式，仅支持.pdf/.md")
    # 3.提取不含后缀名的文件名作为全局标识
    file_name = os.path.basename(import_file_path)
    state["file_title"] = splitext(file_name)[0]
    logger.info(f"文件业务标识提取完成：file_title = {state['file_title']}")
    return state


# if __name__ == "__main__":
#     setup_logging()
#
#     node_entry = NodeEntry()
#     node_state = create_default_state(
#         task_id="task_001",
#         import_file_path="d:/Code Demo/zhanggui_zhiku/doc/test.pdf",
#         local_dir="d:/Code Demo/zhanggui_zhiku/output"
#     )
#     # node_state_final = node_entry.process(node_state) #没有增强的版本
#     node_state_final = node_entry(node_state)  # 增强的版本

# if __name__ == '__main__':
#
#     from processor.import_process.state import create_default_state
#
#     # 配置日志（这样所有节点的日志都会显示）
#     setup_logging()
#
#     logger = logging.getLogger(__name__)
#
#
#     # 单元测试：覆盖不支持类型、MD、PDF三种场景
#     logger.info("===== 开始node_entry节点单元测试 =====")
#
#     # 测试1: 不支持的TXT文件
#     test_state1 = create_default_state(
#         task_id="test_entry_task_001",
#         import_file_path="联想海豚用户手册.txt",
#         local_dir="output",
#     )
#     node_entry1 = NodeEntry()
#     node_entry1(test_state1)
#
#     # 测试2: MD文件
#     test_state2 = create_default_state(
#         task_id="test_entry_task_002",
#         import_file_path="小米用户手册.md",
#         local_dir="output",
#     )
#     node_entry2 = NodeEntry()
#     node_entry2(test_state2)
#
#     # 测试3: PDF文件
#     test_state3 = create_default_state(
#         task_id="test_entry_task_003",
#         import_file_path="万用表的使用.pdf",
#         local_dir="output",
#     )
#     node_entry3 = NodeEntry()
#     node_entry3(test_state3)
#
#     logger.info("===== 结束node_entry节点单元测试 =====")
