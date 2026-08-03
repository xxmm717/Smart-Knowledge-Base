import base64
import os
import re
import sys
from collections import deque
from pathlib import Path
from typing import Tuple, Dict, List

from langchain_core.exceptions import LangChainException
from langchain_core.messages import HumanMessage
from minio import Minio
from minio.deleteobjects import DeleteObject
from pyexpat.errors import messages

from processor.common.logger import logger
from processor.config.llm_config import llm_config
from processor.config.minio_config import minio_config
from processor.import_process.core.state import ImportGraphState
from processor.utils.client.minio_client import minio_client
from processor.utils.core.task_utils import add_running_task
from processor.utils.format_utils import format_state
from processor.utils.lm import llm_client
from processor.utils.prompt.load_prompt import load_prompt
from processor.utils.rate_limit_utils import apply_api_rate_limit




def node_md_img(state:ImportGraphState)->ImportGraphState:
    """
       节点: 图片处理
       处理 Markdown 中的图片资源。
       """
    """
    MD文件图片处理核心节点
    核心流程：
    1. 获取MD内容、文件路径、图片文件夹路径
    2. 扫描图片文件夹，筛选MD中实际引用的支持格式图片
    3. 调用多模态大模型为图片生成内容摘要
    4. 将图片上传至MinIO，替换MD中本地图片路径为MinIO访问URL，并填充图片摘要
    5. 备份原MD文件，保存处理后的新MD文件并更新状态
    必要参数：task_id、md_path、md_content
    更新参数：md_path、md_content
    :param state: 工作流状态对象
    :return: 更新后的状态对象
    """
    # 动态获取函数名避免硬编码
    name = sys._getframe().f_code.co_name
    # 节点启动日志，打印当前工作流状态
    logger.debug(f"【{name}】节点启动，\n当前工作流状态：{format_state(state)}")
    # 开始：记录节点运行状态
    add_running_task(state["task_id"], name)
    # 1.初始化数据并校验
    md_content,md_path_obj,images_dir = _step_1_get_content(state)
    #无图片文件夹，直接跳过图片处理逻辑
    if not images_dir.exists():
        logger.info(f"图片文件夹不存在，跳过图片处理：{images_dir.absolute()}")
        return state
    # 2.扫描并筛选MD中引用的图片
    target_images = _step_2_get_scan_images(md_content,images_dir)
    if not target_images:
        logger.info("未检测到MD中引用的支持格式图片，跳过后续处理")
    # 3.调用大模型生成图片摘要
    summaries = _step_3_generate_summaries(md_path_obj.stem,target_images)
    # 4.上传图片至MinIO，替换md图片路径并填充摘要
    new_md_content = _step_4_upload_and_replace(minio_client,md_path_obj.stem,target_images,summaries,md_content)
    # 5.备份并保存新md文件
    new_md_file_name = _step_5_backup_new_md_file(state['md_path'], new_md_content)
    # 6.更新state状态值
    state["md_path"] = new_md_file_name
    state["md_content"] = new_md_content
    return state
def _step_1_get_content(state:ImportGraphState)->Tuple[str,Path,Path]:
    """
    从全局状态中提取并初始化MD处理所需核心数据
    :param state: 流程全局状态对象
    :return: 三元组(MD文件内容, MD文件路径, 图片文件夹路径)
    :raise FileNotFoundError: 当状态中无有效MD文件路径时抛出
    """
    # 1.非空校验
    md_path = state.get("md_path","")
    if not md_path:
        raise ValueError("缺失参数md_path")
    # 2.文件路径转换
    md_path_obj = Path(md_path)
    # 3.校验文件是否存在
    if not md_path_obj.exists():
        raise ValueError(f"MD文件不存在，绝对路径: {md_path_obj.absolute()}")
    # 4.获取md文件内容，优先获取已经保存的内容，没有则从文件读取
    if not state["md_content"]:
        with open(md_path_obj,"r",encoding="utf-8") as f:
            md_content = f.read()
        logger.info(f"从文件读取md内容完成：文件大小：{len(md_content)}")
    else:
        md_content = state["md_content"]
        logger.info(f"从全局状态获取MD内容完成，内容大小：{len(md_content)} 字符")
    # 5.组装图片文件夹路径
    images_dir = md_path_obj.parent / "images"
    return md_content,md_path_obj,images_dir
def _step_2_get_scan_images(md_content:str, images_dir:Path)-> list[Tuple[str,str,Tuple[str,str]]]:
    """
    扫描图片文件夹，过滤出「支持格式+MD中实际引用」的图片，组装处理元数据
    :param md_content: MD文件完整内容
    :param images_dir: 图片文件夹路径对象
    :return: 待处理图片列表，每个元素为(图片文件名, 图片完整路径, 图片上下文)元组
    """
    #MinIO支持的图片格式合集
    images_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    target_images = []
    # 遍历图片文件夹
    for image_file in os.listdir(images_dir):
        # 1.过滤无效后缀
        file_extension = os.path.splitext(image_file)[1].lower()
        if file_extension not in images_extensions:
            logger.warning(f"图片格式不支持，跳过：{image_file}")
            continue
        # 2.组装图片完整路径并转换为字符串
        image_path = str(images_dir / image_file)
        # 3.查找图片在md文件中的引用上下文
        context = _find_image_in_md(md_content,image_file)
        # 过滤MD中未引用的图片
        if not context:
            logger.info(f"图片未在MD中引用，跳过处理：{image_file}")
            continue
        # 4.组装处理图片元数据，取第一个匹配的图片上下文
        target_images.append((image_file,image_path,context))
        logger.info(f"图片加入待处理列表：{image_file}")
    logger.info(f"图片扫描完成，共筛选出待处理图片：{len(target_images)} 张")
    return target_images
def _find_image_in_md(md_content: str, image_file: str,context_len: int = 100)->Tuple[str,str]:
    """
    查找MD内容中指定图片的所有引用位置，并返回每个位置的上下文文本
    :param md_content: MD文件完整内容
    :param image_file: 图片文件名（含后缀）
    :param context_len: 上下文截取长度，默认前后各100字符
    :return: 每个图片的(上文, 下文)元组，无匹配则返回None
    """
    # 1.定义正则表达式
    # ![描述](images/文件名.扩展名)
    # r"字符串"：不要将其中的特殊符号进行转义
    # re.escape 转义图片文件名中的特殊字符，避免正则语法错误
    # .* 贪婪匹配 .*? 非贪婪匹配
    pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
    # 2.找到1个匹配项即返回
    match = pattern.search(md_content)
    if not match:
        return None
    # 3.截取匹配位置的上下文（防止索引越界）
    start,end = match.span()
    pre_text = md_content[max(0,start-context_len):start]
    post_text = md_content[end:min(len(md_content),end+context_len)]
    # 打印图片上下文，便于调试
    logger.debug(f"图片[{image_file}]匹配到引用，上文：{pre_text.strip()}")
    logger.debug(f"图片[{image_file}]匹配到引用，下文：{post_text.strip()}")
    # 4、返回上下文元组
    return  pre_text,post_text
def _step_3_generate_summaries(doc_stem: str, target_images: list[Tuple[str,str,Tuple[str,str]]])->Dict[str,str]:
    """
    步骤3：批量为待处理图片生成内容摘要，带API速率限制防止触发大模型限流
    :param doc_stem: 文档文件名（不含后缀），作为大模型prompt上下文
    :param targets: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
    :param requests_per_minute: 每分钟最大API请求数，默认9次（按大模型限制调整）
    :return: 图片摘要字典，键：图片文件名，值：图片内容摘要
    """
    summaries = {}
    # 1.外部初始化双端队列，用于API速率限制，跨循环复用
    requests_deque = deque()
    # 2.循环处理图片
    for image_file,image_path,context in target_images:
        # 2.1速率限制
        apply_api_rate_limit(requests_deque,max_requests=10,window_seconds=60)
        # 2.2调用大模型生成图片摘要
        logger.info(f"开始生成图片摘要：{image_path}")
        summaries[image_file] = _summarize_image(image_path, root_folder=doc_stem, image_content=context)
    logger.info(f"图片摘要批量生成完成，共处理{len(summaries)}张图片")
    return summaries
    return
def _step_4_upload_and_replace(minio_client: Minio, doc_stem: str, targets: List[Tuple[str, str, Tuple[str, str]]],
                          summaries: Dict[str, str], md_content: str)->str:
    """
    步骤4：核心流程-图片上传MinIO + 合并摘要&URL + 替换MD图片引用
    完整流程：清理MinIO旧目录 → 批量上传新图片 → 合并摘要和URL → 替换MD内容
    :param minio_client: 初始化完成的MinIO客户端对象
    :param doc_stem: 文档文件名（不含后缀），作为MinIO上传子目录名（按文档隔离）
    :param targets: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
    :param summaries: 图片摘要字典，键：图片文件名，值：内容摘要
    :param md_content: 原始MD文件内容
    :return: 图片引用替换后的新MD内容
    """
    # 构造MinIO上传目录：配置根目录+文档主名
    minio_image_dir = minio_config.minio_img_dir
    upload_dir = f"{minio_image_dir}/{doc_stem}".replace(" ","")
    # 1.清理该文档对应的Minio旧目录
    _clean_minio_directory(minio_client, upload_dir)
    # 2.批量上传图片至minio，获取url映射
    urls = _upload_images_batch(minio_client, upload_dir, targets)
    # 3.合并图片摘要和URL，过滤上传失败的图片
    image_info = _merge_summary_and_url(summaries, urls)
    # 4.替换md内容中的本地图片引用为minio远程引用
    if image_info:
        md_content = _process_md_file(md_content, image_info)
    return md_content
def _step_5_backup_new_md_file(origin_md_path: str, md_content: str) -> str:
    """
    步骤5：将处理后的MD内容保存为新文件（原文件不变，避免数据丢失）
    新文件命名规则：原文件名 + _new.md（如test.md → test_new.md）
    :param origin_md_path: 原始MD文件完整路径
    :param md_content: 处理后的新MD内容
    :return: 新MD文件的完整路径
    """
    # 构造新文件路径：替换原后缀为 _new.md
    new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"
    # 写入新MD内容（覆盖写入，若文件已存在则更新）
    with open(new_md_file_name, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"处理后MD文件已保存，新文件路径：{new_md_file_name}")
    return new_md_file_name
def _summarize_image(image_path: str, root_folder: str, image_content: Tuple[str, str]) -> str:
    """
    调用多模态大模型生成图片内容摘要（适配LangChain工具类，复用项目统一LLM客户端）
    生成的摘要用于Markdown图片标题，严格控制50字以内中文描述
    :param image_path: 图片本地完整路径
    :param root_folder: 文档所属文件夹/主名，为大模型提供上下文
    :param image_content: 图片在MD中的上下文元组，格式(上文文本, 下文文本)
    :return: 图片内容摘要（异常时返回默认值"图片描述"）
    """

    # 将图片编码为Base64，适配多模态大模型输入要求
    base64_image = _encode_image_to_base64(image_path)
    try:
        # 1.获取项目统一LLM客户端（自动缓存，传入多模态模型名）
        lv_client = llm_client.get_llm_client(model=llm_config.lv_model)
        # 2.提示词工程
        # 加载并渲染提示词模版
        prompts_text = load_prompt(
            name="image_summary",
            root_folder=root_folder,
            image_content=image_content
        )
        # 构造LangChain标准多模态HumanMessage
        messages = [
            HumanMessage(
                content=[
                    # 文本提示词：携带上下文，限定摘要规则
                    {
                        "type": "text",
                        "text": prompts_text
                    },
                    # 多模态核心：Base64编码图片数据
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            )
        ]
        # 3.LangChain调用：invoke方法（工具封装超时、重试等参数）
        response = lv_client.invoke(input=messages)
        # 4.解析响应（LangChain统一返回content字段，统一格式无需多层解析）
        summary = response.content.strip().replace("\n","")
        logger.info(f"图片摘要生成成功：{image_path}，摘要：{summary}")
        return summary
    except LangChainException as e:
        logger.error(f"图片摘要生成失败（LangChain框架异常）：{image_path}，错误信息：{str(e)}")
        return "图片描述"
    except Exception as e:
        logger.error(f"图片摘要生成失败（系统异常）：{image_path}，错误信息：{str(e)}")
        return "图片描述"
def _encode_image_to_base64(image_path: str)->str:
    """
    将本地图片文件编码为Base64字符串（用于多模态大模型输入）
    :param image_path: 图片本地完整路径
    :return: 图片的Base64编码字符串（UTF-8解码）
    """
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode("utf-8")
    logger.debug(f"图片Base64编码完成，文件：{image_path}，编码后长度：{len(base64_str)}")
    return base64_str
def _clean_minio_directory(minio_client: Minio, upload_dir_prefix)->None:
    """
    幂等性清理MinIO指定目录下的所有旧文件，防止重名文件内容混淆和垃圾文件堆积
    幂等性：多次调用结果一致，无文件时不报错
    :param minio_client: 初始化完成的MinIO客户端对象
    :param prefix: MinIO目录前缀（要清理的目录路径）
    """
    try:
        # 列出指定前缀下的所有对象（递归遍历子目录）
        object_to_delete = minio_client.list_objects(
            bucket_name=minio_config.bucket_name,
            prefix=upload_dir_prefix,
            recursive=True,
        )
        # 构造删除对象列表
        delete_list = [DeleteObject(obj.object_name) for obj in object_to_delete]
        if delete_list:
            logger.info(f"开始清理MinIO旧文件，待删除文件数：{len(delete_list)}，目录：{upload_dir_prefix}")
            # 批量删除对象
            errors = minio_client.remove_objects(minio_config.bucket_name,delete_list)
            # 遍历删除错误信息，记录异常
            for error in errors:
                logger.error(f"MinIO文件删除失败：{error}")
        else:
            logger.debug(f"MinIO目录无旧文件，无需清理：{upload_dir_prefix}")
    except Exception as e:
        logger.error(f"MinIO目录清理失败：{upload_dir_prefix}，错误信息：{str(e)}")
def _upload_images_batch(minio_client:Minio, upload_dir:str, targets: List[Tuple[str, str, Tuple[str, str]]])->Dict[str, str]:
    """
    批量上传待处理图片至MinIO，返回图片文件名与访问URL的映射关系
    :param minio_client: 初始化完成的MinIO客户端对象
    :param upload_dir: MinIO上传根目录
    :param targets: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
    :return: 图片URL字典，键：图片文件名，值：MinIO访问URL
    """
    urls = {}
    for img_file, img_path, _ in targets:
        # 构造MinIO对象名称
        object_name = f"{upload_dir}/{img_file}"
        logger.debug(f"构造MinIO对象名称完成：{object_name}")
        # 上传单张图片并获取URL
        """
        := 是 Python 3.8+ 引入的海象运算符（Walrus Operator），核心作用是 **「表达式内赋值 + 结果判断」一体化 **：
        在执行判断、循环等逻辑的同一个表达式中，完成变量赋值和赋值结果的使用 / 判断，替代传统「先赋值、后判断」的两行代码，让逻辑更简洁。
        """
        if img_url := _upload_to_minio(minio_client, img_path, object_name):
            urls[img_file] = img_url
    logger.info(f"图片批量上传完成，成功上传{len(urls)}/{len(targets)}张图片")
    return urls
def _upload_to_minio(minio_client: Minio, local_path: str, object_name: str) -> str | None:
    """
    将单张本地图片上传至MinIO对象存储，并返回公网可访问URL
    :param minio_client: 初始化完成的MinIO客户端对象
    :param local_path: 图片本地完整路径
    :param object_name: MinIO中要存储的对象名称（带目录）
    :return: 图片MinIO访问URL（上传失败返回None）
    """
    try:
        logger.info(f"开始上传图片至MinIO：本地路径={local_path}，MinIO对象名={object_name}")
        # 上传本地文件至MinIO（fput_object：文件流上传，适合大文件）
        minio_client.fput_object(
            bucket_name=minio_config.bucket_name,  # MinIO存储桶名（从配置读取）
            object_name=object_name,  # MinIO对象名称
            file_path=local_path,  # 本地文件路径
            # 自动推断图片Content-Type（如image/png、image/jpeg）
            # 入参：文件路径字符串（可带目录，如/a/b/test.jpg、demo.tar.gz）；
            # 返回值：元组(root, ext)，其中：
            # root：文件主名（含目录，去掉最后一个后缀的完整部分）；
            # ext：文件后缀（以.开头，仅包含最后一个扩展名，如.jpg、.gz，无后缀则为空字符串""）；
            # 关键规则：仅识别 ** 最后一个.** 作为后缀分隔符，多后缀文件仅拆分最后一个（如test.tar.gz拆分为("test.tar", ".gz")）。
            content_type=f"image/{os.path.splitext(local_path)[1][1:]}"
        )
        # 处理路径特殊字符，避免URL解析错误
        object_name = object_name.replace("\\", "%5C")
        # 根据配置选择HTTP/HTTPS协议
        protocol = "https" if minio_config.minio_secure else "http"
        # 构造MinIO基础访问URL
        base_url = f"{protocol}://{minio_config.endpoint}/{minio_config.bucket_name}"
        # 拼接完整图片访问URL base_url 后面带 / 中间直接两个字符串拼接即可
        img_url = f"{base_url}{object_name}"
        logger.info(f"图片上传成功，访问URL：{img_url}")
        return img_url
    except Exception as e:
        logger.error(f"图片上传MinIO失败：{local_path}，错误信息：{str(e)}")
        return None
def _merge_summary_and_url(summaries: Dict[str, str], urls: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    """
    合并图片摘要字典和URL字典，过滤掉上传失败无URL的图片
    :param summaries: 图片摘要字典，键：图片文件名，值：内容摘要
    :param urls: 图片URL字典，键：图片文件名，值：MinIO访问URL
    :return: 合并后的图片信息字典，键：图片文件名，值：(摘要, URL)元组
    """
    image_info = {}
    # 遍历摘要字典，仅保留有对应URL的图片
    for image_file, summary in summaries.items():
        if url := urls.get(image_file):
            image_info[image_file] = (summary, url)
    logger.info(f"图片摘要与URL合并完成，有效图片信息{len(image_info)}条")
    return image_info
def _process_md_file(md_content: str, image_info: Dict[str, Tuple[str, str]]) -> str:
    """
    核心功能：替换MD内容中的本地图片引用为MinIO远程引用
    替换规则：![原描述](本地路径) → ![图片摘要](MinIO访问URL)
    :param md_content: 原始MD文件内容
    :param image_info: 合并后的图片信息字典，键：图片文件名，值：(摘要, URL)
    :return: 替换后的新MD内容
    """
    for img_filename, (summary, new_url) in image_info.items():
        # 正则匹配MD图片标签，忽略大小写，兼容不同路径写法
        # 正则规则：![任意描述](任意路径+图片文件名+任意后缀)
        pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(img_filename) + r".*?\)",
            re.IGNORECASE
        )
        # 替换匹配内容：使用新摘要作为图片描述，新URL作为图片路径
        # - 如果你的 summary 和 new_url 是完全可控的纯文本（不含反斜杠） ：这两种写法确实 一模一样 。
        # - 如果你想写出“防御性代码”（Defensive Code），防止未来某天被特殊字符坑 ：请坚持使用 Lambda 写法 。它是最稳健、最安全的做法。
        # md_content = pattern.sub(lambda m: f"![{summary}]({new_url})", md_content)
        md_content = pattern.sub(f"![{summary}]({new_url})", md_content)
        logger.debug(f"完成MD图片引用替换：{img_filename} → {new_url}")
    logger.info(f"MD文件图片引用替换完成，共替换{len(image_info)}处图片引用")
    logger.debug(f"替换后MD内容：{md_content[:500]}..." if len(md_content) > 500 else f"替换后MD内容：{md_content}")
    return md_content


# if __name__ == "__main__":
#     import logging
#     from processor.import_process.core.state import create_default_state
#     from processor.import_process.core.base import setup_logging
#
#     setup_logging(logging.INFO)
#
#     state = create_default_state(
#         task_id="test_img_001",
#         md_path="D:/Code Demo/zhanggui_zhiku/output/test/test.md",
#     )
#
#     node = NodeMdImg()
#
#     # === 1. 获取内容 ===
#     print("=" * 60)
#     print("1. _step_1_get_content")
#     print("=" * 60)
#     md_content, md_path_obj, images_dir = node._step_1_get_content(state)
#     print(f"   MD 内容: {len(md_content)} 字符")
#     print(f"   MD 路径: {md_path_obj}")
#     print(f"   图片目录: {images_dir}")
#     print(f"   目录存在: {images_dir.exists()}")
#
#     if not images_dir.exists():
#         print("   无图片目录，跳过")
#         exit(0)
#
#     print(f"\n{"=" * 60}")
#     print("2. 图片文件列表")
#     print("=" * 60)
#     for f in sorted(images_dir.iterdir()):
#         print(f"   {f.name}  ({f.stat().st_size} bytes)")
#
#     print(f"\n{"=" * 60}")
#     print("3. _find_image_in_md — 图片上下文 (pre_text / post_text)")
#     print("=" * 60)
#
#     for image_file in sorted(os.listdir(images_dir)):
#         ext = os.path.splitext(image_file)[1].lower()
#         if ext not in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}:
#             continue
#
#         context = node._find_image_in_md(md_content, image_file)
#         if context is None:
#             print(f"\n   [{image_file}]  未在 MD 中引用")
#             continue
#
#         pre_text, post_text = context
#         print(f"\n   >>> {image_file}")
#         print(f"   [pre_text(前100字)]: {pre_text.strip()}")
#         print(f"   [post_text(后100字)]: {post_text.strip()}")
#
#     print(f"\n{"=" * 60}")
#     print("4. _step_2_get_scan_images — 整体扫描结果")
#     print("=" * 60)
#     result = node._step_2_get_scan_images(md_content, images_dir)
#     print(f"   返回值: {result}")
