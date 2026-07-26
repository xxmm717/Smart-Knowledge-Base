import shutil
import time
import zipfile
from pathlib import Path

import requests
from requests.exceptions import RequestException

from processor.config.config import get_config
from processor.import_process.core.base import BaseNode
from processor.import_process.core.state import ImportGraphState


class NodePdfToMd(BaseNode):
    """
    节点: PDF转Markdown (node_pdf_to_md)
    核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_pdf_to_md"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
         必要参数：task_id、pdf_path、local_dir
         更新参数：md_path、md_content
         :param state: 工作流状态对象
         :return: 更新后的状态对象
         """
        # 1.校验pdf路径和输出目录
        pdf_path_ojb,output_dir_ojb = self._step_1_validate_path(state)

        # 2.上传pdf到MinerU并轮询解析结果
        zip_url = self._step_2_upload_and_poll(pdf_path_ojb)

        # 3.下载zip并提取md文件
        md_path = self._step_3_download_and_extract(zip_url,output_dir_ojb,pdf_path_ojb.stem)

        # 4.读取md内容
        # Markdown是文本文件，使用文本模式读取，保证传给后续节点的是str而不是bytes。
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # 5.更新state状态
        state["md_path"] = str(md_path)
        state["md_content"] = md_content

        return state

    # 校验pdf路径和输出目录
    def _step_1_validate_path(self, state):
        """
        步骤1：校验PDF文件路径和输出目录
        核心职责：参数非空校验 | 路径转换 | PDF文件有效性校验 | 输出目录自动创建
        返回：合法的PDF文件Path对象、输出目录Path对象
        异常：ValueError(参数缺失)、FileNotFoundError(文件无效)
        """
        # 1.参数非空校验
        pdf_path = state.get("pdf_path","").strip()
        file_path = state.get("file_path","").strip()
        if not pdf_path:
            raise ValueError("缺失参数：pdf_path")
        if not file_path:
            raise ValueError("缺失参数：file_path")

        # 2.转换为path对象统一处理路径
        pdf_path_ojb = Path(pdf_path)
        file_path_ojb = Path(file_path)

        # 3.PDF文件有效性校验
        if not pdf_path_ojb.exists():
            raise FileNotFoundError(f"PDF文件不存在，绝对路径：{pdf_path_ojb.absolute()}")

        # 4.确保输出目录存在，不存在则递归创建
        if not file_path_ojb.exists():
            self.logger.info(f"输出目录不存在，自动创建{file_path_ojb.absolute()}")
            file_path_ojb.mkdir(parents=True, exist_ok=True)

        return pdf_path_ojb, file_path_ojb

    # 上传pdf到MinerU并轮询解析结果
    def _step_2_upload_and_poll(self,pdf_path_obj: Path):
        """
        步骤2：上传PDF至MinerU并轮询解析任务状态
        核心流程：配置校验 → 获取上传链接 → 文件上传 → 任务轮询（直至完成/失败/超时）
        参数：pdf_path_obj-已校验的PDF Path对象
        返回：解析结果ZIP包下载链接full_zip_url
        异常：ValueError(配置缺失)、RuntimeError(请求/上传失败)、TimeoutError(任务超时)
        """
        # 1.校验参数
        cfg = get_config()
        if not cfg.minerU_base_url or not cfg.minerU_api_token:
            raise ValueError("请在 .env 文件中正确配置 MINERU_API_TOKEN 和 MINERU_BASE_URL 参数")
        self.logger.info(f"【配置校验】MinerU配置校验成功，开始处理文件：{pdf_path_obj.name}")

        # 2.从MinerU获取上传链接
        token = cfg.minerU_api_token
        url = f"{cfg.minerU_base_url}/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }
        self.logger.info(f"【获取上传链接】调用接口：{url}，请求参数：{data}")


        response = requests.post(url, headers=header, json=data)
        if response.status_code != 200:
            raise RuntimeError()
        else:
            result = response.json()
            if result["code"] == 0:
                singed_url  = result["data"]["file_urls"][0]
                batch_id = result["data"]["batch_id"]
                self.logger.info(f"【获取上传链接】成功：上传链接已生成，batch_id：{batch_id}")
            else:
                raise RuntimeError(f"【获取上传链接】接口调用业务失败：返回数据：{result}")

        # 3.文件上传
        self.logger.info(f"【文件上传】开始上传PDF文件：{pdf_path_obj.name}")
        with open(pdf_path_obj, 'rb') as f:
            res_upload = requests.put(singed_url, data=f)
            if res_upload.status_code == 200:
                print(f"{singed_url} 文件上传成功")
            else:
                raise RuntimeError(f"【文件上传】上传失败：状态码：{res_upload.status_code}，响应结果：{res_upload}")

        # 4.轮询解析结果
        poll_url = f"{cfg.minerU_base_url}/extract-results/batch/{batch_id}"

        start_time = time.time() #记录开始时间
        timeout_seconds = 600 #最大超时时间
        poll_interval = 3 #轮询间隔时间
        self.logger.info(f"【任务轮询】开始轮询解析结果，最大超时：{timeout_seconds}s，batch_id：{batch_id}")

        # 根据batch_id轮询任务状态直到成功
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise TimeoutError(f"【任务轮询】超时！任务处理超{timeout_seconds}秒，batch_id：{batch_id}")

            # 发起轮询请求，短超时10秒，异常则重试
            try:
                res_poll = requests.get(poll_url, headers=header,timeout=10)
            except Exception as e:
                self.logger.warning(f"【任务轮询】网络请求异常，{poll_interval}秒后重试：{str(e)}")
                time.sleep(poll_interval)
                continue

            # 处理HTTP响应错误
            if res_poll.status_code != 200:
                raise RuntimeError(f"【任务轮询】HTTP请求失败，状态码：{res_poll.status_code}，响应内容：{res_poll}")

            # 解析轮询结果，校验业务状态
            poll_data = res_poll.json()
            if poll_data["code"] != 0:
                raise RuntimeError(f"【任务轮询】业务错误，返回数据：{poll_data}")
            extract_result = poll_data["data"]["extract_result"]

            #获取结果
            result_item = extract_result[0]
            data_state = result_item["state"]

            # 状态为 done
            if data_state ==  "done":
                self.logger.info(f"【任务轮询】解析任务完成！总耗时{int(elapsed_time)}s，batch_id：{batch_id}")

                full_zip_url = result_item["full_zip_url"]
                self.logger.info(f"【任务轮询】返回ZIP包下载链接：{full_zip_url}")

                return full_zip_url
            elif data_state == "failed":
                err_msg = result_item.get("err_msg", "未知错误，无具体信息")
                raise RuntimeError(f"【任务轮询】解析任务失败！batch_id：{batch_id}，错误信息：{err_msg}")
            else:
                self.logger.info(f"【任务轮询】处理中... 已耗时{int(elapsed_time)}s，状态：{data_state}， batch_id：{batch_id}")
                time.sleep(poll_interval)

    # 下载与解压 (Download & Extract)
    def _step_3_download_and_extract(self,zip_url:str,output_dir_obj:Path,pdf_stem:str)-> str:
        """
       步骤3：下载MinerU解析结果ZIP包并解压，提取目标MD文件
       核心流程：下载ZIP → 清理旧目录并解压 → 查找MD文件 → 重命名统一为PDF同名
       参数：zip_url-ZIP包下载链接；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
       返回：最终MD文件的字符串格式绝对路径
       异常：RuntimeError(下载失败)、FileNotFoundError(无MD文件)
       """
        # 1.下载ZIP包并校验。使用流式下载和重试，避免代理连接中断导致response.content读取不完整。
        zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"
        self.logger.info(f"【ZIP下载】开始下载ZIP包：{zip_url} ...")
        download_error = None
        for attempt in range(1, 4):
            try:
                response = requests.get(
                    zip_url,
                    stream=True,
                    timeout=(20, 300),
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"HTTP状态码：{response.status_code}，响应结果：{response}"
                    )

                with open(zip_save_path, "wb") as f:
                    for data in response.iter_content(chunk_size=1024 * 1024):
                        if data:
                            f.write(data)
                response.close()

                # 下载完成后先验证ZIP结构，避免把不完整文件交给解压步骤。
                if not zipfile.is_zipfile(zip_save_path):
                    raise RuntimeError("下载文件不是有效的ZIP文件，可能是连接中断")

                self.logger.info(
                    f"【ZIP下载】ZIP包下载成功：保存路径：{zip_save_path}，尝试次数：{attempt}"
                )
                break
            except (RequestException, RuntimeError, zipfile.BadZipFile) as e:
                download_error = e
                if zip_save_path.exists():
                    zip_save_path.unlink()
                if attempt == 3:
                    raise RuntimeError(
                        f"【ZIP下载】重试3次后仍失败：{download_error}"
                    ) from download_error
                self.logger.warning(
                    f"【ZIP下载】第{attempt}次下载失败，2秒后重试：{download_error}"
                )
                time.sleep(2)

        # 2.清空解压目录
        extract_target_dir = output_dir_obj / pdf_stem
        shutil.rmtree(extract_target_dir, ignore_errors=True)
        self.logger.info(f"【ZIP解压】已清空旧的解压目录：{extract_target_dir}")

        # 3.创建解压目录
        extract_target_dir.mkdir(parents=True, exist_ok=True)

        # 4.解压
        self.logger.info(f"【ZIP解压】开始解压ZIP包：{output_dir_obj} ...")
        with zipfile.ZipFile(zip_save_path, "r") as zip_ref:
            zip_ref.extractall(extract_target_dir)
        self.logger.info(f"【ZIP解压】ZIP解压完成，解压目录：{extract_target_dir}")

        # 5.重命名
        self.logger.info(f"【MD重命名】找到MinerU生成的full.md文件")
        target_md_file = extract_target_dir / "full.md"
        self.logger.info(f"【MD重命名】开始将full.md文件进行重命名")
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
        target_md_file.rename(new_md_path)
        self.logger.info(f"【MD重命名】重命名成功，文件名：{pdf_stem}.md")

        return str(new_md_path.absolute())
