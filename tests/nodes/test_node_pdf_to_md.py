"""
node_pdf_to_md 节点单元测试

测试策略：
  _step_1_validate_path  真实文件系统 + 边界条件（纯单元级）
  _step_2_upload_and_poll mock requests，模拟 MinerU API 完整交互
  _step_3_download_and_extract mock 下载，真实 zip 解压 + 重命名
  process()               mock 内部步骤，验证 state 字段更新
"""

import io
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest

from processor.import_process.core.state import create_default_state


def make_mock_zip(md_content: str = "# Mock\n\nHello World") -> bytes:
    """生成一个内含 full.md 文件的 ZIP 字节数据"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("full.md", md_content)
    return buf.getvalue()


class TestStep1ValidatePath:
    """测试 _step_1_validate_path 的路径校验逻辑"""

    def test_missing_pdf_path_raises_value_error(self, node, tmp_path):
        """缺少 pdf_path -> ValueError"""
        state = create_default_state()
        state["file_path"] = str(tmp_path)
        with pytest.raises(ValueError, match="pdf_path"):
            node._step_1_validate_path(state)

    def test_missing_file_path_raises_value_error(self, node, real_pdf):
        """缺少 file_path -> ValueError"""
        state = create_default_state(pdf_path=str(real_pdf))
        state["file_path"] = ""
        with pytest.raises(ValueError, match="file_path"):
            node._step_1_validate_path(state)

    def test_pdf_not_exist_raises_file_not_found(self, node, tmp_path):
        """PDF 文件不存在 -> FileNotFoundError"""
        state = create_default_state()
        state["pdf_path"] = str(tmp_path / "non_existent.pdf")
        state["file_path"] = str(tmp_path)
        with pytest.raises(FileNotFoundError):
            node._step_1_validate_path(state)

    def test_output_dir_auto_created(self, node, real_pdf, tmp_path):
        """输出目录不存在时自动递归创建"""
        output_dir = tmp_path / "nested" / "output"
        assert not output_dir.exists()
        state = create_default_state(pdf_path=str(real_pdf))
        state["file_path"] = str(output_dir)
        _, dir_obj = node._step_1_validate_path(state)
        assert dir_obj.exists()
        assert dir_obj.is_dir()

    def test_valid_paths_returns_paths(self, node, real_pdf, tmp_path):
        """正常路径 -> 返回 (Path, Path)"""
        state = create_default_state(pdf_path=str(real_pdf))
        state["file_path"] = str(tmp_path)
        pdf_obj, dir_obj = node._step_1_validate_path(state)
        assert isinstance(pdf_obj, Path)
        assert isinstance(dir_obj, Path)
        assert pdf_obj.exists()
        assert dir_obj.exists()
        assert pdf_obj.name == real_pdf.name


class TestStep2UploadAndPoll:
    """测试 _step_2_upload_and_poll（完全 mock requests 层）"""

    MOCK_URL = "https://fake-mineru.net/api/v4/extract/task"
    MOCK_TOKEN = "test_token"

    @patch("processor.import_process.nodes.node_pdf_to_md.requests")
    def test_missing_config_raises_value_error(self, mock_req, node, real_pdf):
        """MINERU_BASE_URL / API_TOKEN 为空 -> ValueError"""
        with patch("processor.import_process.nodes.node_pdf_to_md.get_config") as mock_cfg:
            mock_cfg.return_value.minerU_base_url = ""
            mock_cfg.return_value.minerU_api_token = ""
            with pytest.raises(ValueError, match="MINERU_API_TOKEN|MINERU_BASE_URL"):
                node._step_2_upload_and_poll(real_pdf)

    @patch("processor.import_process.nodes.node_pdf_to_md.requests")
    def test_upload_api_http_error(self, mock_req, node, real_pdf):
        """获取上传链接时返回 500 -> RuntimeError"""
        mock_req.post.return_value = MagicMock(status_code=500, text="Internal Server Error")
        with patch("processor.import_process.nodes.node_pdf_to_md.get_config") as mock_cfg:
            mock_cfg.return_value.minerU_base_url = self.MOCK_URL
            mock_cfg.return_value.minerU_api_token = self.MOCK_TOKEN
            with pytest.raises(RuntimeError):
                node._step_2_upload_and_poll(real_pdf)

    @patch("processor.import_process.nodes.node_pdf_to_md.requests")
    def test_upload_api_business_error(self, mock_req, node, real_pdf):
        """获取上传链接返回业务错误 (code != 0) -> RuntimeError"""
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"code": 1001, "message": "invalid params"}
        mock_req.post.return_value = resp
        with patch("processor.import_process.nodes.node_pdf_to_md.get_config") as mock_cfg:
            mock_cfg.return_value.minerU_base_url = self.MOCK_URL
            mock_cfg.return_value.minerU_api_token = self.MOCK_TOKEN
            with pytest.raises(RuntimeError, match="1001|invalid"):
                node._step_2_upload_and_poll(real_pdf)

    @patch("processor.import_process.nodes.node_pdf_to_md.requests")
    def test_upload_file_http_error(self, mock_req, node, real_pdf):
        """文件上传（PUT）失败 -> RuntimeError"""
        post_resp = MagicMock(status_code=200)
        post_resp.json.return_value = {
            "code": 0,
            "data": {"file_urls": ["https://upload.url/test.pdf"], "batch_id": "b001"},
        }
        mock_req.post.return_value = post_resp
        mock_req.put.return_value = MagicMock(status_code=403, text="Forbidden")
        with patch("processor.import_process.nodes.node_pdf_to_md.get_config") as mock_cfg:
            mock_cfg.return_value.minerU_base_url = self.MOCK_URL
            mock_cfg.return_value.minerU_api_token = self.MOCK_TOKEN
            with pytest.raises(RuntimeError, match="上传失败|403"):
                node._step_2_upload_and_poll(real_pdf)

    @patch("processor.import_process.nodes.node_pdf_to_md.requests")
    def test_poll_result_failed_state(self, mock_req, node, real_pdf):
        """轮询返回 state=failed -> RuntimeError"""
        post_resp = MagicMock(status_code=200)
        post_resp.json.return_value = {
            "code": 0,
            "data": {"file_urls": ["https://upload.url/test.pdf"], "batch_id": "b001"},
        }
        mock_req.post.return_value = post_resp
        mock_req.put.return_value = MagicMock(status_code=200)
        poll_resp = MagicMock(status_code=200)
        poll_resp.json.return_value = {
            "code": 0,
            "data": {
                "extract_result": [{
                    "state": "failed",
                    "err_msg": "pdf parse error",
                }]
            },
        }
        mock_req.get.return_value = poll_resp
        with patch("processor.import_process.nodes.node_pdf_to_md.get_config") as mock_cfg:
            mock_cfg.return_value.minerU_base_url = self.MOCK_URL
            mock_cfg.return_value.minerU_api_token = self.MOCK_TOKEN
            with pytest.raises(RuntimeError, match="解析任务失败|pdf parse error"):
                node._step_2_upload_and_poll(real_pdf)
        mock_req.get.assert_called_once()

    @patch("processor.import_process.nodes.node_pdf_to_md.requests")
    def test_upload_and_poll_success(self, mock_req, node, real_pdf):
        """完整成功流程"""
        post_resp = MagicMock(status_code=200)
        post_resp.json.return_value = {
            "code": 0,
            "data": {
                "file_urls": ["https://upload.mineru/test.pdf"],
                "batch_id": "batch_done_001",
            },
        }
        mock_req.post.return_value = post_resp
        mock_req.put.return_value = MagicMock(status_code=200)
        poll_resp = MagicMock(status_code=200)
        poll_resp.json.return_value = {
            "code": 0,
            "data": {
                "extract_result": [{
                    "state": "done",
                    "full_zip_url": "https://dl.mineru/result.zip",
                }]
            },
        }
        mock_req.get.return_value = poll_resp
        with patch("processor.import_process.nodes.node_pdf_to_md.get_config") as mock_cfg:
            mock_cfg.return_value.minerU_base_url = self.MOCK_URL
            mock_cfg.return_value.minerU_api_token = self.MOCK_TOKEN
            zip_url = node._step_2_upload_and_poll(real_pdf)
        assert zip_url == "https://dl.mineru/result.zip"
        mock_req.post.assert_called_once()
        mock_req.put.assert_called_once()
        mock_req.get.assert_called_once()
        _, kwargs = mock_req.post.call_args
        assert "Authorization" in kwargs.get("headers", {})
        assert kwargs["headers"]["Authorization"] == f"Bearer {self.MOCK_TOKEN}"
        assert kwargs["json"]["model_version"] == "vlm"


class TestStep3DownloadAndExtract:
    """测试 _step_3_download_and_extract（mock 下载，真实 zip 操作）"""

    @patch("processor.import_process.nodes.node_pdf_to_md.requests")
    def test_download_failure_raises_runtime_error(self, mock_req, node, tmp_path):
        """ZIP 下载返回非 200 -> RuntimeError"""
        mock_req.get.return_value = MagicMock(status_code=404, text="Not Found")
        with pytest.raises(RuntimeError, match="ZIP|下载失败|404"):
            node._step_3_download_and_extract(
                "https://bad.url/result.zip", tmp_path, "test"
            )

    @patch("processor.import_process.nodes.node_pdf_to_md.requests")
    def test_download_and_extract_success(self, mock_req, node, tmp_path):
        """正常下载 -> 解压 -> full.md -> {pdf_stem}.md"""
        mock_content = "# MinerU Output\n\nExtracted content."
        zip_bytes = make_mock_zip(md_content=mock_content)
        dl_resp = MagicMock(status_code=200)
        dl_resp.content = zip_bytes
        dl_resp.iter_content.return_value = [zip_bytes]
        mock_req.get.return_value = dl_resp
        md_path_str = node._step_3_download_and_extract(
            "https://dl.mineru/result.zip", tmp_path, "test_manual"
        )
        md_path = Path(md_path_str)
        assert md_path.name == "test_manual.md"
        assert md_path.read_text(encoding="utf-8") == mock_content
        assert (tmp_path / "test_manual_result.zip").exists()
        mock_req.get.assert_called_once_with(
            "https://dl.mineru/result.zip",
            stream=True,
            timeout=(20, 300),
        )


class TestProcess:
    """测试 process() 整体流程（mock 外部依赖步骤）"""

    def test_process_happy_path(self, node, real_pdf, tmp_path):
        """mock _step_2 和 _step_3 后验证 state 正确更新"""
        mock_md_content = "# 测试 Markdown\n\nPDF-to-MD 结果。"
        state = create_default_state(pdf_path=str(real_pdf))
        state["file_path"] = str(tmp_path)
        with (
            patch.object(node, "_step_2_upload_and_poll", return_value="https://fake.zip"),
            patch.object(node, "_step_3_download_and_extract", return_value="fake.md"),
            patch("builtins.open", mock_open(read_data=mock_md_content)),
        ):
            result = node.process(state)
        assert result["md_path"] == "fake.md"
        assert result["md_content"] == mock_md_content
        assert state["md_path"] == result["md_path"]

    def test_process_propagates_exception(self, node, real_pdf, tmp_path):
        """_step_2 异常透传"""
        state = create_default_state(pdf_path=str(real_pdf))
        state["file_path"] = str(tmp_path)
        with patch.object(node, "_step_2_upload_and_poll", side_effect=RuntimeError("API unavailable")):
            with pytest.raises(RuntimeError, match="API unavailable"):
                node.process(state)
        assert state.get("md_path", "") == ""

    def test_process_missing_file_path_raises_value_error(self, node, real_pdf, tmp_path):
        """缺失 file_path -> ValueError"""
        state = create_default_state(pdf_path=str(real_pdf))
        state["file_path"] = ""
        with pytest.raises(ValueError, match="file_path"):
            node.process(state)
