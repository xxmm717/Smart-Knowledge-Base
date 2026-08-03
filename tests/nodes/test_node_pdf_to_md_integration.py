import requests
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from processor.config.config import get_config
from processor.import_process.nodes.node_pdf_to_md import node_pdf_to_md

import pytest

cfg = get_config()
if not cfg.minerU_base_url or not cfg.minerU_api_token:
    pytest.skip("缺少 MinerU 配置", allow_module_level=True)

API_BASE = "https://mineru.net/api/v4"
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {cfg.minerU_api_token}"}


@pytest.fixture
def node():
    return node_pdf_to_md


class TestRealMinerUIntegration:

    def test_api_connectivity(self):
        """验证 MinerU API 可达、认证有效"""
        r = requests.post(f"{API_BASE}/file-urls/batch", headers=HEADERS,
                          json={"files": [{"name": "test.pdf"}], "model_version": "vlm"})
        assert r.status_code == 200
        data = r.json()
        assert data["code"] == 0
        assert "file_urls" in data.get("data", {})
        print("  MinerU API OK")

    def test_step2_real_upload(self, node, real_pdf):
        """真实上传 + 轮询，验证拿到 ZIP 下载链接"""
        zip_url = _step_2_upload_and_poll(real_pdf)
        assert zip_url
        print(f"  ZIP URL 获取成功 ({zip_url[:60]}...)")

    def test_full_pipeline(self, node, real_pdf, tmp_path):
        """完整管道：上传 → 轮询 → 下载 → 解压 → 验证 MD 内容"""
        out = tmp_path / "mineru_out"
        out.mkdir(parents=True, exist_ok=True)
        zip_url = _step_2_upload_and_poll(real_pdf)
        md_str = _step_3_download_and_extract(zip_url, out, real_pdf.stem)
        md_file = Path(md_str)
        assert md_file.exists()
        content = md_file.read_text(encoding="utf-8")
        assert len(content) > 100
        assert "#" in content
        print(f"  完整管道成功: {md_file.name} ({md_file.stat().st_size} bytes, {len(content)} chars)")
        print(f"  标题: {content.splitlines()[0]}")
