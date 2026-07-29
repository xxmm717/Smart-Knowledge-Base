"""
node_md_img 真实 API 集成测试

使用 output/test/ 下的 MinerU 转换产物进行真实 API 测试。
依赖：VLM 模型（qwen3-vl-32b-thinking），MinIO（可选）。

运行：uv run pytest tests/nodes/test_node_md_img_integration.py -v -s
"""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ── 模块导入 ──
try:
    from processor.import_process.nodes.node_md_img import NodeMdImg
    from processor.import_process.core.state import create_default_state
    from processor.utils.client.minio_client import minio_client
    from processor.config.llm_config import llm_config
    IMPORT_OK = True
except Exception as e:
    IMPORT_OK = False
    _import_err = e

if not IMPORT_OK:
    pytest.skip(f"模块导入失败，跳过: {_import_err}", allow_module_level=True)

# ── 测试数据检查 ──
TEST_MD = Path("output/test/test.md")
TEST_IMAGES = TEST_DIR = Path("output/test") / "images"
TEST_MD_PARENT = Path("output/test")

if not TEST_MD.exists():
    pytest.skip(f"测试文件不存在: {TEST_MD}", allow_module_level=True)

# ── 外部依赖检查 ──
HAS_MINIO = minio_client is not None
HAS_VLM = bool(getattr(llm_config, "lv_model", None))


# ── Fixtures ──

@pytest.fixture
def node():
    return NodeMdImg()


@pytest.fixture
def md_state():
    return create_default_state(
        task_id="test_md_img_001",
        md_path=str(TEST_MD),
    )


# ══════════════════════════════════════════
# 测试用例
# ══════════════════════════════════════════

class TestStep1GetContent:
    """_step_1_get_content — 读取 MD"""

    def test_read_content(self, node, md_state):
        content, md_path, img_dir = node._step_1_get_content(md_state)
        assert len(content) > 100
        assert md_path == TEST_MD
        assert "H3C" in content or "LA2608" in content
        print(f"  MD: {len(content)} 字符, 图片目录: {img_dir.name}")

    def test_images_dir_exists(self, node, md_state):
        _, _, img_dir = node._step_1_get_content(md_state)
        assert img_dir == TEST_IMAGES
        assert img_dir.exists()
        assert len(list(img_dir.glob("*.jpg"))) == 4

    def test_missing_md_path_raises(self, node):
        with pytest.raises(ValueError, match="md_path"):
            node._step_1_get_content(create_default_state())


class TestFindImageInMd:
    """_find_image_in_md — 图片上下文提取"""

    @pytest.fixture
    def content(self, node, md_state):
        c, _, _ = node._step_1_get_content(md_state)
        return c

    def test_all_images_found(self, node, content):
        for img_file in sorted(TEST_IMAGES.iterdir()):
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".gif"):
                continue
            ctx = node._find_image_in_md(content, img_file.name)
            assert ctx is not None, f"{img_file.name} 未匹配"
            pre, post = ctx
            assert len(pre) > 0 or len(post) > 0
        print("  4/4 图片全部匹配")

    def test_context_length(self, node, content):
        pre, post = node._find_image_in_md(content, sorted(TEST_IMAGES.glob("*.jpg"))[0].name)
        assert len(pre) <= 100
        assert len(post) <= 100

    def test_unmatched_image(self, node, content):
        assert node._find_image_in_md(content, "nope.png") is None


class TestStep2ScanImages:
    """_step_2_get_scan_images — 扫描图片"""

    def test_scan(self, node, md_state):
        content, _, img_dir = node._step_1_get_content(md_state)
        targets = node._step_2_get_scan_images(content, img_dir)
        assert len(targets) == 4
        for fname, fpath, ctx in targets:
            assert Path(fpath).exists()
            assert isinstance(ctx, tuple) and len(ctx) == 2
        print(f"  扫描到 {len(targets)} 张图片")


class TestStep5BackupNewMdFile:
    """_step_5_backup_new_md_file — 备份保存"""

    def test_creates_new_file(self, node, md_state):
        content, md_path, _ = node._step_1_get_content(md_state)
        new = node._step_5_backup_new_md_file(str(md_path), content + "\n# EXTRA")
        p = Path(new)
        assert p.exists()
        assert p.name == "test_new.md"
        assert "EXTRA" in p.read_text(encoding="utf-8")
        p.unlink(missing_ok=True)

    def test_original_unchanged(self, node, md_state):
        orig = TEST_MD.read_text(encoding="utf-8")
        content, md_path, _ = node._step_1_get_content(md_state)
        new = node._step_5_backup_new_md_file(str(md_path), content.upper())
        Path(new).unlink(missing_ok=True)
        assert TEST_MD.read_text(encoding="utf-8") == orig


@pytest.mark.skipif(not HAS_VLM, reason="VLM 模型未配置 (llm_config.lv_model)")
class TestStep3GenerateSummaries:
    """_step_3_generate_summaries — VLM 多模态摘要"""

    def test_generate_summaries(self, node, md_state):
        content, _, img_dir = node._step_1_get_content(md_state)
        targets = node._step_2_get_scan_images(content, img_dir)
        summaries = node._step_3_generate_summaries(TEST_MD.stem, targets)
        assert len(summaries) == 4
        for fname, summary in summaries.items():
            assert summary is not None and len(summary) > 0
            print(f"  {fname[:16]}...: {summary[:60]}...")


@pytest.mark.skipif(not HAS_MINIO, reason="MinIO 客户端未初始化")
class TestStep4UploadAndReplace:
    """_step_4_upload_and_replace — MinIO 上传 + MD 替换"""

    def test_upload_and_replace(self, node, md_state):
        from processor.utils.client.minio_client import minio_client
        content, _, img_dir = node._step_1_get_content(md_state)
        targets = node._step_2_get_scan_images(content, img_dir)
        from processor.import_process.nodes.node_md_img import NodeMdImg
        summaries = {f: "摘要" for f, _, _ in targets}
        new_content = node._step_4_upload_and_replace(minio_client, TEST_MD.stem, targets, summaries, content)
        assert "http" in new_content, "替换后应包含 MinIO URL"


@pytest.mark.skipif(not HAS_VLM or not HAS_MINIO, reason="需要 VLM + MinIO 同时可用")
class TestFullPipeline:
    """完整 process() 端到端"""

    def test_full_process(self, node, md_state):
        result = node.process(md_state)
        assert "md_path" in result
        assert "md_content" in result
        new_md = Path(result["md_path"])
        assert new_md.exists()
        assert "http" in result["md_content"]
        new_md.unlink(missing_ok=True)
        print(f"  完整管道成功: {new_md.name}")
