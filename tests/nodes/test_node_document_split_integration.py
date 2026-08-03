"""
RAG 1-3阶段真实API集成测试。

测试链路：hl3040网络说明书.pdf -> MinerU -> VLM/MinIO -> 文档切分。
运行命令：
    uv run pytest tests/nodes/test_node_document_split_integration.py -v -s

测试产物会保留在 output/hl3040网络说明书/，便于人工检查Markdown和chunk.json。
"""

import json
import os
import re
from unittest.mock import patch
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

from processor.config.config import get_config
from processor.config.llm_config import llm_config
from processor.config.minio_config import minio_config
from processor.import_process.core.state import create_default_state
from processor.import_process.nodes.node_document_split import (
    DEFAULT_MAX_CONTENT_LENGTH,
    node_document_split,
)
from processor.import_process.nodes import node_md_img as md_img_node
from processor.import_process.nodes.node_pdf_to_md import node_pdf_to_md
from processor.utils.client.minio_client import minio_client


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = PROJECT_ROOT / "doc" / "hl3040网络说明书.pdf"
OUTPUT_ROOT = PROJECT_ROOT / "output"
# HL3040包含大量图片；默认只对前4张执行真实VLM/MinIO，避免一次测试触发数百次模型调用。
# 可通过环境变量REAL_IMAGE_LIMIT调整，例如REAL_IMAGE_LIMIT=10。
REAL_IMAGE_LIMIT = int(os.getenv("REAL_IMAGE_LIMIT", "4"))


def _check_real_api_dependencies() -> None:
    """真实链路缺少配置时明确跳过，不把环境问题误报成业务失败。"""
    cfg = get_config()
    missing = []
    if not cfg.minerU_base_url or not cfg.minerU_api_token:
        missing.append("MinerU")
    if not llm_config.base_url or not llm_config.api_key or not llm_config.lv_model:
        missing.append("VLM")
    if not minio_config.endpoint or not minio_config.bucket_name or minio_client is None:
        missing.append("MinIO")
    if missing:
        pytest.skip(f"真实API测试缺少可用配置：{', '.join(missing)}")


@pytest.fixture(scope="module")
def hl3040_pipeline_result():
    """按生产顺序执行三个节点，并共享同一份状态对象。"""
    assert PDF_PATH.exists(), f"测试PDF不存在：{PDF_PATH}"
    _check_real_api_dependencies()

    state = create_default_state(
        task_id="integration_hl3040_rag_1_3",
        local_file_path=str(PDF_PATH),
        pdf_path=str(PDF_PATH),
        local_dir=str(OUTPUT_ROOT),
        file_title=PDF_PATH.stem,
    )

    # 节点1：真实上传PDF到MinerU，轮询、下载并解压Markdown和图片。
    state = node_pdf_to_md(state)
    original_md_path = Path(state["md_path"])
    assert original_md_path.exists()
    assert isinstance(state["md_content"], str)
    assert len(state["md_content"]) > 100
    original_md_content = state["md_content"]

    images_dir = original_md_path.parent / "images"

    # 节点2：真实调用VLM生成摘要，上传MinIO，并生成同目录下的*_new.md。
    all_targets = md_img_node._step_2_get_scan_images(original_md_content, images_dir)
    selected_targets = all_targets[:REAL_IMAGE_LIMIT]
    selected_image_names = [image_file for image_file, _, _ in selected_targets]

    # 仍调用真实NodeMdImg.process()，只限制本次测试实际发送给VLM/MinIO的图片样本。
    original_scan = md_img_node._step_2_get_scan_images

    def scan_limited_images(md_content, scan_images_dir):
        return original_scan(md_content, scan_images_dir)[:REAL_IMAGE_LIMIT]

    with patch.object(md_img_node, "_step_2_get_scan_images", side_effect=scan_limited_images):
        state = md_img_node.node_md_img(state)
    processed_md_path = Path(state["md_path"])
    assert processed_md_path.exists()
    assert processed_md_path.name == f"{PDF_PATH.stem}_new.md"

    # 节点3：执行标题初切、超长切分、短块合并并备份chunk.json。
    state = node_document_split(state)
    chunk_json_path = processed_md_path.parent / "chunks" / "chunk.json"

    return {
        "state": state,
        "original_md_path": original_md_path,
        "processed_md_path": processed_md_path,
        "images_dir": images_dir,
        "selected_image_names": selected_image_names,
        "chunk_json_path": chunk_json_path,
    }


def test_pdf_to_md_real_result(hl3040_pipeline_result):
    """MinerU应生成真实Markdown，并保留对应的解析目录。"""
    result = hl3040_pipeline_result
    md_path = result["original_md_path"]

    assert md_path.parent.name == PDF_PATH.stem
    assert md_path.stat().st_size > 100
    assert "#" in md_path.read_text(encoding="utf-8")
    print(f"MinerU输出：{md_path}")


def test_md_img_real_result(hl3040_pipeline_result):
    """有本地图片时，应生成摘要并将图片引用替换成HTTP地址。"""
    result = hl3040_pipeline_result
    processed_content = result["processed_md_path"].read_text(encoding="utf-8")
    selected_image_names = result["selected_image_names"]

    assert result["images_dir"].exists()
    assert selected_image_names, "MinerU结果中没有本地图片，未覆盖VLM/MinIO真实链路"
    remote_image_refs = re.findall(r"!\[[^\]]+\]\(https?://[^)]+\)", processed_content)
    assert len(remote_image_refs) >= len(selected_image_names)
    for image_name in selected_image_names:
        assert re.search(rf"!\[[^\]]+\]\(https?://[^)]*{re.escape(image_name)}\)", processed_content)
    print(f"图片替换：真实处理{len(selected_image_names)}张 -> {len(remote_image_refs)}个HTTP引用")


def test_document_split_real_result(hl3040_pipeline_result):
    """最终状态和本地备份应包含一致、非空、长度受控的Chunk列表。"""
    result = hl3040_pipeline_result
    state = result["state"]
    chunk_json_path = result["chunk_json_path"]

    chunks = state.get("chunks")
    assert isinstance(chunks, list) and chunks
    assert all(isinstance(chunk.get("content"), str) and chunk["content"].strip() for chunk in chunks)
    assert all(chunk.get("file_title") == PDF_PATH.stem for chunk in chunks)
    assert all(chunk.get("parent_title") for chunk in chunks)
    assert max(len(chunk["content"]) for chunk in chunks) <= DEFAULT_MAX_CONTENT_LENGTH

    assert chunk_json_path.exists(), f"Chunk备份不存在：{chunk_json_path}"
    backup_chunks = json.loads(chunk_json_path.read_text(encoding="utf-8"))
    assert backup_chunks == chunks

    print(f"文档切分：{len(chunks)}个Chunk")
    print(f"最大Chunk长度：{max(len(chunk['content']) for chunk in chunks)}")
    print(f"Chunk备份：{chunk_json_path}")
