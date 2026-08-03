"""node_bge_embedding 节点的 HAK180 真实模型集成测试。"""

import copy
import json
import math
from pathlib import Path

import pytest

from processor.import_process.core.state import create_default_state
from processor.import_process.nodes.node_bge_embedding import node_bge_embedding


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HAK180_CHUNKS_PATH = (
    PROJECT_ROOT
    / "output"
    / "hak180产品安全手册"
    / "chunks"
    / "chunks.json"
)
HAK180_ITEM_NAME = "BrotherHAK180烫金机"


@pytest.fixture(scope="module")
def hak180_embedding_state():
    """读取 HAK180 真实切片，并补充节点要求的商品标签。"""
    assert HAK180_CHUNKS_PATH.exists(), (
        f"HAK180 测试切片不存在：{HAK180_CHUNKS_PATH}"
    )
    chunks = json.loads(HAK180_CHUNKS_PATH.read_text(encoding="utf-8"))
    assert isinstance(chunks, list) and chunks

    for chunk in chunks:
        chunk["item_name"] = HAK180_ITEM_NAME

    return create_default_state(
        task_id="integration_hak180_bge_embedding",
        file_title="hak180产品安全手册",
        item_name=HAK180_ITEM_NAME,
        chunks=chunks,
    )


def test_node_bge_embedding_with_real_hak180_and_real_model(
    hak180_embedding_state,
):
    """真实 BGE-M3 应为全部 HAK180 切片生成可供 Milvus 使用的双向量。"""
    original_chunks = copy.deepcopy(hak180_embedding_state["chunks"])

    result = node_bge_embedding(hak180_embedding_state)

    assert result is hak180_embedding_state
    output_chunks = result["chunks"]
    assert len(original_chunks) == 6
    assert len(output_chunks) == len(original_chunks)

    for original, output in zip(original_chunks, output_chunks):
        for field, value in original.items():
            assert output[field] == value

        dense_vector = output.get("dense_vector")
        sparse_vector = output.get("sparse_vector")

        assert isinstance(dense_vector, list)
        assert len(dense_vector) == 1024
        assert all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in dense_vector
        )
        assert math.sqrt(sum(value * value for value in dense_vector)) == pytest.approx(
            1.0,
            abs=1e-3,
        )

        assert isinstance(sparse_vector, dict)
        assert sparse_vector
        assert all(isinstance(index, int) and index >= 0 for index in sparse_vector)
        assert all(
            isinstance(weight, (int, float)) and math.isfinite(weight)
            for weight in sparse_vector.values()
        )

    assert output_chunks[0]["dense_vector"] != output_chunks[1]["dense_vector"]
    assert all("dense_vector" not in chunk for chunk in original_chunks)
    assert all("sparse_vector" not in chunk for chunk in original_chunks)
