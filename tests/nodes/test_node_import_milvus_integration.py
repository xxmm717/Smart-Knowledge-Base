"""node_import_milvus 的 HAK180 真实 Milvus 集成测试。"""

import copy
import json
from pathlib import Path
from uuid import uuid4

from processor.config.milvus_config import milvus_config
from processor.import_process.nodes.node_import_milvus import NodeImportMilvus
from processor.utils.client.milvus_client import get_milvus_client
from processor.utils.escape_milvus_string_utils import escape_milvus_string


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HAK180_VECTORS_PATH = (
    PROJECT_ROOT
    / "output"
    / "hak180产品安全手册"
    / "chunks"
    / "chunks_with_embeddings_debug.json"
)


def _load_hak180_chunks_for_milvus(test_item_name: str) -> list[dict]:
    """恢复调试 JSON 的向量类型，使其与运行时 state.chunks 保持一致。"""
    assert HAK180_VECTORS_PATH.exists(), (
        f"HAK180 向量调试产物不存在：{HAK180_VECTORS_PATH}"
    )
    chunks = json.loads(HAK180_VECTORS_PATH.read_text(encoding="utf-8"))
    assert isinstance(chunks, list) and len(chunks) == 6

    restored_chunks = []
    for chunk in chunks:
        restored = copy.deepcopy(chunk)
        restored.pop("chunk_id", None)
        restored["item_name"] = test_item_name
        restored["dense_vector"] = [float(value) for value in restored["dense_vector"]]
        restored["sparse_vector"] = {
            int(index): float(weight)
            for index, weight in restored["sparse_vector"].items()
        }
        restored_chunks.append(restored)
    return restored_chunks


def test_node_import_milvus_with_real_hak180_and_real_milvus():
    """真实 Milvus 应保存 HAK180 的 6 个双向量 Chunk，并能由唯一标签查询。"""
    assert milvus_config.chunks_collection, "未配置 CHUNKS_COLLECTION"
    client = get_milvus_client()
    assert client is not None, "无法连接 Milvus"

    collection_name = milvus_config.chunks_collection
    test_item_name = f"__test_hak180_import_{uuid4().hex}"
    chunks = _load_hak180_chunks_for_milvus(test_item_name)
    state = {
        "task_id": "integration_hak180_import_milvus",
        "file_title": "hak180产品安全手册",
        "item_name": test_item_name,
        "chunks": chunks,
    }

    filter_expr = f'item_name == "{escape_milvus_string(test_item_name)}"'

    try:
        result = NodeImportMilvus().process(state)
        output_chunks = result["chunks"]

        assert client.has_collection(collection_name=collection_name)
        assert {"dense_vector_index", "sparse_vector_index"}.issubset(
            set(client.list_indexes(collection_name=collection_name))
        )
        assert len(output_chunks) == 6
        assert all(str(chunk["chunk_id"]).isdigit() for chunk in output_chunks)
        assert len({chunk["chunk_id"] for chunk in output_chunks}) == 6

        client.flush(collection_name=collection_name)
        client.load_collection(collection_name=collection_name)
        rows = client.query(
            collection_name=collection_name,
            filter=filter_expr,
            output_fields=["chunk_id", "item_name", "title", "content"],
        )

        assert len(rows) == len(chunks)
        assert {row["item_name"] for row in rows} == {test_item_name}
        assert {row["title"] for row in rows} == {
            chunk["title"] for chunk in chunks
        }
    finally:
        if client.has_collection(collection_name=collection_name):
            client.delete(collection_name=collection_name, filter=filter_expr)
            client.flush(collection_name=collection_name)
