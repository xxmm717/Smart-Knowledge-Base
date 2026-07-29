"""node_item_name_recognition 节点单元测试。

测试使用 HAK180 产品安全手册的真实切片作为业务输入，但会 mock LLM、
BGE-M3 和 Milvus。这样既能验证真实文档结构，又不会产生网络请求、模型加载
或数据库写入，测试结果可以稳定重复。
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from processor.import_process.nodes import node_item_name_recognition as target


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HAK180_CHUNKS_PATH = (
    PROJECT_ROOT
    / "output"
    / "hak180产品安全手册"
    / "chunks"
    / "chunks.json"
)


@pytest.fixture
def hak180_chunks():
    """读取真实 HAK180 切片，并为每个测试返回独立副本。"""
    assert HAK180_CHUNKS_PATH.exists(), (
        f"HAK180 测试切片不存在：{HAK180_CHUNKS_PATH}"
    )
    chunks = json.loads(HAK180_CHUNKS_PATH.read_text(encoding="utf-8"))
    assert isinstance(chunks, list) and chunks
    return copy.deepcopy(chunks)


def test_step_1_uses_file_title_from_real_hak180_chunks(hak180_chunks):
    """state 无标题时，应从 HAK180 第一个切片取得文件标题。"""
    state = {"chunks": hak180_chunks}

    file_title, chunks = target.step_1_get_inputs(state)

    assert file_title == "hak180产品安全手册"
    assert chunks == hak180_chunks


def test_step_1_rejects_non_list_chunks():
    """chunks 不是列表时，应返回空列表，避免后续遍历异常。"""
    file_title, chunks = target.step_1_get_inputs(
        {"file_title": "HAK180", "chunks": "invalid"}
    )

    assert file_title == "HAK180"
    assert chunks == []


def test_step_2_builds_context_from_multiple_real_chunks(hak180_chunks):
    """上下文应包含多个前置切片，并受到总字符数上限约束。"""
    context = target.step_2_build_context(hak180_chunks)

    assert "【切片1" in context
    assert "# HAK 180 烫金机" in context
    assert "【切片2" in context
    assert "# 警告" in context
    assert len(context) <= target.CONTEXT_TOTAL_MAX_CHARS


def test_step_2_filters_invalid_chunks_and_limits_chunk_count():
    """应过滤无效切片，并且最多只读取配置指定的前 K 个切片。"""
    chunks = [None, {"title": "", "content": ""}]
    chunks.extend(
        {"title": f"标题{i}", "content": f"内容{i}"}
        for i in range(1, target.DEFAULT_ITEM_NAME_CHUNK_K + 3)
    )

    context = target.step_2_build_context(chunks)

    assert "标题1" in context
    assert f"标题{target.DEFAULT_ITEM_NAME_CHUNK_K}" not in context


def test_step_3_calls_llm_and_cleans_entity_name():
    """LLM 成功时，应构造两类消息并清理实体名称中的空白字符。"""
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content="  HAK 180 烫金机\n")

    def fake_load_prompt(name, **kwargs):
        return f"{name}:{kwargs}"

    with (
        patch.object(target, "get_llm_client", return_value=llm),
        patch.object(target, "load_prompt", side_effect=fake_load_prompt),
    ):
        item_name = target.step_3_call_llm(
            "hak180产品安全手册",
            "HAK 180 产品安全手册上下文",
        )

    assert item_name == "HAK180烫金机"
    llm.invoke.assert_called_once()
    messages = llm.invoke.call_args.args[0]
    assert len(messages) == 2
    assert messages[0].content.startswith("product_recognition_system:")
    assert messages[1].content.startswith("item_name_recognition:")


@pytest.mark.parametrize(
    ("response", "side_effect"),
    [
        (SimpleNamespace(content=" \n\t "), None),
        (None, RuntimeError("LLM unavailable")),
    ],
)
def test_step_3_falls_back_to_file_title(response, side_effect):
    """LLM 返回空值或调用异常时，应使用文件标题兜底。"""
    llm = MagicMock()
    if side_effect is not None:
        llm.invoke.side_effect = side_effect
    else:
        llm.invoke.return_value = response

    with (
        patch.object(target, "get_llm_client", return_value=llm),
        patch.object(target, "load_prompt", return_value="prompt"),
    ):
        item_name = target.step_3_call_llm("hak180产品安全手册", "context")

    assert item_name == "hak180产品安全手册"


def test_step_3_skips_llm_when_context_is_empty():
    """无上下文时不应调用 LLM，应直接返回文件标题。"""
    with patch.object(target, "get_llm_client") as get_llm:
        item_name = target.step_3_call_llm("hak180产品安全手册", "")

    assert item_name == "hak180产品安全手册"
    get_llm.assert_not_called()


def test_step_4_updates_state_and_all_real_chunks(hak180_chunks):
    """识别出的实体应写入全局状态及 HAK180 的每个切片。"""
    state = {"chunks": hak180_chunks}

    target.step_4_update_chunks(state, hak180_chunks, "HAK180烫金机")

    assert state["item_name"] == "HAK180烫金机"
    assert state["chunks"] is hak180_chunks
    assert all(
        chunk["item_name"] == "HAK180烫金机"
        for chunk in hak180_chunks
    )


def test_step_5_reads_embedding_utils_dense_and_sparse_contract():
    """应按 embedding_utils 的 dense/sparse 返回契约提取第一条向量。"""
    embedding_result = {
        "dense": [[0.1, 0.2, 0.3]],
        "sparse": [{12: 0.8, 48: 0.25}],
    }

    with patch.object(
        target,
        "generate_embedding",
        return_value=embedding_result,
    ) as generate:
        dense, sparse = target.step_5_generate_vectors("HAK180烫金机")

    generate.assert_called_once_with(["HAK180烫金机"])
    assert dense == [0.1, 0.2, 0.3]
    assert sparse == {12: 0.8, 48: 0.25}


def test_step_5_skips_embedding_for_empty_item_name():
    """实体名称为空时，不应加载 BGE-M3 模型。"""
    with patch.object(target, "generate_embedding") as generate:
        dense, sparse = target.step_5_generate_vectors("")

    assert dense is None
    assert sparse is None
    generate.assert_not_called()


def test_step_6_replaces_existing_entity_record(monkeypatch):
    """集合已存在时，也应先删除同名实体，再插入最新向量数据。"""
    monkeypatch.setenv("MILVUS_URL", "http://milvus.test:19530")
    monkeypatch.setenv("ITEM_NAME_COLLECTION", "item_names")
    client = MagicMock()
    client.has_collection.return_value = True
    state = {}

    with patch.object(target, "get_milvus_client", return_value=client):
        saved = target.step_6_save_to_milvus(
            state=state,
            file_title="hak180产品安全手册",
            item_name="HAK180烫金机",
            dense_vector=[0.1, 0.2],
            sparse_vector={12: 0.8},
        )

    client.delete.assert_called_once()
    assert "HAK180烫金机" in client.delete.call_args.kwargs["filter"]
    client.insert.assert_called_once_with(
        collection_name="item_names",
        data=[
            {
                "file_title": "hak180产品安全手册",
                "item_name": "HAK180烫金机",
                "dense_vector": [0.1, 0.2],
                "sparse_vector": {12: 0.8},
            }
        ],
    )
    assert saved is True
    assert state["item_name"] == "HAK180烫金机"


def test_process_uses_real_hak180_chunks_without_external_calls(hak180_chunks):
    """节点整体流程应关联真实 HAK180 切片，但不触发外部服务。"""
    state = {
        "task_id": "unit_hak180_item_name",
        "file_title": "hak180产品安全手册",
        "chunks": hak180_chunks,
    }

    with (
        patch.object(target, "step_3_call_llm", return_value="HAK180烫金机"),
        patch.object(
            target,
            "step_5_generate_vectors",
            return_value=([0.1, 0.2], {12: 0.8}),
        ),
        patch.object(
            target,
            "step_6_save_to_milvus",
            return_value=True,
        ) as save_to_milvus,
    ):
        result = target.NodeItemNameRecognition().process(state)

    assert result is state
    assert result["item_name"] == "HAK180烫金机"
    assert len(result["chunks"]) == len(hak180_chunks)
    assert all(
        chunk["item_name"] == "HAK180烫金机"
        for chunk in result["chunks"]
    )
    save_to_milvus.assert_called_once_with(
        state,
        "hak180产品安全手册",
        "HAK180烫金机",
        [0.1, 0.2],
        {12: 0.8},
    )


def test_process_does_not_report_milvus_success_when_save_fails(hak180_chunks):
    """Milvus保存失败时应保留识别结果，但不能记录虚假的入库成功日志。"""
    state = {
        "file_title": "hak180产品安全手册",
        "chunks": hak180_chunks,
    }

    with (
        patch.object(target, "step_3_call_llm", return_value="HAK180烫金机"),
        patch.object(target, "step_5_generate_vectors", return_value=(None, None)),
        patch.object(target, "step_6_save_to_milvus", return_value=False),
        patch.object(target, "logger") as mocked_logger,
    ):
        result = target.NodeItemNameRecognition().process(state)

    assert result["item_name"] == "HAK180烫金机"
    info_messages = [str(call.args[0]) for call in mocked_logger.info.call_args_list]
    warning_messages = [
        str(call.args[0]) for call in mocked_logger.warning.call_args_list
    ]
    assert not any("已存入Milvus" in message for message in info_messages)
    assert any("但未存入Milvus" in message for message in warning_messages)


def test_process_sets_default_entity_when_unexpected_error_occurs(hak180_chunks):
    """未预期异常不应中断工作流，并应写入默认实体名称。"""
    state = {
        "file_title": "hak180产品安全手册",
        "chunks": hak180_chunks,
    }

    with patch.object(
        target,
        "step_3_call_llm",
        side_effect=RuntimeError("unexpected"),
    ):
        result = target.NodeItemNameRecognition().process(state)

    assert result["item_name"] == "未知商品"
