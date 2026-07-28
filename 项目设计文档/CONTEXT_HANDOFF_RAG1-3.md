# RAG 1-3 阶段上下文交接文档

本文档用于将当前窗口已经完成的讨论、代码状态、测试结果和后续注意事项交接给新的对话窗口。

## 1. 项目背景

- 项目路径：`D:\Code Demo\zhanggui_zhiku`
- 项目类型：Python `uv` 项目
- 项目目标：RAG 智能知识库
- 当前已经完成的 RAG 1-3 阶段：
  1. `node_pdf_to_md`：PDF 调用 MinerU 转换为 Markdown
  2. `node_md_img`：图片多模态语义对齐
  3. `node_document_split`：Markdown 文档切分
- 当前工作区最近提交：

```text
d9efb19 feat(import): complete document splitting and real pipeline test
```

提交时按照用户要求，将 `.gitignore` 未排除的文件全部提交，包括两个日志文件。

## 2. 项目测试结构

企业项目中，测试代码放在项目根目录的 `tests/` 下，当前结构主要是：

```text
tests/
├── __init__.py
├── conftest.py
└── nodes/
    ├── __init__.py
    ├── test_node_pdf_to_md.py
    ├── test_node_pdf_to_md_integration.py
    ├── test_node_md_img_integration.py
    └── test_node_document_split_integration.py
```

`tests/conftest.py` 负责把项目根目录加入 `sys.path`，并提供 `doc_dir`、`real_pdf` 等共享 fixture。

测试使用 `pytest`，通过 `uv` 执行：

```powershell
uv run pytest tests/nodes/test_node_pdf_to_md.py -q
uv run pytest tests/nodes/test_node_document_split_integration.py -v -s
```

## 3. 节点一：node_pdf_to_md

文件：`processor/import_process/nodes/node_pdf_to_md.py`

### 3.1 主要职责

```text
校验 PDF 和输出目录
    -> 获取 MinerU 上传 URL
    -> 上传 PDF
    -> 轮询解析状态
    -> 下载解析结果 ZIP
    -> 解压 full.md 和 images
    -> 重命名为 PDF 同名 Markdown
    -> 更新 state["md_path"] 和 state["md_content"]
```

### 3.2 状态字段要求

当前实现的 `_step_1_validate_path()` 实际要求：

```python
state["pdf_path"]   # PDF 文件路径
state["file_path"]  # 输出目录
```

`process()` 会写入：

```python
state["md_path"]
state["md_content"]
```

虽然部分文档注释曾写成 `local_dir`，但当前 PDF 节点实际使用的是 `file_path`。端到端测试状态同时传入了 `file_path` 和 `local_dir`。

### 3.3 已修复问题

1. Markdown 原来使用二进制模式读取，却传入 `encoding`：

```python
open(md_path, "rb", encoding="utf-8")
```

已改为文本模式：

```python
open(md_path, "r", encoding="utf-8")
```

这样后续节点收到的是 `str`，而不是 `bytes`。

2. MinerU ZIP 下载原来直接使用 `response.content`，代理连接中断时会出现 `ChunkedEncodingError`。

现在改为：

- `stream=True` 流式下载；
- 每次写入 1 MB 数据块；
- 下载后通过 `zipfile.is_zipfile()` 校验；
- 最多重试 3 次；
- 每次失败删除不完整 ZIP。

## 4. 节点二：node_md_img

文件：`processor/import_process/nodes/node_md_img.py`

### 4.1 主要职责

```text
读取 Markdown
    -> 找到 Markdown 中引用的本地图片
    -> 提取图片前后文 pre_text/post_text
    -> 调用 VLM 生成图片摘要
    -> 上传图片到 MinIO
    -> 将本地图片引用替换为 HTTP URL
    -> 将 VLM 摘要写入 alt 属性
    -> 保存 *_new.md
    -> 更新 state["md_path"] 和 state["md_content"]
```

### 4.2 MinIO 配置

此前已经确认：

- `9010`：MinIO API 端口
- `9011`：MinIO Console 控制台端口
- 图片 HTTP 地址通常类似：

```text
http://192.168.246.128:9010/knowledge-base-files/upload-images/<document>/<image>.jpg
```

Clash 代理曾导致 MinIO 图片下载失败，修改代理规则放行内网地址后，图片上传和访问测试成功。

### 4.3 已完成测试

之前使用 `output/test/test.md` 测试过：

- 图片上下文提取；
- `pre_text` 和 `post_text`；
- VLM 图片摘要；
- MinIO 上传；
- Markdown 图片替换；
- 生成 `output/test/test_new.md`。

曾经出现的代码问题包括 `signed_url` 拼写、`window_seconds` 拼写、`return summary`、`llm_config` 变量名、MinIO bucket 字段和方法 `self` 等，已经修复。

## 5. 节点三：node_document_split

文件：`processor/import_process/nodes/node_document_split.py`

### 5.1 总体流程

```text
步骤1：读取和标准化 Markdown
    -> 步骤2：按 Markdown 标题初步切分
    -> 步骤3：无标题场景兜底
    -> 步骤4：超长章节切分、短 Chunk 合并、元数据补充
    -> 步骤5：输出统计日志
    -> 步骤6：写入 state["chunks"] 并备份 chunk.json
```

### 5.2 步骤 2：按标题切分

标题正则：

```python
title_pattern = r"^\s*#{1,6}\s+.+"
```

支持 1 到 6 级 Markdown 标题，并且通过 `in_code_block` 避免把代码块中的 `#` 误识别为标题。

关键逻辑：

```python
if stripped_line.startswith("```") or stripped_line.startswith("~~~"):
    in_code_block = not in_code_block
```

`in_code_block = not in_code_block` 是布尔值翻转：

```text
False -> True：进入代码块
True  -> False：离开代码块
```

遇到新标题时：

```python
_flush_section()
current_title = line.strip()
current_lines = [current_title]
```

先把上一章节保存到 `sections`，再创建新的行列表。上一章节不会丢失。

此前有一个重要错误：标题判断曾经错误地匹配 `file_title`：

```python
re.match(title_pattern, file_title)
```

已修正为匹配当前行：

```python
re.match(title_pattern, line)
```

### 5.3 步骤 3：无标题兜底

标题切分完成后，根据 `title_count` 判断：

```python
if title_count == 0:
    return [{
        "title": "无标题",
        "content": content,
        "file_title": file_title,
    }]
```

之所以放在标题切分之后，是因为必须先遍历完整篇 Markdown，才能确认全文确实没有任何标题。

### 5.4 步骤 4：精细化切分

默认配置：

```python
DEFAULT_MAX_CONTENT_LENGTH = 2000
MIN_CONTENT_LENGTH = 500
```

超长章节使用 LangChain：

```python
RecursiveCharacterTextSplitter(
    chunk_size=available_len,
    chunk_overlap=0,
    separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "]
)
```

优先按照以下顺序切分：

```text
空行/段落 -> 换行 -> 中文标点 -> 英文标点 -> 空格 -> 必要时硬切
```

标题切分和最大长度切分的关系：

- 按标题切分：保留文档语义结构，但某个章节可能过长；
- 按最大长度切分：控制模型输入大小，但可能破坏语义；
- 当前项目采用：先按标题切分，再只对超长章节按长度二次切分。

### 5.5 短 Chunk 合并

`_merge_short_sections()` 设计为合并：

- 当前 Chunk 长度小于 `MIN_CONTENT_LENGTH`；
- 当前 Chunk 与下一个 Chunk 属于同一个 `parent_title`；
- 合并后不能超过 `max_len`。

合并时使用：

```python
current_chunk["content"] += "\n\n" + next_content
```

两个 `\n` 表示空一行，用于保持两个原始文本块之间的 Markdown 段落边界。

已处理的元数据问题：

- 未二次切分的章节也补充自己的 `parent_title`；
- 避免不同章节的 `parent_title=None` 被错误判断为同一父章节；
- 合并后检查最大长度；
- `step_4_refine_chunks()` 现在正确返回 `final_sections`。

### 5.6 步骤 6：Chunk 备份

为了不改变原有 Markdown 路径逻辑，备份目录根据当前 `state["md_path"]` 推导：

```text
output/
└── <document>/
    ├── <document>.md
    ├── <document>_new.md
    ├── images/
    └── chunks/
        └── chunk.json
```

当前代码核心逻辑：

```python
document_dir = os.path.dirname(os.path.abspath(md_path))
backup_dir = os.path.join(document_dir, "chunks")
backup_path = os.path.join(backup_dir, "chunk.json")
```

使用 `w` 模式意味着重新处理同一个文档时，会覆盖该文档自己的旧备份，不会把不同文档混到一个 JSON 中。

## 6. HL3040 真实端到端测试

目标 PDF：

```text
doc/hl3040网络说明书.pdf
```

测试文件：

```text
tests/nodes/test_node_document_split_integration.py
```

运行：

```powershell
uv run pytest tests/nodes/test_node_document_split_integration.py -v -s
```

测试结果：

| 测试阶段 | 结果 |
|---|---|
| MinerU PDF 上传、解析、下载、解压 | 通过 |
| 生成 `hl3040网络说明书.md` | 通过 |
| VLM 图片摘要 | 真实处理 4 张，全部成功 |
| MinIO 图片上传 | `4/4` 成功 |
| Markdown 图片引用替换 | 4 个 HTTP 图片引用 |
| Markdown 标题识别 | 428 个标题 |
| 初始子 Chunk | 435 个 |
| 最终 Chunk | 426 个 |
| 最大 Chunk 长度 | 1928，未超过 2000 |
| `chunks/chunk.json` 备份 | 通过，426 个 Chunk |

最终测试结果：

```text
3 passed in 46.42s
```

### 图片样本限制说明

HL3040 解析出了 489 张图片。由于 `node_md_img` 当前逐张串行调用 VLM，并且有速率限制，测试默认只处理前 4 张，避免一次集成测试消耗数百次模型调用。

可以调整样本数：

```powershell
$env:REAL_IMAGE_LIMIT="10"
uv run pytest tests/nodes/test_node_document_split_integration.py -v -s
```

不要默认设置为 489，除非明确需要做全量图片处理测试。

## 7. 回归测试结果

```text
uv run pytest tests/nodes/test_node_pdf_to_md.py -q
16 passed
```

端到端测试：

```text
uv run pytest tests/nodes/test_node_document_split_integration.py -v -s
3 passed in 46.42s
```

## 8. 当前提交状态

最近提交：

```text
d9efb19 feat(import): complete document splitting and real pipeline test
```

本次提交包含：

```text
processor/import_process/core/state.py
processor/import_process/nodes/node_entry.py
processor/import_process/nodes/node_document_split.py
processor/import_process/nodes/node_pdf_to_md.py
tests/nodes/test_node_document_split_integration.py
tests/nodes/test_node_pdf_to_md.py
logs/app_20260726.log
logs/app_20260727.log
```

提交统计：

```text
8 files changed, 767 insertions(+), 26 deletions(-)
```

当前工作区在提交后已干净。

## 9. 新窗口继续工作时的建议

新窗口开始后，建议先执行：

```powershell
git status --short
git log -1 --oneline
```

然后阅读：

```text
processor/import_process/nodes/node_pdf_to_md.py
processor/import_process/nodes/node_md_img.py
processor/import_process/nodes/node_document_split.py
processor/import_process/core/state.py
tests/nodes/test_node_document_split_integration.py
```

后续实现应继续沿用当前状态传递方式：

```text
pdf_path/file_path
    -> md_path/md_content
    -> md_path/md_content（处理后的_new.md）
    -> chunks
```

注意不要随意修改现有 `output/<document>/` 下 Markdown 和图片的路径约定。Chunk 备份应继续使用：

```text
<当前Markdown所在目录>/chunks/chunk.json
```

如果后续阶段要接入向量库，重点检查：

- `chunks` 中每个元素的 `content` 是否为非空字符串；
- `file_title`、`parent_title` 是否始终存在；
- 是否需要增加 `document_id`、`chunk_index`、`part_start`、`part_end` 等溯源字段；
- 是否需要将测试 fixture 与生产路径彻底解耦；
- 是否需要把全量 489 张图片测试拆成单独的长耗时测试。

