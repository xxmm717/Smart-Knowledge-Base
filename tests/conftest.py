"""
conftest - 测试共享 Fixtures

确保 processor 包可 import，并提供节点实例、文件路径等共享 fixture。
"""

import sys
from pathlib import Path

import pytest

# ── 确保项目根目录可 import ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── 共享 Fixtures ──

@pytest.fixture(scope="session")
def doc_dir() -> Path:
    """doc/ 目录的绝对路径"""
    return PROJECT_ROOT / "doc"


@pytest.fixture(scope="session")
def real_pdf(doc_dir: Path) -> Path:
    """从 doc/ 中选取一个真实 PDF 文件用于路径校验测试"""
    pdf = doc_dir / "test.pdf"
    if pdf.exists():
        return pdf
    pdfs = sorted(doc_dir.glob("*.pdf"))
    if pdfs:
        return pdfs[0]
    pytest.skip("doc/ 目录下没有任何 PDF 文件")


@pytest.fixture
def node():
    """每个测试获得一个新的 NodePdfToMd 实例"""
    from processor.import_process.nodes.node_pdf_to_md import NodePdfToMd
    return NodePdfToMd()
