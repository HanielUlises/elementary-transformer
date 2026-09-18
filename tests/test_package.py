import elementary_transformer
from elementary_transformer import _core


def test_versions_agree() -> None:
    assert elementary_transformer.__version__ == _core.version()
