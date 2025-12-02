from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_patch_modeling_modules_replaces_transformers(tmp_path, monkeypatch):
    # create a dummy local module structure: models/foo/modeling_bar.py
    models_dir = tmp_path / "models" / "foo"
    models_dir.mkdir(parents=True)
    module_path = models_dir / "modeling_bar.py"
    module_path.write_text("value = 'local version'\n", encoding="utf-8")

    # create __init__.py files so packages import cleanly
    (tmp_path / "models" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "models" / "foo" / "__init__.py").write_text("", encoding="utf-8")

    sys.path.insert(0, str(tmp_path))
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    import sniff
    try:
        sniff.patch_modeling_modules(root=tmp_path / "models")
        module = importlib.import_module("transformers.models.foo.modeling_bar")
        assert module.value == "local version"
    finally:
        sys.path.remove(str(tmp_path))
        sys.path.remove(str(repo_root))
        for key in list(sys.modules):
            if key.startswith("models.") or key.startswith("transformers.models.foo"):
                sys.modules.pop(key, None)
