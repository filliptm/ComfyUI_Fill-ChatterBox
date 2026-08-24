import builtins
import importlib
import sys
import tomllib
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class OptionalPerthTests(unittest.TestCase):
    def test_import_continues_when_perth_raises_python_312_attribute_error(self):
        original_import = builtins.__import__

        def import_with_broken_perth(name, *args, **kwargs):
            if name == "perth":
                raise AttributeError("module 'pkgutil' has no attribute 'ImpImporter'")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_with_broken_perth):
            package = importlib.import_module("local_chatterbox.chatterbox")

        for module_name in ("tts", "tts_turbo", "mtl_tts", "vc"):
            module = getattr(package, module_name)
            self.assertFalse(module.PERTH_AVAILABLE)

    def test_dependency_manifests_match_without_optional_perth(self):
        requirements = {
            line.strip()
            for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

        self.assertEqual(requirements, set(project["dependencies"]))
        self.assertNotIn("resemble-perth", requirements)


if __name__ == "__main__":
    unittest.main()
