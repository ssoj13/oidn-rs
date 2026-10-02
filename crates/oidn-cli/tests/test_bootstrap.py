"""Bootstrap target discovery and failure propagation regressions (no Cargo execution)."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("oidn_bootstrap", ROOT / "bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        bootstrap._META = None
        bootstrap.VERBOSE = False

    def test_install_uses_manifest_directory_bin_name_and_debug_profile(self):
        metadata = {"workspace_members": ["cli"], "packages": [{
            "id": "cli", "name": "oidn-cli", "manifest_path": str(ROOT / "crates/oidn-cli/Cargo.toml"),
            "targets": [{"name": "oidn-rs", "kind": ["bin"]}], "dependencies": []}]}
        with patch.object(bootstrap, "meta", return_value=metadata), \
             patch.object(bootstrap, "run", return_value=(0, "", 1.0)) as run:
            self.assertEqual(bootstrap.install(True, False), 0)
            run.assert_called_once_with(["cargo", "install", "--path",
                str(ROOT / "crates/oidn-cli"), "--bin", "oidn-rs", "--force", "--debug"])

    def test_metadata_failure_is_not_an_empty_success(self):
        with patch.object(bootstrap.subprocess, "run",
                          return_value=SimpleNamespace(returncode=101, stdout="", stderr="bad manifest")):
            with self.assertRaisesRegex(RuntimeError, "bad manifest"):
                bootstrap.bin_packages()

    def test_invalid_metadata_is_not_cached(self):
        invalid = SimpleNamespace(returncode=0, stdout='{"packages": {}, "workspace_members": []}', stderr="")
        with patch.object(bootstrap.subprocess, "run", return_value=invalid):
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "invalid workspace"):
                    bootstrap.meta()
                self.assertIsNone(bootstrap._META)

    def test_verbose_reaches_cargo(self):
        bootstrap.VERBOSE = True
        with patch.object(bootstrap.subprocess, "run",
                          return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as run:
            bootstrap.run(["cargo", "check"])
            self.assertEqual(run.call_args.args[0], ["cargo", "--verbose", "check"])


if __name__ == "__main__":
    unittest.main()
