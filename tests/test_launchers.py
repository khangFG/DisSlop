import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestLaunchers(unittest.TestCase):
    """Verifies launchers anchor working directory to script location."""

    def test_run_bat_anchored_to_script_dir(self) -> None:
        root = Path(__file__).resolve().parent.parent
        with open(root / "run.bat", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("%~dp0", content)
        self.assertIn('cd /d "%~dp0"', content)

    def test_run_ps1_anchored_to_script_dir(self) -> None:
        root = Path(__file__).resolve().parent.parent
        with open(root / "run.ps1", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertTrue("PSScriptRoot" in content or "MyInvocation" in content)
        self.assertIn("Set-Location", content)

    def test_run_bat_changes_directory_when_launched_from_elsewhere(self) -> None:
        root = Path(__file__).resolve().parent.parent
        py_exe = sys.executable
        with tempfile.TemporaryDirectory() as foreign_dir:
            test_bat = root / "_test_anchor.bat"
            try:
                test_bat.write_text(
                    f'@echo off\ncd /d "%~dp0"\n"{py_exe}" -c "import os, pathlib; print(pathlib.Path(os.getcwd()).resolve())"\n',
                    encoding="utf-8",
                )
                res = subprocess.run(
                    ["cmd.exe", "/c", f'cd /d "{foreign_dir}" && call "{test_bat}"'],
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=foreign_dir,
                )
                self.assertEqual(Path(res.stdout.strip()).resolve(), root.resolve())
            finally:
                if test_bat.exists():
                    test_bat.unlink()

    def test_run_ps1_changes_directory_when_launched_from_elsewhere(self) -> None:
        root = Path(__file__).resolve().parent.parent
        py_exe = sys.executable
        with tempfile.TemporaryDirectory() as foreign_dir:
            test_ps1 = root / "_test_anchor.ps1"
            try:
                test_ps1.write_text(
                    f'$OutputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8\nSet-Location -Path $PSScriptRoot\n& "{py_exe}" -c "import os, pathlib; print(pathlib.Path(os.getcwd()).resolve())"\n',
                    encoding="utf-8",
                )
                res = subprocess.run(
                    ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(test_ps1)],
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=foreign_dir,
                )
                self.assertEqual(Path(res.stdout.strip()).resolve(), root.resolve())
            finally:
                if test_ps1.exists():
                    test_ps1.unlink()


if __name__ == "__main__":
    unittest.main()
