from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import run


def test_launcher_uses_active_python_and_absolute_app_path() -> None:
    command = run.build_command(["--server.port", "8502"])

    assert command[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert Path(command[4]).is_absolute()
    assert Path(command[4]) == run.APP_PATH
    assert command[5:] == ["--server.port", "8502"]
    assert run.APP_PATH.is_file()


def test_launcher_displays_configured_or_default_port() -> None:
    assert run.display_port([]) == "8501"
    assert run.display_port(["--server.port", "8502"]) == "8502"
    assert run.display_port(["--server.port=8503"]) == "8503"


def test_redirected_launcher_and_child_write_utf8(tmp_path: Path) -> None:
    """Exercise real pipes with an incompatible inherited encoding, offline."""
    package = tmp_path / "streamlit"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    message = "Tiếng Việt: khởi động ứng dụng — 中文"
    (package / "__main__.py").write_text(
        "import os, sys\n"
        "assert os.environ['PYTHONIOENCODING'] == 'utf-8'\n"
        "assert os.environ['PYTHONUTF8'] == '1'\n"
        f"print({message!r})\n"
        f"print({message!r}, file=sys.stderr)\n",
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(tmp_path), os.environ.get("PYTHONPATH", "")]
        ),
        "PYTHONIOENCODING": "cp1252:strict",
        "PYTHONUTF8": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [sys.executable, str(run.PROJECT_ROOT / "run.py")],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0
    assert "Đang khởi động VietMailGuard Mail" in completed.stdout.decode("utf-8")
    assert message in completed.stdout.decode("utf-8")
    assert message in completed.stderr.decode("utf-8")
    assert b"UnicodeEncodeError" not in completed.stderr


def test_redirected_setup_error_is_utf8(tmp_path: Path) -> None:
    """The early stderr error must also work before Streamlit is launched."""
    missing = tmp_path / "ứng dụng không tồn tại.py"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import run; "
            f"run.APP_PATH = Path({str(missing)!r}); raise SystemExit(run.main())",
        ],
        cwd=run.PROJECT_ROOT,
        env={
            **os.environ,
            "PYTHONIOENCODING": "cp1252:strict",
            "PYTHONUTF8": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 1
    assert "Không tìm thấy Streamlit app" in completed.stderr.decode("utf-8")
    assert str(missing) in completed.stderr.decode("utf-8")
    assert b"UnicodeEncodeError" not in completed.stderr
