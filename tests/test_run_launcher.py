from __future__ import annotations

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
