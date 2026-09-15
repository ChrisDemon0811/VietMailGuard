"""Launch the VietMailGuard Mail Streamlit application.

Usage:
    python run.py
    python run.py --server.port 8502
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
APP_PATH = PROJECT_ROOT / "app" / "app.py"


def build_command(extra_arguments: list[str] | None = None) -> list[str]:
    """Build a path-safe Streamlit command using the active interpreter."""

    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        *(extra_arguments or []),
    ]


def display_port(arguments: list[str]) -> str:
    """Read Streamlit's optional port argument for the startup hint."""

    for index, argument in enumerate(arguments):
        if argument == "--server.port" and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith("--server.port="):
            return argument.partition("=")[2]
    return "8501"


def main() -> int:
    """Validate the local setup and run VietMailGuard Mail."""

    if not APP_PATH.is_file():
        print(f"Không tìm thấy Streamlit app: {APP_PATH}", file=sys.stderr)
        return 1
    if importlib.util.find_spec("streamlit") is None:
        print(
            "Chưa cài Streamlit. Hãy chạy: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    extra_arguments = sys.argv[1:]
    print("Đang khởi động VietMailGuard Mail...")
    print(f"Sau khi server sẵn sàng, mở http://localhost:{display_port(extra_arguments)}")
    try:
        completed = subprocess.run(
            build_command(extra_arguments),
            cwd=PROJECT_ROOT,
            check=False,
        )
    except KeyboardInterrupt:
        return 0
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
