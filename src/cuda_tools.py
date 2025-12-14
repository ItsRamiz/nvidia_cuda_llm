from pathlib import Path
from langchain.tools import tool
import subprocess
from models import AgentState

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
BUILD_SCRIPT = CODE_DIR / "build.bat"

def has_compiled_executables() -> int:
    """
    Checks whether the project root /code directory contains any .exe files.

    Returns:
        1 if at least one .exe file is found
        0 otherwise
    """
    return 1 if any(CODE_DIR.rglob("*.exe")) else 0


def run_build_script(state: AgentState) -> AgentState:
    """
    Runs the Windows build script located at project_root/code/build.bat in order
    to compile the CUDA code into .exe files and be able to run the benchmarks.

    Returns:
        1 On success, 0 on failure.
    """
    try:
        result = subprocess.run(
            ["cmd.exe", "/c", str(BUILD_SCRIPT)],
            cwd=str(CODE_DIR),
            capture_output=True,
            text=True,
            check=False
        )

        output = result.stdout.strip()
        error = result.stderr.strip()

        if result.returncode != 0:
            state["is_build_benchmark"] = False
            return state

        state["is_build_benchmark"] = True
        return state

    except Exception as e:
        return f"EXCEPTION: {str(e)}"