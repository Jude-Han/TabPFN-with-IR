"""Local environment loading without exposing secret values."""

from __future__ import annotations

from pathlib import Path


def load_project_dotenv(
    dotenv_path: str | Path | None = None,
    *,
    override: bool = False,
) -> Path | None:
    """Load a project ``.env`` file and return its path when one was found.

    Existing process variables take precedence by default. This function never
    reads, returns, or logs individual secret values.
    """

    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError as exc:  # pragma: no cover - optional benchmark dependency
        raise ImportError(
            "Environment-file support requires python-dotenv. Install with "
            "`python -m pip install -e '.[benchmark]'`."
        ) from exc

    if dotenv_path is None:
        discovered = find_dotenv(filename=".env", usecwd=True)
        if not discovered:
            return None
        resolved_path = Path(discovered).resolve()
    else:
        resolved_path = Path(dotenv_path).expanduser().resolve()
        if not resolved_path.is_file():
            return None

    load_dotenv(dotenv_path=resolved_path, override=override)
    return resolved_path
