"""Package CLI entrypoint (delegates to package runner)."""

from __future__ import annotations

from .runner import main


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
