"""Allow `python -m scene_understanding` execution."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
