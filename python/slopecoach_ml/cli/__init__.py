from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    """Invoke the CLI without importing ``__main__`` during module execution."""
    from .__main__ import main as command_main

    return command_main(argv)


__all__ = ["main"]
