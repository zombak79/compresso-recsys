"""Command-line wrapper for building Compresso Recsys checkpoints."""

from __future__ import annotations

from compresso_recsys.builder import (
    _build_args,
    _build_recsys_checkpoint_from_args,
    build_recsys_checkpoint,
    parse_args,
)

__all__ = ["build_recsys_checkpoint", "main"]


def main() -> None:
    """Build a checkpoint from command-line arguments."""
    args = parse_args()
    path = _build_recsys_checkpoint_from_args(args)
    print(f"Saved {args.dataset} data split checkpoint to: {path}")


if __name__ == "__main__":
    main()
