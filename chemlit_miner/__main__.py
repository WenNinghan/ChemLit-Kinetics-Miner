"""Module entrypoint for `python -m chemlit_miner`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
