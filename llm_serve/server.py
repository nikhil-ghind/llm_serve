"""CLI entry point: ``python -m llm_serve.server --config configs/serving.yaml``."""

from __future__ import annotations

import argparse
import logging
import sys

from .backends.base import available_backends
from .config import ConfigError, build_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm_serve.server",
        description="Serve Mistral 7B (QLoRA) over an OpenAI-compatible API.",
    )
    parser.add_argument("--config", default=None, help="path to a YAML config file")
    parser.add_argument(
        "--backend",
        default=None,
        choices=available_backends(),
        help="override backend.kind",
    )
    parser.add_argument("--host", default=None, help="override server.host")
    parser.add_argument("--port", type=int, default=None, help="override server.port")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.OPTION=VALUE",
        help="ad-hoc config override; repeatable",
    )
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="resolve the configuration, print it and exit without serving",
    )
    return parser


def resolve_config(argv: list[str] | None = None):
    """Parse CLI args into an effective :class:`~llm_serve.config.Config`."""
    args = build_parser().parse_args(argv)
    overrides = list(args.overrides)
    if args.backend:
        overrides.append(f"backend.kind={args.backend}")
    if args.host:
        overrides.append(f"server.host={args.host}")
    if args.port:
        overrides.append(f"server.port={args.port}")
    return args, build_config(args.config, cli_overrides=overrides)


def main(argv: list[str] | None = None) -> int:
    try:
        args, config = resolve_config(argv)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.print_config:
        import json

        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        return 0

    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is not installed; run: pip install 'uvicorn[standard]' fastapi",
            file=sys.stderr,
        )
        return 3

    from .api.app import create_app

    logging.getLogger("llm_serve").info(
        "serving %s via %s on %s:%d",
        config.model.name,
        config.backend.kind,
        config.server.host,
        config.server.port,
    )
    uvicorn.run(
        create_app(config),
        host=config.server.host,
        port=config.server.port,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
