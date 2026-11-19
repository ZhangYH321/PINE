from __future__ import annotations

import ast
from configparser import ConfigParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_config_path(config_arg: str) -> Path:
    config_path = Path(config_arg)
    if config_path.suffix.lower() != ".ini":
        config_path = PROJECT_ROOT / "configs" / f"{config_arg}.ini"
    elif not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    return config_path.resolve()


def load_config(config_file: str | Path) -> ConfigParser:
    cfg = ConfigParser()
    if not cfg.read(str(config_file), encoding="utf-8"):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return cfg


def parse_net_arch(cfg: ConfigParser):
    return ast.literal_eval(cfg.get("options", "net_arch"))
