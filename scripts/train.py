import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.config import resolve_config_path
from scripts.core.training import TrainingRunner


def get_parser():
    parser = argparse.ArgumentParser(description="Train PINE-TD3 without the PyQt UI.")
    parser.add_argument(
        "-c",
        "--config",
        default="config_NH_center_SimpleMultirotor_3D",
        help="Config name under configs/ without .ini, or an absolute/relative .ini path.",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb for this run.")
    return parser


def main():
    args = get_parser().parse_args()
    config_file = resolve_config_path(args.config)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    runner = TrainingRunner(config_file, use_wandb=False if args.no_wandb else None)

    try:
        runner.run()
    finally:
        if runner.cfg.getboolean("options", "use_wandb", fallback=False):
            import wandb

            wandb.finish()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("system exit")
