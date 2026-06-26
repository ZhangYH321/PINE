import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.evaluation import EvaluationRunner  # noqa: E402


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate a trained PINE-TD3 model.")
    parser.add_argument("--run-dir", required=True, help="Training run directory under logs/ or an absolute path.")
    parser.add_argument("--model", default="models/model_sb3.zip", help="Model file relative to run-dir or absolute path.")
    parser.add_argument("--config", default="config/config.ini", help="Config file relative to run-dir or absolute path.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes.")
    return parser


def resolve_run_file(run_dir: Path, file_arg: str) -> Path:
    path = Path(file_arg)
    return path if path.is_absolute() else run_dir / path


def main():
    args = get_parser().parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    run_dir = run_dir.resolve()

    config_file = resolve_run_file(run_dir, args.config).resolve()
    model_file = resolve_run_file(run_dir, args.model).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    result = EvaluationRunner(run_dir, config_file, model_file, args.episodes).run()
    print(
        "evaluation finished: "
        f"avg_reward={result.average_reward:.3f}, "
        f"success_rate={result.success_rate:.3f}, "
        f"crash_rate={result.crash_rate:.3f}, "
        f"avg_success_steps={result.average_success_steps:.1f}"
    )
    print(f"results saved to: {result.output_dir}")


if __name__ == "__main__":
    main()

