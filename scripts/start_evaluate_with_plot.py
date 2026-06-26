import argparse
import sys
from pathlib import Path

from PyQt5 import QtWidgets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.evaluation import EvaluateThread  # noqa: E402
from scripts.utils.ui_train import TrainingUi  # noqa: E402


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate PINE-TD3 with the PyQt trajectory UI.")
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

    app = QtWidgets.QApplication(sys.argv)
    gui = TrainingUi(str(config_file))
    gui.show()

    evaluate_thread = EvaluateThread(run_dir, config_file, model_file, args.episodes)
    evaluate_thread.env.action_signal.connect(gui.action_cb)
    evaluate_thread.env.state_signal.connect(gui.state_cb)
    evaluate_thread.env.attitude_signal.connect(gui.attitude_plot_cb)
    evaluate_thread.env.reward_signal.connect(gui.reward_plot_cb)
    evaluate_thread.env.pose_signal.connect(gui.traj_plot_cb)
    evaluate_thread.start()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

