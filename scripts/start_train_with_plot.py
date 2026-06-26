import argparse
import sys
from pathlib import Path

from PyQt5 import QtWidgets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.config import resolve_config_path  # noqa: E402
from scripts.core.training import TrainingThread  # noqa: E402
from scripts.utils.ui_train import TrainingUi  # noqa: E402


def get_parser():
    parser = argparse.ArgumentParser(description="Train PINE-TD3 with the PyQt training UI.")
    parser.add_argument(
        "-c",
        "--config",
        default="config_NH_center_SimpleMultirotor_3D",
        help="Config name under configs/ without .ini, or an absolute/relative .ini path.",
    )
    return parser


def main():
    args = get_parser().parse_args()
    config_file = resolve_config_path(args.config)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    app = QtWidgets.QApplication(sys.argv)
    gui = TrainingUi(str(config_file))
    gui.show()

    training_thread = TrainingThread(str(config_file))
    training_thread.env.action_signal.connect(gui.action_cb)
    training_thread.env.state_signal.connect(gui.state_cb)
    training_thread.env.attitude_signal.connect(gui.attitude_plot_cb)
    training_thread.env.reward_signal.connect(gui.reward_plot_cb)
    training_thread.env.pose_signal.connect(gui.traj_plot_cb)
    training_thread.start()

    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("system exit")

