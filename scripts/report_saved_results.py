from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "results"
DEFAULT_ALGORITHM_ORDER = (
    "DDPG",
    "PPO",
    "SAC",
    "TD3",
    "PINE-SAC",
    "PINE-TD3",
)
PINE_BASELINE_PAIRS = (
    ("PINE-SAC", "SAC"),
    ("PINE-TD3", "TD3"),
)


@dataclass(frozen=True)
class Episode:
    algorithm: str
    episode_index: int
    status: str
    success: bool
    reward: float
    steps: float
    energy_j: float
    speed_mps: float
    energy_per_step_j: float
    path_distance_per_step_m: float
    goal_progress_per_step_m: float


@dataclass(frozen=True)
class MetricSpec:
    label: str
    field: str
    unit: str
    higher_is_better: bool


METRIC_SPECS = (
    MetricSpec("Reward (successful)", "reward", "", True),
    MetricSpec("Steps (successful)", "steps", "step", False),
    MetricSpec(
        "Energy Consumption (successful)", "energy_j", "J", False
    ),
    MetricSpec("Speed (successful)", "speed_mps", "m/s", True),
    MetricSpec(
        "Energy Consumption / Steps", "energy_per_step_j", "J/step", False
    ),
    MetricSpec(
        "Path distance / step", "path_distance_per_step_m", "m/step", True
    ),
    MetricSpec(
        "Goal progress / step", "goal_progress_per_step_m", "m/step", True
    ),
)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _canonical_label(label: str) -> str:
    return label.strip().upper().replace("_", "-").replace(" ", "-")


def _number(
    row: Mapping[str, str],
    *keys: str,
    default: float = math.nan,
    line_number: int | None = None,
) -> float:
    for key in keys:
        value = row.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            location = f" on CSV line {line_number}" if line_number else ""
            raise ValueError(f"Invalid numeric value for {key!r}{location}: {value!r}") from exc
    return default


def _normalise_status(row: Mapping[str, str]) -> str:
    value = row.get("status") or row.get("termination_reason") or ""
    status = value.strip().lower().replace(" ", "_")
    if status in {"crash", "crashed"}:
        return "collision"
    if status in {"time_limit", "timeout", "step_limit", "max_steps"}:
        return "step_limit_1000"
    if status:
        return status
    if _number(row, "success", default=0.0) > 0.5:
        return "success"
    if _number(row, "crash", default=0.0) > 0.5:
        return "collision"
    if _number(row, "outside", default=0.0) > 0.5:
        return "outside"
    if _number(row, "step_limit_1000", default=0.0) > 0.5:
        return "step_limit_1000"
    return "unknown"


def _load_results(input_dir: str | Path) -> dict[str, list[Episode]]:
    input_dir = _resolve_path(input_dir)
    metrics_path = input_dir / "episode_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Saved metrics file not found: {metrics_path}")

    results: dict[str, list[Episode]] = {}
    seen: set[tuple[str, int]] = set()
    with metrics_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "algorithm" not in reader.fieldnames:
            raise ValueError(
                f"{metrics_path} must contain an 'algorithm' column."
            )
        for line_number, row in enumerate(reader, start=2):
            label = (row.get("algorithm") or "").strip()
            if not label:
                raise ValueError(f"Missing algorithm on CSV line {line_number}.")
            episode_value = _number(
                row, "episode_index", line_number=line_number
            )
            if not np.isfinite(episode_value):
                raise ValueError(f"Missing episode_index on CSV line {line_number}.")
            episode_index = int(episode_value)
            key = (label, episode_index)
            if key in seen:
                raise ValueError(
                    f"Duplicate algorithm/episode pair {key!r} in {metrics_path}."
                )
            seen.add(key)

            status = _normalise_status(row)
            success_value = _number(row, "success", default=math.nan)
            success = (
                bool(success_value > 0.5)
                if np.isfinite(success_value)
                else status == "success"
            )
            steps = _number(row, "steps", line_number=line_number)
            energy = _number(
                row, "energy_consumption_j", "energy_j", line_number=line_number
            )
            energy_per_step = _number(
                row,
                "energy_consumption_per_step_j",
                "energy_per_step_j",
                line_number=line_number,
            )
            if not np.isfinite(energy_per_step) and np.isfinite(steps) and steps > 0:
                energy_per_step = energy / steps
            path_per_step = _number(
                row, "path_distance_per_step_m", line_number=line_number
            )
            if not np.isfinite(path_per_step):
                path_distance = _number(row, "path_distance_m", default=math.nan)
                if np.isfinite(path_distance) and np.isfinite(steps) and steps > 0:
                    path_per_step = path_distance / steps
            goal_per_step = _number(
                row, "goal_progress_per_step_m", line_number=line_number
            )
            if not np.isfinite(goal_per_step):
                goal_distance = _number(
                    row, "goal_progress_distance_m", default=math.nan
                )
                if np.isfinite(goal_distance) and np.isfinite(steps) and steps > 0:
                    goal_per_step = goal_distance / steps

            results.setdefault(label, []).append(
                Episode(
                    algorithm=label,
                    episode_index=episode_index,
                    status=status,
                    success=success,
                    reward=_number(row, "reward", line_number=line_number),
                    steps=steps,
                    energy_j=energy,
                    speed_mps=_number(row, "speed_mps", line_number=line_number),
                    energy_per_step_j=energy_per_step,
                    path_distance_per_step_m=path_per_step,
                    goal_progress_per_step_m=goal_per_step,
                )
            )
    for episodes in results.values():
        episodes.sort(key=lambda item: item.episode_index)
    return results


def _ordered_labels(
    results: Mapping[str, list[Episode]], requested: Sequence[str] | None
) -> list[str]:
    available = {_canonical_label(label): label for label in results}
    if requested:
        labels: list[str] = []
        for value in requested:
            match = available.get(_canonical_label(value))
            if match is None:
                choices = ", ".join(results)
                raise ValueError(
                    f"Algorithm {value!r} was not found. Available algorithms: {choices}"
                )
            if match not in labels:
                labels.append(match)
        return labels

    labels = []
    for preferred in DEFAULT_ALGORITHM_ORDER:
        match = available.get(_canonical_label(preferred))
        if match is not None:
            labels.append(match)
    labels.extend(label for label in results if label not in labels)
    return labels


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def _fmt(value: float, unit: str = "") -> str:
    if not np.isfinite(value):
        return "n/a"
    if unit == "%":
        return f"{value * 100.0:7.2f}%"
    if unit == "J":
        return f"{value:9.2f} J"
    if unit == "J/step":
        return f"{value:9.4f} J/step"
    if unit == "m/s":
        return f"{value:9.3f} m/s"
    if unit == "m/step":
        return f"{value:9.4f} m/step"
    if unit == "step":
        return f"{value:8.2f}"
    return f"{value:9.3f}"


def _fmt_stats(values: Iterable[float], unit: str = "") -> str:
    array = _finite(values)
    if array.size == 0:
        return "n/a"
    mean = float(np.mean(array))
    std = float(np.std(array))
    variance = float(np.var(array))
    return f"{_fmt(mean, unit)} +/- {_fmt(std, unit)} (var={variance:.4g})"


def _fmt_improvement(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value * 100.0:+.2f}%"


def _print_table(
    title: str, headers: Sequence[str], rows: Sequence[Sequence[str]]
) -> None:
    all_rows = [tuple(headers), *(tuple(row) for row in rows)]
    if not all_rows:
        return
    widths = [
        max(len(str(row[index])) for row in all_rows)
        for index in range(len(headers))
    ]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print(f"\n{title}")
    print(separator)
    for row_number, row in enumerate(all_rows):
        print(
            "| "
            + " | ".join(
                str(row[index]).ljust(widths[index])
                for index in range(len(headers))
            )
            + " |"
        )
        if row_number == 0:
            print(separator)
    print(separator)


def _success_rate(episodes: Sequence[Episode]) -> float:
    return float(np.mean([float(item.success) for item in episodes])) if episodes else math.nan


def _summary_rows(
    results: Mapping[str, list[Episode]], labels: Sequence[str]
) -> list[list[str]]:
    rows: list[list[str]] = [
        ["Test episodes"] + [str(len(results[label])) for label in labels],
        ["Successful episodes"]
        + [str(sum(item.success for item in results[label])) for label in labels],
        ["Reward Success Rate"]
        + [_fmt(_success_rate(results[label]), "%") for label in labels],
    ]
    successful = {
        label: [item for item in results[label] if item.success] for label in labels
    }
    for spec in METRIC_SPECS:
        rows.append(
            [spec.label]
            + [
                _fmt_stats(
                    (float(getattr(item, spec.field)) for item in successful[label]),
                    spec.unit,
                )
                for label in labels
            ]
        )
    return rows


def _status_label(status: str) -> str:
    return {
        "success": "Success",
        "collision": "Collision",
        "outside": "Outside",
        "step_limit_1000": "Step limit (1000)",
    }.get(status, status.replace("_", " ").title())


def _status_rows(
    results: Mapping[str, list[Episode]], labels: Sequence[str]
) -> list[list[str]]:
    counters = {label: Counter(item.status for item in results[label]) for label in labels}
    known = ["success", "collision", "outside", "step_limit_1000"]
    unknown = sorted(
        {
            status
            for counter in counters.values()
            for status in counter
            if status not in known
        }
    )
    statuses = known + unknown
    rows = [["Test episodes"] + [str(len(results[label])) for label in labels]]
    for status in statuses:
        rows.append(
            [_status_label(status)]
            + [str(counters[label].get(status, 0)) for label in labels]
        )
    rows.append(
        ["Non-success total"]
        + [
            str(sum(not item.success for item in results[label]))
            for label in labels
        ]
    )
    return rows


def _common_indices(
    results: Mapping[str, list[Episode]], labels: Sequence[str]
) -> tuple[list[int], list[int]]:
    if not labels:
        return [], []
    by_label = {
        label: {item.episode_index: item for item in results[label]}
        for label in labels
    }
    common = sorted(
        set.intersection(*(set(values) for values in by_label.values()))
    )
    successful = [
        index
        for index in common
        if all(by_label[label][index].success for label in labels)
    ]
    return common, successful


def _print_common_table(
    title: str,
    results: Mapping[str, list[Episode]],
    labels: Sequence[str],
) -> None:
    common, successful_indices = _common_indices(results, labels)
    by_label = {
        label: {item.episode_index: item for item in results[label]}
        for label in labels
    }
    rows = [
        ["Common test episodes"] + [str(len(common))] * len(labels),
        ["Common successful episodes"]
        + [str(len(successful_indices))] * len(labels),
        ["Common Reward Success Rate"]
        + [
            _fmt(
                len(successful_indices) / len(common) if common else math.nan,
                "%",
            )
        ]
        * len(labels),
    ]
    for spec in METRIC_SPECS:
        rows.append(
            [spec.label.replace("(successful)", "(common success)")]
            + [
                _fmt_stats(
                    (
                        float(getattr(by_label[label][index], spec.field))
                        for index in successful_indices
                    ),
                    spec.unit,
                )
                for label in labels
            ]
        )
    _print_table(title, ["Metric", *labels], rows)


def _improvement(
    baseline: float, pine: float, higher_is_better: bool
) -> float:
    if not np.isfinite(baseline) or not np.isfinite(pine):
        return math.nan
    if abs(baseline) <= 1e-12:
        return 0.0 if abs(pine) <= 1e-12 else math.nan
    delta = pine - baseline if higher_is_better else baseline - pine
    return float(delta / abs(baseline))


def _mean(values: Iterable[float]) -> float:
    array = _finite(values)
    return float(np.mean(array)) if array.size else math.nan


def _improvement_rows(
    results: Mapping[str, list[Episode]],
    pine_label: str,
    baseline_label: str,
    common_success_only: bool,
) -> list[list[str]]:
    pine_all = results[pine_label]
    baseline_all = results[baseline_label]
    if common_success_only:
        common, successful = _common_indices(
            results, [baseline_label, pine_label]
        )
        del common
        by_label = {
            label: {item.episode_index: item for item in results[label]}
            for label in (baseline_label, pine_label)
        }
        baseline_items = [by_label[baseline_label][index] for index in successful]
        pine_items = [by_label[pine_label][index] for index in successful]
        scope_label = "common successful episodes"
    else:
        baseline_items = [item for item in baseline_all if item.success]
        pine_items = [item for item in pine_all if item.success]
        scope_label = "each algorithm's own successful episodes"

    rows: list[list[str]] = [
        ["Test episodes", str(len(baseline_all)), str(len(pine_all)), "n/a"],
        [
            "Successful episodes",
            str(len(baseline_items)),
            str(len(pine_items)),
            "n/a",
        ],
    ]
    baseline_rate = _success_rate(baseline_all)
    pine_rate = _success_rate(pine_all)
    rate_label = "Reward Success Rate (all test episodes)"
    rate_improvement = _improvement(baseline_rate, pine_rate, True)
    if common_success_only:
        common, successful = _common_indices(
            results, [baseline_label, pine_label]
        )
        baseline_rate = (
            len(successful) / len(common) if common else math.nan
        )
        pine_rate = baseline_rate
        rate_label = "Common Reward Success Rate"
        rate_improvement = 0.0 if np.isfinite(baseline_rate) else math.nan
    rows.append(
        [
            rate_label,
            _fmt(baseline_rate, "%"),
            _fmt(pine_rate, "%"),
            _fmt_improvement(rate_improvement),
        ]
    )
    for spec in METRIC_SPECS:
        baseline_values = [float(getattr(item, spec.field)) for item in baseline_items]
        pine_values = [float(getattr(item, spec.field)) for item in pine_items]
        baseline_mean = _mean(baseline_values)
        pine_mean = _mean(pine_values)
        rows.append(
            [
                (
                    spec.label.replace("(successful)", "(common success)")
                    if common_success_only
                    else spec.label
                ),
                _fmt_stats(baseline_values, spec.unit),
                _fmt_stats(pine_values, spec.unit),
                _fmt_improvement(
                    _improvement(
                        baseline_mean, pine_mean, spec.higher_is_better
                    )
                ),
            ]
        )
    rows.insert(
        0,
        [
            f"Scope: {scope_label}",
            baseline_label,
            pine_label,
            "PINE improvement",
        ],
    )
    return rows


def _print_improvement_tables(
    results: Mapping[str, list[Episode]], labels: Sequence[str]
) -> None:
    available = {_canonical_label(label): label for label in labels}
    for pine_expected, baseline_expected in PINE_BASELINE_PAIRS:
        pine_label = available.get(_canonical_label(pine_expected))
        baseline_label = available.get(_canonical_label(baseline_expected))
        if pine_label is None or baseline_label is None:
            print(
                f"\nPINE improvement: {pine_expected} vs {baseline_expected} "
                "(skipped: both algorithms are not available in the selected data)"
            )
            continue
        pair_name = f"{pine_label} relative to {baseline_label}"
        _print_table(
            f"{pair_name} - own successful episodes",
            ["Metric", baseline_label, pine_label, "PINE improvement"],
            _improvement_rows(
                results, pine_label, baseline_label, common_success_only=False
            ),
        )
        _print_table(
            f"{pair_name} - common successful episodes",
            ["Metric", baseline_label, pine_label, "PINE improvement"],
            _improvement_rows(
                results, pine_label, baseline_label, common_success_only=True
            ),
        )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print detailed tables from saved model-test results."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing episode_metrics.csv.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        help="Optional algorithm names; omit to report all saved algorithms.",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()
    results = _load_results(args.input_dir)
    labels = _ordered_labels(results, args.algorithms)
    if not labels:
        raise ValueError("No algorithms found in the saved metrics file.")

    input_dir = _resolve_path(args.input_dir)
    print("Saved-result report")
    print(f"  input       : {input_dir / 'episode_metrics.csv'}")
    print("  algorithms  : " + ", ".join(labels))

    _print_table(
        "Episode status summary",
        ["Status", *labels],
        _status_rows(results, labels),
    )
    _print_table(
        "All successful episodes (each algorithm's own successes)",
        ["Metric", *labels],
        _summary_rows(results, labels),
    )
    has_ppo = any(_canonical_label(label) == "PPO" for label in labels)
    if has_ppo:
        _print_common_table(
            "Common successful episodes (including PPO)",
            results,
            labels,
        )
        without_ppo = [
            label for label in labels if _canonical_label(label) != "PPO"
        ]
        if without_ppo:
            _print_common_table(
                "Common successful episodes (excluding PPO)",
                results,
                without_ppo,
            )
    else:
        _print_common_table(
            "Common successful episodes (selected algorithms)",
            results,
            labels,
        )
    _print_improvement_tables(results, labels)


if __name__ == "__main__":
    main()
