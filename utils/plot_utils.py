"""Plotting utilities for FL simulation results.

This module provides functions to load JSON result files produced by
the simulation and plot (1) loss over rounds and (2) centralized accuracy.

Usage example:
    from utils.plot_utils import plot_from_files
    plot_from_files({"FMNIST": "outputs/results_FMNIST.json", "MNIST": "outputs/results_MNIST.json", "CIFAR10": "outputs/results_CIFAR10.json"})
"""
from pathlib import Path
import json
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np


def _make_serializable(obj: Any) -> Any:
    """Recursively convert common non-JSON types (numpy arrays, etc.) into JSON serializable types."""
    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, float)):
        # Convert numpy scalars to native
        try:
            return obj.item()
        except Exception:
            return obj
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    # numpy types
    try:
        import numpy as _np

        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, _np.generic):
            return obj.item()
    except Exception:
        pass
    # Fallback to string
    return str(obj)


def _load_json(path: str) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_series(data: Any, key_candidates: List[str]) -> List[float]:
    """Try to extract a numeric series from JSON data using several heuristics.

    The function supports several common shapes:
    - dict with direct list: {"loss": [..]}
    - dict with history/metrics: {"history": {"loss": [..]}}
    - list of per-round dicts: [{"loss": .., "accuracy": ..}, ...]
    - nested server entries: {"server": [{...}, ...]}
    Returns an empty list if nothing found.
    """
    # If direct dict with candidates
    if isinstance(data, dict):
        # common containers
        for prefix in (None, "history", "metrics", "server", "results"):
            container = data if prefix is None else data.get(prefix)
            if isinstance(container, dict):
                for key in key_candidates:
                    if key in container and isinstance(container[key], list):
                        return [_to_float_safe(x) for x in container[key]]
            # container can be list of dicts
            if isinstance(container, list):
                series = []
                for item in container:
                    if not isinstance(item, dict):
                        continue
                    # search nested keys
                    found = False
                    for key in key_candidates:
                        if key in item:
                            series.append(_to_float_safe(item[key]))
                            found = True
                            break
                    if not found:
                        # try nested metrics
                        for key in ("metrics", "result"):
                            if key in item and isinstance(item[key], dict):
                                for c in key_candidates:
                                    if c in item[key]:
                                        series.append(_to_float_safe(item[key][c]))
                                        found = True
                                        break
                    # continue regardless
                if series:
                    return series
        # fallback: check if keys exist at top level as scalars per round index
        for key in key_candidates:
            if key in data and isinstance(data[key], list):
                return [_to_float_safe(x) for x in data[key]]

    # If data is a list of dicts
    if isinstance(data, list):
        series = []
        for item in data:
            if not isinstance(item, dict):
                continue
            for key in key_candidates:
                if key in item:
                    series.append(_to_float_safe(item[key]))
                    break
            else:
                # try metrics nested
                if "metrics" in item and isinstance(item["metrics"], dict):
                    for key in key_candidates:
                        if key in item["metrics"]:
                            series.append(_to_float_safe(item["metrics"][key]))
                            break
        if series:
            return series

    return []


def _to_float_safe(v: Any) -> float:
    try:
        import numpy as _np

        if isinstance(v, _np.generic):
            return float(v.item())
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return float("nan")


def plot_loss(json_paths: Dict[str, str], out_path: str = "outputs/loss.png") -> None:
    """Plot loss curves from multiple JSON result files.

    Args:
        json_paths: mapping label->json filepath
        out_path: output image filepath
    """
    plt.figure(figsize=(8, 5))
    for label, path in json_paths.items():
        try:
            data = _load_json(path)
            series = _extract_series(data, ["loss", "train_loss", "client_loss"])
            if not series:
                print(f"警告: 在 {path} 中未找到损失序列，跳过")
                continue
            plt.plot(range(1, len(series) + 1), series, marker="o", label=label)
        except Exception as e:
            print(f"无法加载 {path}: {e}")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Loss over rounds")
    plt.legend()
    plt.grid(True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"已保存损失图: {out_path}")


def plot_centralized_accuracy(json_paths: Dict[str, str], out_path: str = "outputs/centralized_accuracy.png") -> None:
    """Plot centralized accuracy from multiple JSON result files.

    Args:
        json_paths: mapping label->json filepath
        out_path: output image filepath
    """
    plt.figure(figsize=(8, 5))
    for label, path in json_paths.items():
        try:
            data = _load_json(path)
            series = _extract_series(data, ["centralized_accuracy", "centralized_acc", "accuracy", "test_accuracy", "val_accuracy"])
            if not series:
                print(f"警告: 在 {path} 中未找到准确率序列，跳过")
                continue
            plt.plot(range(1, len(series) + 1), series, marker="o", label=label)
        except Exception as e:
            print(f"无法加载 {path}: {e}")
    plt.xlabel("Round")
    plt.ylabel("Centralized Accuracy")
    plt.title("Centralized Accuracy over rounds")
    plt.legend()
    plt.grid(True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"已保存准确率图: {out_path}")


def plot_from_files(json_paths: Dict[str, str], out_dir: str = "outputs") -> None:
    """Convenience function to plot both loss and centralized accuracy.

    Args:
        json_paths: mapping label->json filepath
        out_dir: directory where plots will be saved
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_loss(json_paths, out_path=str(out_dir / "loss.png"))
    plot_centralized_accuracy(json_paths, out_path=str(out_dir / "centralized_accuracy.png"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot FL simulation results from JSON files.")
    parser.add_argument("files", nargs="+", help="JSON result files (expect one per dataset)")
    parser.add_argument("--labels", nargs="*", help="Optional labels matching files")
    parser.add_argument("--out", default="outputs", help="Output directory for plots")
    args = parser.parse_args()
    files = args.files
    if args.labels and len(args.labels) == len(files):
        mapping = {lab: fp for lab, fp in zip(args.labels, files)}
    else:
        # derive labels from filenames
        mapping = {Path(fp).stem: fp for fp in files}
    plot_from_files(mapping, out_dir=args.out)
