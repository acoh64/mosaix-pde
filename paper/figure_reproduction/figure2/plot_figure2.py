"""Plot the three Figure 2 panels from saved timing measurements."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

axis_label_size = 18
legend_size = 14
tick_size = 14


def plot_timing(json_files, labels, xlabel="Problem Size", ylabel="Solution Time (s)"):

    plt.figure(figsize=(10, 6))

    # Define consistent styling scheme
    def get_style(label):
        if "forward" in label.lower() and "float32" in label.lower():
            return "#FFC107", "o--", "Forward float32"
        elif "forward" in label.lower() and "float64" in label.lower():
            return "#FFC107", "^-", "Forward float64"
        elif "reverse" in label.lower() and "float32" in label.lower():
            return "#004D40", "o--", "Reverse float32"
        elif "reverse" in label.lower() and "float64" in label.lower():
            return "#004D40", "^-", "Reverse float64"
        else:
            return "#666666", "o-", label

    for i in range(len(json_files)):
        with open(json_files[i], "r") as f:
            solve_times = json.load(f)

        # Convert keys to integers and sort
        x_values = sorted([int(k) for k in solve_times])
        # Average repetitions after warm-up
        means = [np.mean(solve_times[str(x)][1:]) for x in x_values]

        color, fmt, display_label = get_style(labels[i])
        plt.errorbar(
            x_values,
            means,
            fmt=fmt,
            capsize=5,
            label=display_label,
            color=color,
            linewidth=2,
            markersize=6,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xlabel, fontsize=axis_label_size)
    plt.ylabel(ylabel, fontsize=axis_label_size)
    plt.tick_params(axis="both", which="major", labelsize=tick_size)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=legend_size)
    return plt.gcf()


def plot_timing_solve(
    json_files, labels, xlabel="Problem Size", ylabel="Solution Time (s)"
):

    plt.figure(figsize=(10, 6))

    # Define consistent styling scheme for device plot
    def get_device_style(label):
        if "gpu" in label.lower() and "float32" in label.lower():
            return "#D81B60", "o--", "GPU float32"
        elif "gpu" in label.lower() and "float64" in label.lower():
            return "#D81B60", "^-", "GPU float64"
        elif "cpu" in label.lower() and "float32" in label.lower():
            return "#1E88E5", "o--", "CPU float32"
        elif "cpu" in label.lower() and "float64" in label.lower():
            return "#1E88E5", "^-", "CPU float64"
        else:
            return "#666666", "o-", label

    for i in range(len(json_files)):
        with open(json_files[i], "r") as f:
            solve_times = json.load(f)

        # Convert keys to integers and sort
        x_values = sorted([int(k) for k in solve_times])
        # Average repetitions after warm-up
        means = [np.mean(solve_times[str(x)][1:]) for x in x_values]

        color, fmt, display_label = get_device_style(labels[i])
        plt.errorbar(
            [x**2 for x in x_values],
            means,
            fmt=fmt,
            capsize=5,
            label=display_label,
            color=color,
            linewidth=2,
            markersize=6,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xlabel, fontsize=axis_label_size)
    plt.ylabel(ylabel, fontsize=axis_label_size)
    plt.tick_params(axis="both", which="major", labelsize=tick_size)
    plt.xlim(100, 10000000)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=legend_size)
    return plt.gcf()


def plot_timing_timestepping(
    json_files, labels, xlabel="Problem Size", ylabel="Solution Time (s)"
):

    plt.figure(figsize=(10, 6))

    # Define consistent styling scheme for device plot
    def get_device_style(label):
        if "tsit5" in label.lower():
            return "#CC79A7", "^-", "Tsit5"
        elif "rock2jax" in label.lower():
            return "#009E73", "^-", "ROCK2"
        else:
            return "#000000", "^-", label

    for i in range(len(json_files)):
        with open(json_files[i], "r") as f:
            solve_times = json.load(f)

        # Convert keys to integers and sort
        x_values = sorted([int(k) for k in solve_times])
        # Average repetitions after warm-up
        means = [np.mean(solve_times[str(x)][1:]) for x in x_values]

        color, fmt, display_label = get_device_style(labels[i])
        plt.errorbar(
            [x**2 for x in x_values],
            means,
            fmt=fmt,
            capsize=5,
            label=display_label,
            color=color,
            linewidth=2,
            markersize=6,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xlabel, fontsize=axis_label_size)
    plt.ylabel(ylabel, fontsize=axis_label_size)
    plt.tick_params(axis="both", which="major", labelsize=tick_size)
    plt.xlim(100, 10000000)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=legend_size)
    return plt.gcf()


def plot(data_dir=None, output_dir="generated"):
    """Save panels a–c; the first measurement in each series is warm-up."""
    data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    fig = plot_timing_solve(
        [
            data_dir / "solve_times_gpu_float32.json",
            data_dir / "solve_times_gpu_float64.json",
            data_dir / "solve_times_cpu_float32.json",
            data_dir / "solve_times_cpu_float64.json",
        ],
        ["GPU float32", "GPU float64", "CPU float32", "CPU float64"],
        xlabel="Grid Points",
        ylabel="Wall Time (s)",
    )
    path = output_dir / "figure2a.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)
    fig = plot_timing(
        [
            data_dir / "solve_times_forward_log_gpu_float32.json",
            data_dir / "solve_times_forward_log_gpu_float64.json",
            data_dir / "solve_times_reverse_log_gpu_float32.json",
            data_dir / "solve_times_reverse_log_gpu_float64.json",
        ],
        ["forward float32", "forward float64", "reverse float32", "reverse float64"],
        xlabel="Number of Parameters",
        ylabel="Wall Time (s)",
    )
    path = output_dir / "figure2b.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)
    fig = plot_timing_timestepping(
        [
            data_dir / "solve_times_gpu_float64_Tsit5_cahn_hilliard.json",
            data_dir / "solve_times_gpu_float64_ROCK2JAX_cahn_hilliard.json",
            data_dir
            / "solve_times_gpu_float64_SemiImplicitFourierSpectral_cahn_hilliard.json",
        ],
        ["Tsit5", "ROCK2JAX", "Semi-implicit Fourier"],
        xlabel="Grid Points",
        ylabel="Wall Time (s)",
    )
    path = output_dir / "figure2c.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)
    return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("generated"))
    args = parser.parse_args()
    for path in plot(args.data_dir, args.output_dir):
        print(path)
