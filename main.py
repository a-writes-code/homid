"""Run HomID sample generation, ID estimation, and paper-plot reproduction."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = {
    "generate": "data_knn_gen.py",
    "estimate": "ide_main.py",
}
PLOT_NOTEBOOKS = {
    "benchmark": "benchmark_plots.ipynb",
    "mechanism": "mechanism_plots.ipynb",
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate HomID samples, estimate intrinsic dimension, or reproduce paper plots.",
        epilog="Use 'generate --help', 'estimate --help', or 'plot --help' for command options.",
    )
    parser.add_argument("command", choices=(*SCRIPTS, "plot"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Options for the selected command.")
    args = parser.parse_args(argv)

    if args.command in SCRIPTS:
        return subprocess.run(
            [sys.executable, str(ROOT / "src" / SCRIPTS[args.command]), *args.arguments],
            cwd=ROOT,
        ).returncode

    plot_parser = argparse.ArgumentParser(
        prog=f"{parser.prog} plot",
        description="Reproduce the main paper figures from the bundled numerical data.",
    )
    plot_parser.add_argument(
        "--figure-group", choices=("all", *PLOT_NOTEBOOKS), default="all",
        help="Figures 2–4 (benchmark), Figures 5–7 (mechanism), or all figures (default).",
    )
    plot_args = plot_parser.parse_args(args.arguments)
    groups = PLOT_NOTEBOOKS if plot_args.figure_group == "all" else (plot_args.figure_group,)
    output = ROOT / "plots" / "generated"
    output.mkdir(parents=True, exist_ok=True)
    for group in groups:
        result = subprocess.run(
            [
                sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=300", "--output-dir", str(output),
                str(ROOT / "plots" / PLOT_NOTEBOOKS[group]),
            ],
            cwd=ROOT,
        )
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
