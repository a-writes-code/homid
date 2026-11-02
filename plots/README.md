# Reproduce the main paper plots

These notebooks reproduce Figures 2–7 of [HomID, arXiv:2510.01335v2](https://arxiv.org/abs/2510.01335v2). They read the compact archives and manifests in [`data/`](../data/README.md). Six released PDF snapshots are included directly in this folder. Normal plotting needs no original experiment files and runs no ID-estimation sweeps.

## Run from the terminal

Install the dependencies as described in the [root README](../README.md). Activate that environment. From the repository root, run:

```bash
python main.py plot
```

The command uses the current Python environment to execute both notebooks with nbconvert. It saves regenerated PDFs and executed notebook copies in `plots/generated/`. These temporary files are ignored by Git. The six released PDFs and source notebooks stay unchanged.

To run one group:

```bash
python main.py plot --figure-group benchmark
python main.py plot --figure-group mechanism
```

## Run interactively

From the repository root:

```bash
python -m notebook plots/benchmark_plots.ipynb
python -m notebook plots/mechanism_plots.ipynb
```

Select **Run All Cells**. Both notebooks also support `plots/` as the current directory. Keep the optional rebuild flags disabled for normal reproduction.

## Released figures and regenerated outputs

The links below open the six tracked PDF snapshots in this folder. Reruns save separate copies with the same filenames under `plots/generated/`. Executed notebook copies and compression round-trip files also stay in that ignored directory.

| Figure | Notebook | Released PDF |
| --- | --- | --- |
| 2 | [benchmark_plots.ipynb](benchmark_plots.ipynb) | [new_compare_rel_sphere_only.pdf](new_compare_rel_sphere_only.pdf) |
| 3 | [benchmark_plots.ipynb](benchmark_plots.ipynb) | [score_vs_epsilon_sub.pdf](score_vs_epsilon_sub.pdf) |
| 4 | [benchmark_plots.ipynb](benchmark_plots.ipynb) | [noise_tre_scaling_rel_sub.pdf](noise_tre_scaling_rel_sub.pdf) |
| 5 | [mechanism_plots.ipynb](mechanism_plots.ipynb) | [mle_squeeze_noise.pdf](mle_squeeze_noise.pdf) |
| 6 | [mechanism_plots.ipynb](mechanism_plots.ipynb) | [lpca_interp_mt.pdf](lpca_interp_mt.pdf) |
| 7 | [mechanism_plots.ipynb](mechanism_plots.ipynb) | [abid_interp.pdf](abid_interp.pdf) |

Figure 1 is a conceptual overview. It has no numerical reproduction here.

## Data and compression

Each notebook includes the optional code used to select and compress its numerical inputs. Rebuilding requires the original external result files. It writes the compressed arrays and manifests to `data/`. Both exporters reject a destination inside the original source directory.

See [data/README.md](../data/README.md) for source dates, retained fields, missing inputs, aggregation rules, uncertainty definitions, and rebuild instructions. Compression preserves the retained arrays without rounding or reducing their precision. The original scientific source files remain unchanged in `src/`.
