# Main-paper plot data

These files reproduce the numerical main-text plots in [HomID, arXiv:2510.01335v2](https://arxiv.org/abs/2510.01335v2). They contain the plotted quantities and the seed summaries used to calculate them. The full point clouds, neighbor arrays, unused diagnostics, and appendix experiments are excluded.

The two compressed archives total about **0.44 MiB**. Each is a standard NumPy `.npz` file. Load it with `numpy.load(path, allow_pickle=False)`. No Git LFS or external data download is needed to draw the plots.

## Files and figures

| Paper figure | Numerical content retained | Notebook | Released PDF |
| --- | --- | --- | --- |
| 2 | Mean absolute relative errors for HomID and matched spheres, by family, estimator, and seed | [benchmark_plots.ipynb](../plots/benchmark_plots.ipynb) | [new_compare_rel_sphere_only.pdf](../plots/new_compare_rel_sphere_only.pdf) |
| 3 | Signed relative-error means and local standard deviations for each diagonal distortion and seed | [benchmark_plots.ipynb](../plots/benchmark_plots.ipynb) | [score_vs_epsilon_sub.pdf](../plots/score_vs_epsilon_sub.pdf) |
| 4 | Signed relative-error means and local standard deviations for each noise variance, covariance type, and seed | [benchmark_plots.ipynb](../plots/benchmark_plots.ipynb) | [noise_tre_scaling_rel_sub.pdf](../plots/noise_tre_scaling_rel_sub.pdf) |
| 5 | Stored MNIST class means and standard errors for MLE under distortion and noise | [mechanism_plots.ipynb](../plots/mechanism_plots.ipynb) | [mle_squeeze_noise.pdf](../plots/mle_squeeze_noise.pdf) |
| 6 | Every plotted anchor's local covariance eigenvalues at the eight plotted neighborhood sizes | [mechanism_plots.ipynb](../plots/mechanism_plots.ipynb) | [lpca_interp_mt.pdf](../plots/lpca_interp_mt.pdf) |
| 7 | Cosine histogram bin locations, bin probabilities, and plotted skewness values | [mechanism_plots.ipynb](../plots/mechanism_plots.ipynb) | [abid_interp.pdf](../plots/abid_interp.pdf) |

The six released PDF snapshots are tracked directly in `plots/`. Reruns write separate copies with the same names under the ignored `plots/generated/` directory.

Figure 1 is a conceptual overview. It has no numerical reproduction in this folder.

- `benchmark_data.npz` contains Figures 2–4. The corresponding `benchmark_manifest.json` records array shapes, source selection, source file dates and hashes, and aggregation rules.
- `mechanism_data.npz` contains Figures 5–7. The corresponding `mechanism_manifest.json` records retained fields, source dates and hashes, the selected neighborhood sizes, and the Figure 7 derivation.
- The notebooks in [`plots/`](../plots/README.md) contain the plotting code and optional export/compression code. Outputs and execution counts are cleared in the release notebooks.

## Draw the plots

Install the root `requirements.txt` in a fresh Python 3.11 environment. From the repository root, run:

```bash
python main.py plot
```

This executes both notebooks in `plots/` with nbconvert and the current Python environment. The notebooks read the archives and manifests in `data/`. They write regenerated PDFs and executed notebook copies into `plots/generated/`. That directory is ignored by Git. The six released PDF snapshots and source notebooks stay unchanged. The optional rebuild sections remain disabled.

To draw only Figures 2–4 or Figures 5–7:

```bash
python main.py plot --figure-group benchmark
python main.py plot --figure-group mechanism
```

For interactive use:

```bash
python -m notebook plots/benchmark_plots.ipynb
python -m notebook plots/mechanism_plots.ipynb
```

Select **Run All Cells**. The notebooks support the repository root or `plots/` as the current directory. See [plots/README.md](../plots/README.md) for the plotting workflow.

## Source versions and provenance

The original results live in the author's `local_ideb` project. The export follows the latest plotting notebooks that match the main manuscript figures:

| Source notebook | Last modified, UTC | Main figures |
| --- | --- | --- |
| `plotting_2.ipynb` | 2 August 2026 | 2 and 4 |
| `review_plots.ipynb` | 25 July 2026 | 3 |
| `iclr_review/vis.ipynb` | 7 August 2026 | 5–7 |

The exporter follows each notebook's file selection and fallback order. It does not choose an unrelated experiment merely because that file is newer. Exact input paths, modification times, and SHA-256 checksums are recorded in the manifests. Paths are relative to the original project. That project is not needed for normal plot reproduction.

For Figure 2, the matched-sphere scores use the original `compare_review` data selection. For Figures 3–4, the data follows the distortion and noise grids selected by the source plotting cells. For Figure 5, the stored MNIST summaries are copied directly. For Figure 6, unplotted neighborhood sizes and unused diagnostics are removed.

**Figure 7 is a derived summary.** The original notebook calculated the histograms in memory and saved a PDF. It did not save a numerical histogram cache. The export therefore reruns the original histogram functions on the recorded source samples. It uses the original seed, 2,000 points, neighborhood size, distortions, and bins. It retains only the histogram and skewness values used in the plot. The full cosine arrays are discarded. This derivation is recorded explicitly in `mechanism_manifest.json`.

The original source files are read only. Their code and data are not changed by the export.

## Aggregation and uncertainty

The archives preserve the original plot aggregation. They do not standardize the different uncertainty measures:

- Figure 2 uses three estimation seeds. Each seed first averages absolute relative errors over the selected sweeps and manifold instantiations. The plotted score is the HomID error divided by the sphere error, minus one. The annotation uncertainty is the sum of the two across-seed population standard deviations, as in the original plotting code. It is not a propagated uncertainty of the ratio.
- Figures 3–4 retain the three per-seed signed-error means and local standard deviations. Shaded bands use the average within-seed spread of pointwise estimates. They are not confidence intervals for the mean.
- Figure 5 uses the standard errors already stored in the original result dictionaries.
- Figure 6 retains all 50 plotted anchors for each of the three datasets and eight selected neighborhood sizes.
- Figure 7 stores bin probabilities and the skewness values used in the legend.

One Figure 2 input is missing in the original selection: the Flag/TwoNN/HomID value for seed 42. It remains `NaN`. The original `nanmean` aggregation excludes that seed. The missing value is not filled in.

Figure 3's axis labels were swapped in the source plotting cell. The release notebook labels the horizontal axis as distortion `epsilon` and the vertical axis as signed relative error `delta`. Numerical values are unchanged. Plot styling otherwise follows the original figures. Figure 5 preserves the original legend placement.

## Rebuild the compressed archives

Normal reproduction does not read the original experiment directory. The export code is included so that the selection and compression can be inspected and repeated.

For Figures 2–4:

1. Open [`plots/benchmark_plots.ipynb`](../plots/benchmark_plots.ipynb).
2. Read the optional export section.
3. Set `ORIGINAL_ROOT` to the local `local_ideb` directory.
4. Set `REBUILD_FROM_ORIGINALS = True` and run that section.

For Figures 5–7:

1. Open [`plots/mechanism_plots.ipynb`](../plots/mechanism_plots.ipynb).
2. Read the optional export section.
3. Set `ORIGINAL_RESULTS_ROOT` to the local `local_ideb` directory.
4. Run the optional export call described there.

The rebuild writes the public archives and manifests in `data/`. It can take longer than plotting because it reads the original result files and recomputes Figure 7's histograms. Both exporters reject a destination inside the original results directory.

The preparation has two separate steps:

1. **Select and summarize.** Retain exactly the quantities needed by the plots, with per-seed values where used. This intentionally discards raw data that the plots do not need.
2. **Compress without further loss.** Write the selected arrays with `numpy.savez_compressed`. Do not round values or reduce their floating-point precision.

Compression is lossless for the retained arrays. The archives are not full copies of the original experiment datasets. Notebook checks verify compression round trips and the expected figure outputs. Archive checksums are stored in the manifests.
