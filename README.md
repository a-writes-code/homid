# HomID: Homogeneous Spaces for Intrinsic Dimension Estimation

HomID provides synthetic manifolds for testing intrinsic dimension estimators. Intrinsic dimension (ID) is the number of local degrees of freedom in a dataset. An embedding can place that dataset in a much larger Euclidean space. HomID uses homogeneous spaces with anisotropic embeddings to test how this geometry affects ID estimates.

This repository contains the original manifold generators, data storage utilities, nearest-neighbor utilities, and ID estimation routines in `src/`. The `data/` folder contains compact numerical inputs for the main paper plots. The `plots/` folder contains six released PDF snapshots and the plotting and compression notebooks. Use `main.py` to generate samples, run ID estimation, or reproduce the plots.

**Paper:** Aritra Das, Joseph T. Iosue, and Victor V. Albert, [*HomID: Benchmarking Intrinsic Dimension Estimators on Homogeneous Manifolds with Anisotropic Embeddings*, arXiv:2510.01335v2](https://arxiv.org/abs/2510.01335v2). [Read version 2 in HTML](https://arxiv.org/html/2510.01335v2).

## Contents

```text
homid/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── main.py                      # Generate, estimate, or reproduce plots
├── src/                         # Original scientific code, unchanged
│   ├── manifolds.py             # Sample the supported manifolds
│   ├── data_knn_gen.py          # Original sample-generation CLI
│   ├── data_utils.py            # Save and load samples, kNN arrays, and estimates
│   ├── knn.py                   # Euclidean nearest-neighbor search
│   ├── ide.py                   # ID estimators and local PCA utilities
│   └── ide_main.py              # Original ID-estimation CLI
├── data/
│   ├── README.md                # Figure mapping and data provenance
│   ├── benchmark_data.npz       # Numerical inputs for Figures 2–4
│   ├── benchmark_manifest.json
│   ├── mechanism_data.npz       # Numerical inputs for Figures 5–7
│   └── mechanism_manifest.json
└── plots/
    ├── README.md                # Plotting commands and output files
    ├── benchmark_plots.ipynb     # Figures 2–4 and optional compression code
    ├── mechanism_plots.ipynb     # Figures 5–7 and optional compression code
    ├── new_compare_rel_sphere_only.pdf
    ├── score_vs_epsilon_sub.pdf
    ├── noise_tre_scaling_rel_sub.pdf
    ├── mle_squeeze_noise.pdf
    ├── lpca_interp_mt.pdf
    └── abid_interp.pdf
```

## Environment setup

Use Python 3.11 and a CPU. A GPU, MPS, and a cluster are not required. Run these commands in a terminal:

```bash
git clone https://github.com/a-writes-code/homid.git
cd homid
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Check the installation:

```bash
python -m pip check
python -c "import numpy, scipy, sklearn, skdim, numba, pandas, h5py, matplotlib; print('Imports OK')"
python main.py generate --help
python main.py estimate --help
```

Use a fresh environment if an existing installation has incompatible package versions. The requirements include `numba` and `pandas` because scikit-dimension imports need them. The scikit-dimension version is fixed to the API used by the original code.

**Run all commands below from the repository root.** The `main.py` wrapper resolves input and output paths from that root. Direct source scripts and Python API calls resolve paths from the current working directory.

`main.py generate` and `main.py estimate` pass their arguments to the original scripts in `src/`. The scientific code retains its original contents. The direct entry points also work from the repository root:

```bash
python src/data_knn_gen.py --help
python src/ide_main.py --help
```

For Python API calls, put the original flat modules on the import path with `PYTHONPATH=src`. The examples below use this form. To use the API interactively, start `PYTHONPATH=src python` from the repository root.

## Quick test: generate data, estimate ID, and inspect the result

This test uses 100 points: 1% of the generator's default sample count. It samples a 10-dimensional linear space embedded in 20 dimensions. It then runs local principal component analysis (lPCA) and saves one estimate per point.

The generation CLI always creates 10,000 points. For this small test, call the existing generator directly with `N=100`:

```bash
PYTHONPATH=src python - <<'PY'
from manifolds import data_gen_lin_nsp
from data_utils import save_info

params = (20, 10, 42)
info = data_gen_lin_nsp(N=100, params=params)
save_info(info_dict=info, params=params, manifold_name="lin_nsp")
print("Saved", info["samples"].shape, "samples")
PY
```

Expected sample file:

```text
data/samples/_lin_nsp_p_(20, 10, 42).h5
```

Run the estimator:

```bash
python main.py estimate \
  --manifold_name lin_nsp --d1 20 --d2 10 \
  --dg_seed 42 \
  --ide_name lpca_maxgap \
  --k 25 --N 100 \
  --sigma 0.0 --seed 0 --n_jobs 1
```

Expected result file:

```text
data/ide/_lpca_maxgap_p_(25, None, 0, 100, 0.0)_lin_nsp_mp_(20, 10, 42).pkl
```

Load the result and check its shape, numerical values, and known dimension:

```bash
PYTHONPATH=src python - <<'PY'
import numpy as np
from data_utils import load_ide_info

result = load_ide_info(
    ide_name="lpca_maxgap",
    params=(25, None, 0, 100, 0.0),
    manifold_name="lin_nsp",
    m_params=(20, 10, 42),
    root_dir="./data/ide/",
)
estimates = np.asarray(result["lides"])
assert estimates.shape == (100,)
assert np.isfinite(estimates).all()
assert np.all(estimates == 10)
print("Estimated dimensions:", np.unique(estimates))
print("Median estimate:", np.median(estimates))
PY
```

The expected output is `Estimated dimensions: [10]` and a median of `10.0`. This checks the workflow on a linear space. It does not validate every estimator or reproduce the full paper experiment grid.

**Existing files with the same parameters are overwritten.** The quick test uses the usual sample filename. Run the full generator before requesting more than 100 points from that file.

## Generate a full sample dataset

Each command-line generation run creates 10,000 samples. For example:

```bash
python main.py generate \
  --manifold_name gman_proj \
  --d1 10 --d2 6 \
  --seed 42
```

This writes `data/samples/_gman_proj_p_(10, 6, 42).h5`.

Let `n = d1`, `k = d2`, and `l = d3`. Here, `C(n, k)` is the binomial coefficient. `N` is the sample count.

| CLI name | Construction | Parameter tuple | Stored sample shape |
| --- | --- | --- | --- |
| `lin_nsp` | Linear null space | `(n, k, seed)` | `(N, n)` |
| `gman_proj` | Grassmannian projector embedding | `(n, k, seed)` | `(N, n, n)` |
| `gman_vec` | Grassmannian vector embedding | `(n, k, seed)` | `(N, C(n, k))` |
| `steifel_proj` | Stiefel matrix embedding | `(n, k, seed)` | `(N, n, k)` |
| `steifel_vec1` | Stiefel vector embedding | `(n, k, seed)` | `(N, C(n, k) + k²)` |
| `flag_vec` | Flag vector embedding | `(n, k, l, seed)` | `(N, C(n, k) C(n, l))` |

Keep the original spelling **`steifel`** in CLI arguments and function names. Use `0 < k < n`. For flags, use `0 < l < k < n`. For `steifel_vec1`, use `k >= 2`. The code does not validate all parameter constraints.

Examples for all six generators:

```bash
python main.py generate --manifold_name lin_nsp      --d1 20 --d2 10 --seed 42
python main.py generate --manifold_name gman_proj    --d1 10 --d2 6  --seed 42
python main.py generate --manifold_name gman_vec     --d1 10 --d2 5  --seed 42
python main.py generate --manifold_name steifel_proj --d1 5  --d2 2  --seed 42
python main.py generate --manifold_name steifel_vec1 --d1 5  --d2 2  --seed 42
python main.py generate --manifold_name flag_vec     --d1 6  --d2 3 --d3 2 --seed 42
```

Vector embedding sizes grow with binomial coefficients. Large parameters can require substantial memory and time. Use small parameters for a first run.

To choose a different sample count, use the Python API. Each `data_gen_*` function in `src/manifolds.py` accepts `N` and `params`. Save its returned dictionary with `data_utils.save_info`, as in the quick test.

## Run an ID estimation experiment

First generate the selected manifold. Then pass the same parameters and generation seed to the estimation script:

```bash
python main.py estimate \
  --manifold_name gman_proj --d1 10 --d2 6 \
  --dg_seed 42 \
  --ide_name abid \
  --k 100 --N 1000 \
  --sigma 0.0 --seed 0 --n_jobs 1
```

The script loads stored points, flattens matrix-valued embeddings, selects `N` points without replacement, computes Euclidean nearest neighbors, and runs the estimator. Results are saved in `data/ide/`.

The two seeds have different roles:

- Generation `--seed` in `main.py generate` sets the manifold points. Pass that value as `--dg_seed` to `main.py estimate`.
- Estimation `--seed` in `main.py estimate` sets subset selection and added-noise randomness.

### Common estimation arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--manifold_name` | `lin_nsp` | Stored dataset to load |
| `--d1`, `--d2`, `--d3` | `20`, `10`, `-1` | Generator parameters; `d3` is used only for flags |
| `--dg_seed` | `42` | Generation seed in the sample filename |
| `--ide_name` | `lpca_maxgap` | Estimator to run |
| `--N` | `5000` | Number of stored points to select |
| `--k` | `100` | Nearest-neighbor query size; see below |
| `--sigma` | `0.0` | Scale of added data noise |
| `--noise_type` | `isoGauss` | Noise distribution |
| `--seed` | `0` | Estimation seed |
| `--n_jobs` | `1` | CPU parallelism passed to supported operations |

`N` must not exceed the number of stored samples. Choose `N >= 4` and a neighborhood large enough for the dimension and estimator.

### Estimators and their options

| `--ide_name` | Method | Extra arguments and defaults |
| --- | --- | --- |
| `lpca_maxgap` | Maximum gap in the local PCA spectrum | None |
| `lpca_ratio` | Cumulative local PCA variance threshold | `--eps 0.1` |
| `lpca_fo` | First-order local PCA variant in the original implementation | `--eps 0.1` |
| `mle` | Maximum likelihood estimation | `--dNoise 0.1` |
| `danco` | Distance and angle concentration | `--k1 10` |
| `corrint` | Correlation integral estimation | `--k1 10 --k2 2.0` |
| `twonn` | Two-nearest-neighbor estimation | `--dFrac 0.1` |
| `abid` | Angle-based intrinsic dimension estimation | None |

`--dNoise` configures MLE; it is separate from data-noise scale `--sigma`. `--dFrac` is the TwoNN discard fraction. For CorrInt, the upper scale is `int(k1 * k2)`: the CLI `--k2` value is a multiplier.

For example, after generating the linear dataset:

```bash
python main.py estimate --manifold_name lin_nsp --d1 20 --d2 10 --dg_seed 42 \
  --ide_name lpca_ratio --eps 0.1 --k 100 --N 1000 --seed 0

python main.py estimate --manifold_name lin_nsp --d1 20 --d2 10 --dg_seed 42 \
  --ide_name mle --dNoise 0.0 --k 100 --N 1000 --seed 0

python main.py estimate --manifold_name lin_nsp --d1 20 --d2 10 --dg_seed 42 \
  --ide_name abid --k 100 --N 1000 --seed 0
```

### Neighborhood conventions in the original code

`precompute_knn` requests `k` points **including the query point**. The CLI removes that point. The arrays then contain `k - 1` other points. Thus, `--k 100` supplies 99 other points to lPCA and ABID. If `k >= N - 1`, the script changes `k` to `N - 2`.

The scikit-dimension wrappers have their own fitting behavior. DANCo, CorrInt, and TwoNN pass `N - 2` as the pointwise fitting neighborhood size. Their `--k` argument does not have the same effect as it does for lPCA and ABID. Consult `src/ide.py` when comparing methods. These original implementations are preserved.

### Noise experiments

Set `--sigma` to a positive value to add noise. Supported names are:

| `--noise_type` | Distribution before multiplication by `sigma` |
| --- | --- |
| `isoGauss` | Zero-mean Gaussian with covariance `I / D` |
| `anisoGauss` | Zero-mean Gaussian with a random covariance of trace one |
| `uncorrGauss` | Zero-mean Gaussian with a random diagonal covariance of trace one |
| `exp` | Independently signed exponential entries |
| `uniform` | Independent entries from `[-1, 1]` |

`D` is the flattened ambient dimension. Exponential and uniform noise do not use the same trace normalization as the Gaussian options.

```bash
python main.py estimate \
  --manifold_name lin_nsp --d1 20 --d2 10 --dg_seed 42 \
  --ide_name abid --k 100 --N 1000 \
  --noise_type isoGauss --sigma 0.1 --seed 0
```

The original script computes neighbors **before** it adds noise. Precomputed neighborhoods therefore come from clean data. Some estimator wrappers also compute neighborhoods internally.

**Noise type is not encoded in result filenames.** Runs with the same other parameters and different noise types overwrite the same result. Copy the output to a separate local directory before changing noise type.

## Save and inspect results

### Manifold sample files

Samples are stored as:

```text
data/samples/_<manifold>_p_<generator_parameters>.h5
```

The HDF5 `samples` dataset contains the points. Some generators also store arrays such as `matrixA` or `listQs`. Arrays use LZF compression. The `params` attribute stores the generation parameters.

```bash
PYTHONPATH=src python - <<'PY'
from data_utils import load_info

dataset = load_info(
    params=(10, 6, 42),
    manifold_name="gman_proj",
    root_dir="./data/samples/",
)
print(dataset["samples"].shape)
PY
```

### Estimation result files

Results are Python dictionaries saved as pickle files:

```text
data/ide/_<method>_p_<method_parameters_N_sigma>_<manifold>_mp_<generator_parameters>.pkl
```

The filename parameter tuples are:

| Method | Tuple before the manifold name |
| --- | --- |
| `lpca_maxgap` | `(k, None, seed, N, sigma)` |
| `lpca_ratio`, `lpca_fo` | `(k, eps, seed, N, sigma)` |
| `mle` | `(k, dNoise, seed, N, sigma)` |
| `danco` | `(k, adjusted_k1, seed, N, sigma)` |
| `corrint` | `(k1, effective_upper_scale, seed, N, sigma)` |
| `twonn` | `(k, dFrac, seed, N, sigma)` |
| `abid` | `(k, seed, N, sigma)` |

For DANCo, `k1` is raised to `k` when it is no larger than `k`. CorrInt can also adjust its scales if the upper scale reaches `N - 1`. Use the adjusted values when loading a result.

Each dictionary contains `lides` (one local ID estimate per point), `params` (estimator parameters), `manifold`, and `m_params` (generation parameters). The dictionary's `params` omits the filename's final `N` and `sigma`. lPCA also stores `ver` and `exp_var_ratios`.

To summarize a trusted result:

```bash
PYTHONPATH=src python - <<'PY'
import numpy as np
from data_utils import load_ide_info

result = load_ide_info(
    ide_name="abid",
    params=(100, 0, 1000, 0.0),
    manifold_name="gman_proj",
    m_params=(10, 6, 42),
    root_dir="./data/ide/",
)
values = np.asarray(result["lides"])
print("Count:", values.size)
print("Mean:", np.mean(values))
print("Median:", np.median(values))
print("Standard deviation across points:", np.std(values))
PY
```

`load_ide_info` has a different default directory from the CLI output directory. Pass `root_dir="./data/ide/"` explicitly. Load pickle files and pickled HDF5 attributes only from trusted sources. Public plot bundles use NumPy arrays and load with `allow_pickle=False`.

### Repeated seeds

Repeat a test with different estimation seeds:

```bash
for seed in 0 1 2; do
  python main.py estimate \
    --manifold_name gman_proj --d1 10 --d2 6 --dg_seed 42 \
    --ide_name abid --k 100 --N 1000 --sigma 0.0 \
    --seed "$seed" --n_jobs 1
done
```

To vary manifold sampling, regenerate with a different generation seed and set `--dg_seed` to that value. Record both seeds. Variation across pointwise estimates and variation across independent runs are different quantities.

## Reproduce the main paper plots

The notebooks in `plots/` load the compressed numerical bundles from `data/`. They do not require the large raw sample archives or a rerun of the estimator sweeps.

The six released PDF snapshots are included directly in `plots/`. Open them on GitHub without running a notebook:

| Paper figure | Released PDF |
| --- | --- |
| 2 | [HomID compared with matched spheres](plots/new_compare_rel_sphere_only.pdf) |
| 3 | [Diagonal distortion](plots/score_vs_epsilon_sub.pdf) |
| 4 | [Additive noise](plots/noise_tre_scaling_rel_sub.pdf) |
| 5 | [MLE on MNIST](plots/mle_squeeze_noise.pdf) |
| 6 | [Local PCA spectra](plots/lpca_interp_mt.pdf) |
| 7 | [Neighborhood-angle histograms](plots/abid_interp.pdf) |

To reproduce all six numerical main-text figures from the terminal:

```bash
python main.py plot
```

The command executes both notebooks with nbconvert and the current Python environment. Regenerated PDFs and executed notebook copies go into `plots/generated/`. The six released PDFs and source notebooks stay unchanged. The temporary output directory is ignored by Git.

To run one figure group:

```bash
python main.py plot --figure-group benchmark
python main.py plot --figure-group mechanism
```

The benchmark group reproduces Figures 2–4. The mechanism group reproduces Figures 5–7. Figure 1 is a conceptual overview and is excluded from numerical reproduction.

To inspect or run the notebooks interactively:

```bash
python -m notebook plots/benchmark_plots.ipynb
python -m notebook plots/mechanism_plots.ipynb
```

Select **Run All Cells** in each notebook. See [plots/README.md](plots/README.md) for the figure output names. See [data/README.md](data/README.md) for retained quantities, source dates, aggregation rules, and compression instructions.

Each notebook includes the code used to prepare its compressed bundle. Rebuilding a bundle requires the original external experiment files. Plotting from bundled data works from a fresh clone.

## Release validation

The pinned dependencies were checked in a fresh Python 3.11.7 environment on a macOS ARM CPU. All six generators passed a 100-point check. HDF5 save/load preserved their arrays and metadata. The generation CLI also produced its expected 10,000 points.

The lPCA variants, ABID, MLE, TwoNN, and CorrInt completed 100-point CLI checks with finite result arrays. The maximum-gap lPCA test recovered dimension 10 at every point. Both plot notebooks executed successfully and produced all six numerical main-text figures. Lossless archive round trips passed.

DANCo did not finish within the 35-second smoke-test limit. Its completion is not verified by these checks. The other small tests verify execution and output structure; they do not establish estimator accuracy on the full benchmark.

## Troubleshooting and implementation notes

- **`Samples file not found.`** Generate samples first. Check the manifold, dimensions, and generation seed. The loader expects the `.h5` filename produced by `save_info`.
- **Requested `N` exceeds available samples.** A quick test may have saved only 100 points under the same filename. Run the full generator again, or reduce `N`.
- **Imports fail.** Activate the intended environment. Run the installation and import checks above.
- **Small neighborhoods give poor estimates.** Increase sample and neighborhood sizes. A successful run does not prove that an estimate is accurate. Compare against the known dimension.
- **Unexpected lPCA text filenames.** The original helper writes extra explained-variance files with a comma in the prefix, such as `data/ide/,_..._exvar.txt`. Its cache lookup uses a different prefix, so PCA values can be recomputed. These files are ignored.
- **Unexpected `lpca_fo` behavior.** This variant preserves the original threshold search on the PCA variance array. In the small release test, it returned 21 for ambient dimension 20. No algorithm correction is part of this release preparation.
- **Results change after another run.** Matching filenames are overwritten. Noise type is absent from filenames. Keep separate local copies when needed.

The CLI exposes the experiments implemented in the original source. It does not provide a launcher for every paper experiment, including the real-data and mechanistic analyses. The numerical bundles and notebooks provide the main-figure reproduction path.

## AI use declaration

The original scientific code was written by the author without the use of AI. AI assistance was used only to prepare this repository for release, including its organization, README, writeup, technical descriptions, data packaging, plot-reproduction notebooks, and the `main.py` command-line wrapper. The original manifold-generation, data-storage, nearest-neighbor, and ID-estimation source files were moved into `src/` without changes to their contents.

## License

The repository is distributed under the [MIT License](LICENSE).
