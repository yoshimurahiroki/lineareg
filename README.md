# lineareg: Unified Econometrics Toolkit
*(日本語の解説は後半に掲載しています)*

`lineareg` delivers a research-grade implementation of the most commonly used linear and causal estimators in applied microeconometrics. The entire codebase now follows a single naming scheme (`*_name`) for DataFrame column references, exposes a consistent bootstrap-only inference API, and centralises every numerical kernel in `core.linalg` and `core.bootstrap`.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Why lineareg?
- **Unified API surface** – every estimator accepts `id_name`, `t_name`, `cohort_name`, `treat_name`, and `y_name` (when relevant). All public APIs require the canonical column keywords described below.
- **Bootstrap-only inference** – analytical standard errors and p-values are never produced. Wild/multiplier bootstrap with MNW 11/13/31/33/33J variants, Webb enumeration, and multiway clustering is implemented once in `core.bootstrap`.
- **Deterministic linear algebra** – QR with column pivoting, sparse-aware products, and minimum-norm solvers live in `core.linalg`. Estimators never call `np.linalg.inv`.
- **Rich event-study support** – Callaway–Sant’Anna, doubly-robust DID, triple-difference, spatial DID, SDID, and synthetic control all expose uniform sup-t bands, post-period aggregation, and publication-ready plots.
- **Value-only outputs** – the shared `EstimationResult` container stores parameters, bootstrap SEs, confidence bands, and diagnostics in `extra`. No analytic inference artefacts are recorded.

---

## Unified API quick reference

| Concept | Canonical keyword | Applies to |
|---------|-------------------|-----------|
| Unit identifier column | `id_name` | All panel / event-study estimators |
| Time identifier column | `t_name` | Panel / event-study |
| Adoption / cohort column | `cohort_name` | DID / event-study / SDID / SyntheticControl |
| Treatment indicator | `treat_name` | DR-DID, Spatial DID, SDID |
| Outcome column | `y_name` | All estimators needing explicit outcome |
| Covariate list | `covariate_names` | Callaway–Sant’Anna, DR-DID |

All estimators expose:

```python
estimator = EstimatorClass.from_formula(
    data=df,
    formula="y ~ x1 + x2",
    id=...,              # optional shortcut for id_name
    time=...,            # shortcut for t_name
    boot=BootConfig(...)
)
result = estimator.fit()
```

`Estimator.fit_from_formula(...)` provides a one-shot helper when immediate fitting is preferred.

---

## Estimator families

### Linear models
- **OLS / GLS / IV / GMM / QR / SAR2SLS** share the `BaseEstimator` bootstrapping pipeline.
- Fixed-effects absorption and constraint solvers are delegated to `core.fe` and `utils.constraints`, keeping rank-policy logic centralised.
- Weak-IV diagnostics (Cragg–Donald, Kleibergen–Paap, Sanderson–Windmeijer, Montiel-Olea–Pflueger) are reported via `result.extra`.

### Event-study & DID suite
- **`CallawaySantAnnaES`** – long-differences estimator with uniform sup-t bands (`pre`, `post`, `full`), post-period ATT aggregation, and optional covariate adjustment/DR weighting.
- **`DREventStudy`** – cross-fitted doubly-robust DID with compatibility switches for R `did` and `csdid`.
- **`DDDEventStudy`** – difference-in-difference-in-differences wrapper aligning two ES runs on intersection support and providing simultaneous bands.
- **`SpatialDID`** – staggered adoption with spillovers, accepting sparse/dense spatial weights, exposure-based ATT reporting, and Moran’s I diagnostics.
- **`SDID`** – synthetic DID with Frank–Wolfe donor/time weights, placebo bootstrap bands, and post-period aggregation.
- **`SyntheticControl`** – cohort-aware synthetic control with simplex QP/FW solvers, time-direction multiplier bootstrap, and support for optimise-v routines.

### Randomised experiments
- **`RCT`** estimators provide regression adjustment and Hájek-IPW contrasts with experiment-wide sup-t bands, multi-hypothesis handling, and positive-multiplier bootstrap.

---

## Outputs & companion utilities
- **`modelsummary`** – text and LaTeX tables presenting coefficients with bootstrap standard errors (parenthesised), confidence intervals (brackets) for event-study estimators, and footnotes documenting bootstrap settings.
- **`event_study_plot`** – Matplotlib helper for plotting ATT paths with uniform bands, baseline markers, and post-period aggregates.
- **Spatial diagnostics** – Moran’s I residual tests accessible via `spatial.spatial.moran_test(result, W)` for any estimator once residuals are aligned.

---

## GPU Acceleration (Optional)

All linear estimators (`OLS`, `GLS`, `IV`, `GMM`, `QR`) now accept an optional `device` parameter:

```python
from lineareg.estimators import OLS
from lineareg.estimators.base import BootConfig

# Force CPU execution (default)
res = OLS(y, X).fit(device="cpu", boot=BootConfig(n_boot=2000))

# Use first CUDA GPU if available (requires CuPy)
res = OLS(y, X).fit(device="cuda:0", boot=BootConfig(n_boot=2000))
```

When `device="cuda:*"` and CuPy is installed, dense matrix operations are transparently routed through GPU kernels while maintaining strict numerical equivalence with CPU execution. The public API always returns NumPy arrays. GPU acceleration is most effective for large panels (n > 10,000) with bootstrap replication counts > 1000.

**Memory management**: Device contexts automatically release GPU cache after heavy operations. For memory-constrained environments, use smaller bootstrap batches or CPU-only mode.

---

## Usage highlights

### Callaway–Sant’Anna event study
```python
from lineareg.estimators.eventstudy_cs import CallawaySantAnnaES
from lineareg.estimators.base import BootConfig

cs = CallawaySantAnnaES(
    id_name="unit_id",
    t_name="time",
    cohort_name="cohort",
    y_name="outcome",
    tau_weight="group",
    boot=BootConfig(n_boot=1200, dist="rademacher", seed=202)
)
result = cs.fit(panel_df)
print(result.params.head())
print(result.bands["post"])  # bootstrap uniform bands
```

### Synthetic control with simplex QP
```python
from lineareg.estimators.synthetic_control import SyntheticControl
from lineareg.estimators.base import BootConfig

sc = SyntheticControl(
    id_name="id",
    t_name="time",
    cohort_name="g",
    y_name="y",
    solver="qp",
    tau_weight="group",
    boot=BootConfig(n_boot=800, mode="time", seed=7)
)
sc_res = sc.fit(data=sc_panel)
print(sc_res.model_info["PostATT"])
```

### Spatial DID with Moran diagnostics
```python
from lineareg.estimators.spatial_did import SpatialDID
from lineareg.spatial.spatial import moran_test

sp_did = SpatialDID(
    id_name="unit",
    t_name="year",
    cohort_name="g",
    treat_name="treated",
    y_name="y",
    W=spatial_weights,
    tau_weight="treated_t"
)
res = sp_did.fit(df)
print(res.params.loc["PostATT"])
print(moran_test(res, W=spatial_weights))
```

---

## Testing & quality assurance
- Unit tests in `tests/` cover policy guards (no analytic SEs), bootstrap enumeration paths, formula parsing, estimator behaviours, and summary/plot outputs.
- Continuous integration executes the full suite with pinned dependency versions; stochastic routines rely on deterministic seeds for reproducibility.
- Before releasing, run:

```bash
poetry install
poetry run pytest tests
```

---

## Installation
```bash
pip install -e /path/to/lineareg
```

Key dependencies: `numpy>=1.23`, `scipy>=1.9`, `pandas>=1.5`, `patsy>=0.5.3`, `matplotlib>=3.5`, `tabulate>=0.9`.

### Optional GPU acceleration (CuPy)
- Dense linear algebra kernels can run on GPU when [CuPy](https://cupy.dev) is installed and a CUDA device is available. Public APIs continue to return NumPy arrays; GPU is used internally.
- To enable, set an environment variable before running your code:

```bash
export LINEAREG_DEVICE=gpu            # or LINEAREG_USE_GPU=1
```

- You can also selectively enable in-process via:

```python
from lineareg.core import backend
with backend.DeviceGuard("gpu"):
    # code that benefits from GPU-backed dense multiplies/Cholesky/SVD
    ...
```

- Memory management: the library proactively releases CuPy memory pools after large operations; you can force a release via `lineareg.core.backend.free_gpu_cache()`.

---

# lineareg: Python向け統一計量ツールキット

英語セクションと同内容を日本語で再構成しています。API 名称はすべて `*_name` に統一されており、旧スタイルのキーワードは使用できません。

## 特徴
- **API の統一** – すべての推定器が `id_name`, `t_name`, `cohort_name`, `treat_name`, `y_name` を受け付けます。公開APIではこれらの正式キーワードのみ利用可能です。
- **ブートストラップ専用推論** – MNW 乗数ブートストラップ（11/13/31/33/33J）、Webb 列挙、マルチウェイクラスタリングをサポートします。解析的な標準誤差・p 値は一切出力しません。
- **数値計算の集中管理** – `core.linalg` に QR 分解や疎行列演算、`core.bootstrap` に推論ロジックを集約し、反復可能で安定した結果を保証します。
- **イベントスタディの充実** – Callaway–Sant’Anna、DR-DID、DDD、空間 DID、SDID、シンセティックコントロールが均一バンドと投稿期集計を備えています。

## 統一キーワード対応表

| 用語 | 正式キーワード | 主な対象 |
|------|----------------|-----------|
| 個体 ID | `id_name` | パネル/イベントスタディ |
| 時間 | `t_name` | パネル/イベントスタディ |
| 導入時期 | `cohort_name` | DID 系推定器 |
| 処置指標 | `treat_name` | DR-DID, 空間 DID, SDID |
| 被説明変数 | `y_name` | 全推定器 |

## 利用例

```python
from lineareg.estimators.eventstudy_cs import CallawaySantAnnaES

cs = CallawaySantAnnaES(
    id_name="id",
    t_name="time",
    cohort_name="g",
    y_name="y",
    tau_weight="group",
)
result = cs.fit(df)
print(result.params.head())
print(result.bands["post"])
```

## 出力ユーティリティ
- `modelsummary` – 係数とブートストラップ標準誤差（括弧）、イベントスタディ用信頼区間（角括弧）を整形。
- `event_study_plot` – 均一バンド付きの ATT 推移をプロット。
- `spatial.spatial.moran_test` – 任意推定結果に対する Moran’s I を計算。

## 品質管理
- `tests/` ディレクトリの単体テストでポリシー違反を検知。
- 再現性確保のため擬似乱数にはすべて明示的な seed を使用。

## インストール
```bash
pip install -e /path/to/lineareg
```

依存: `numpy>=1.23`, `scipy>=1.9`, `pandas>=1.5`, `patsy>=0.5.3`, `matplotlib>=3.5`, `tabulate>=0.9`。

### GPU（任意、CuPy）
- CuPy が導入済みで CUDA デバイスが利用可能な環境では、一部の密行列演算を GPU で実行できます（外部APIは NumPy 配列のまま）。
- 有効化するには、実行前に環境変数を設定してください:

```bash
export LINEAREG_DEVICE=gpu            # または LINEAREG_USE_GPU=1
```

- プログラム内で一時的に強制することもできます:

```python
from lineareg.core import backend
with backend.DeviceGuard("gpu"):
    ...
```

- メモリ管理: 大規模計算後は自動的に GPU メモリプールの解放を試みます。明示的に解放したい場合は `lineareg.core.backend.free_gpu_cache()` を利用できます。

---

ご意見・バグ報告は Issues まで。
