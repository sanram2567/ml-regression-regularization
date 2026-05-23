# P2 — Finance Regression: Regularization & Uncertainty

**Layer 1 · Classical ML | Dataset: California Housing | Week 1–2**

> End-to-end regression pipeline on the California Housing dataset covering EDA,
> custom feature engineering, regularised linear models, Bayesian uncertainty
> quantification, polynomial feature expansion, Optuna hyperparameter tuning,
> SHAP interpretability, and robust regression — built as a production-grade
> sklearn pipeline with zero data leakage.

---

## Project Structure

```
P2/
├── notebooks/
│   └── finance_regression.ipynb
├── outputs/
│   ├── 01_distributions.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_target_transform.png
│   ├── 04_geo_clusters.png
│   ├── 05_rmse_comparison.png
│   ├── 06_lasso_coefficient_path.png
│   ├── 07_bayesian_ridge_pi.png
│   ├── 08_poly_ridge_diagnostics.png
│   ├── 09_final_comparison.png
│   ├── 10_optuna_histories.png
│   ├── 11_optuna_enet_contour.png
│   ├── 12_shap_ridge_beeswarm.png
│   ├── 13_shap_ridge_bar.png
│   ├── 14_shap_ridge_waterfall.png
│   ├── 15_shap_poly_bar.png
│   ├── 16_shap_poly_waterfall.png
│   ├── 17_shap_comparison.png
│   └── huber_vs_ridge_residuals.png
└── README.md
```

---

## Environment Setup

```bash
python -m venv p2_env
source p2_env/bin/activate        # Windows: p2_env\Scripts\activate
pip install scikit-learn numpy pandas matplotlib seaborn scipy statsmodels
pip install optuna shap
```

---

## Dataset

**`sklearn.datasets.fetch_california_housing()`** — 20,640 rows × 8 features
(1990 California census block groups).

| Feature | Description |
|---------|-------------|
| `MedInc` | Median income in block group ($10,000s) |
| `HouseAge` | Median house age (years) |
| `AveRooms` | Average rooms per household |
| `AveBedrms` | Average bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average household members |
| `Latitude` / `Longitude` | Block group coordinates |
| `MedHouseVal` *(target)* | Median house value ($100,000s) |

**Target transform:** `log1p(MedHouseVal)` — reduces skewness from 0.978 to 0.276.
All predictions back-transformed with `expm1()` before computing metrics.

---

## Data Cleaning

Two-stage cleaning policy — physically impossible values removed before the split;
data-driven filters applied to training data only after the split.

### Stage 1 — Before split (domain knowledge, not model-derived)

| Filter | Rows removed | Reason |
|--------|:-----------:|--------|
| `AveOccup >= 50` | 7 | Max 1,243 people/household — census entry error |
| `AveRooms >= 50` | 9 | Max 141.9 rooms/household — impossible |
| `AveBedrms >= 20` | 2 | Max 34.1 bedrooms/household — impossible |
| **Total** | **16 (0.08%)** | 20,624 rows retained |

### Stage 2 — Training set only, after split (data-driven)

| Filter | Rows removed | Reason |
|--------|:-----------:|--------|
| `AveOccup < 1.0` AND `AveRooms > 15` | ~20–30 | Sparse vacation/seasonal blocks — structurally different market segment linear models cannot generalise |

The test set is untouched — the model is evaluated on all samples including
unusual ones, which is the correct production-equivalent setup.

---

## Feature Engineering Pipeline

Custom `CaliforniaFeatureEngineer` (`BaseEstimator` + `TransformerMixin`).
All transformers fit on training data only. KMeans geo-cluster fit on training
coordinates only — no leakage.

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `bedroom_ratio` | `AveBedrms / AveRooms` | De-correlates AveRooms & AveBedrms (r=0.73) |
| `rooms_per_person` | `AveRooms / AveOccup` | Crowding index |
| `income_rooms` | `MedInc × AveRooms` | Explicit interaction — 2nd-highest SHAP feature |
| `dist_to_LA` | Euclidean to (34.05, −118.24) | LA coastal premium |
| `dist_to_SF` | Euclidean to (37.77, −122.42) | Bay Area premium |
| `log_population` | `log1p(Population)` | Compresses right-skewed population |
| `log_ave_occup` | `log1p(AveOccup)` | 3rd-highest SHAP feature |
| `log_ave_rooms` | `log1p(AveRooms)` | Compresses right skew (skew > 4) |
| `log_ave_bedrms` | `log1p(AveBedrms)` | Same |
| `geo_cluster` | KMeans(k=20) on lat/lon → OHE | 20 spatial price zones → 19 dummy columns |

**Result:** 8 raw → 28 numeric + 19 OHE = **47 engineered features** fed into
the ColumnTransformer; StandardScaler applied to all numeric features.

---

## Complete Results

> All R² and RMSE/MAE computed on the **test set** using back-transformed
> (`expm1`) predictions. RMSE and MAE are in USD.

### Baseline vs Engineered (4 shared models)

| Model | Baseline RMSE ($) | Engineered RMSE ($) | Improvement |
|-------|:-----------------:|:-------------------:|:-----------:|
| OLS | 73,912 | 72,638 | −$1,274 |
| Ridge | 73,941 | 72,640 | −$1,301 |
| Lasso | 73,897 | 73,129 | −$768 |
| ElasticNet | 73,899 | 72,832 | −$1,067 |

Feature engineering contributed a modest ~$1,000–1,300 RMSE reduction for linear
models. The dominant gain comes from polynomial feature expansion (see below).

---

### All Models — Full Comparison

| Section | Model | RMSE ($) | MAE ($) | R² | Notes |
|---------|-------|:--------:|:-------:|:--:|-------|
| Baseline | OLS | 73,912 | 49,821 | 0.587 | 8 raw features |
| Baseline | Ridge (RidgeCV) | 73,941 | 49,833 | 0.586 | α = 6.593 |
| Baseline | Lasso (LassoCV) | 73,897 | 49,850 | 0.587 | α = 0.00026 |
| Baseline | ElasticNet | 73,899 | 49,850 | 0.587 | α = 0.00029 |
| Engineered | OLS | 72,638 | 43,444 | 0.601 | 26 engineered features |
| Engineered | Ridge (RidgeCV) | 72,640 | 43,447 | 0.601 | α = 0.055 |
| Engineered | Lasso (LassoCV) | 73,129 | 43,650 | 0.595 | Zeroed: `geo_cluster_7` |
| Engineered | ElasticNet | 72,832 | 43,523 | 0.599 | l1_ratio = 0.019 (≈ Ridge) |
| Engineered | BayesianRidge | 72,770 | 43,502 | 0.599 | 95% PI coverage = 93.8% |
| Engineered | **Poly(2)+Ridge** | **56,346** | **38,411** | **0.760** | 146 features, α = 27.05 |
| Optuna | Ridge | 72,640 | — | — | α = 0.057 |
| Optuna | Lasso | 73,128 | — | — | α = 0.0001 |
| Optuna | ElasticNet | 72,840 | — | — | α = 0.00013, l1_ratio = 0.023 |
| Optuna | BayesianRidge | 72,727 | — | — | 95% PI coverage = 93.8% |
| Robust | HuberRegressor (ε=2.0) | 85,923 | 43,637 | 0.441 | RobustScaler |

---

### Polynomial Feature Expansion — Key Result

| Metric | Linear Ridge | Poly(2)+Ridge | Change |
|--------|:------------:|:-------------:|:------:|
| RMSE ($) | 72,640 | 56,346 | **−22.4%** |
| MAE ($) | 43,447 | 38,411 | **−11.6%** |
| R² | 0.601 | 0.760 | **+0.159** |
| Features | 26 | 146 | +120 poly terms |

The polynomial expansion is the single largest performance gain in the project.
Polynomial terms capture non-linear relationships (MedInc curve, geographic
interactions) that linear Ridge cannot express regardless of regularisation strength.

---

### Lasso Feature Selection

At best alpha (0.00010), Lasso zeroed **1 of 26 features**: `geo_cluster_7`.

This cluster carries no additional price signal beyond what the other spatial
features and geo-cluster dummies already capture. All 25 remaining features —
including all engineered numeric features — were retained as active predictors,
confirming the feature engineering step produced useful non-redundant signal.

---

### BayesianRidge — 95% Prediction Interval Coverage

**Delta method for log1p transform:**
```python
# y = log1p(x)  →  dx/dy = exp(y) = y_pred + 1
y_std = y_std_log * (y_pred + 1)
```

| Metric | Value |
|--------|-------|
| Empirical 95% coverage | **93.8%** |
| Target | 95.0% |
| Gap | −1.2% |

**Coverage by decile:**

| Decile | Coverage | Decile | Coverage |
|--------|:--------:|--------|:--------:|
| 0 (cheapest) | 94.4% | 5 | 92.5% |
| 1 | 97.1% | 6 | 91.3% |
| 2 | 97.6% | 7 | 90.6% |
| 3 | 96.6% | **8** | **88.1%** |
| 4 | 95.9% | 9 (priciest) | 94.2% |

Coverage weakest at decile 8 — the $500K ceiling in the dataset creates a
non-Gaussian spike that the Gaussian posterior underestimates.

---

### Optuna Hyperparameter Tuning

80 trials × 5-fold CV, TPE sampler. Gain vs sklearn CV:

| Model | sklearn RMSE ($) | Optuna RMSE ($) | Gain ($) |
|-------|:----------------:|:---------------:|:--------:|
| Ridge | 72,640 | 72,640 | 0 |
| Lasso | 73,129 | 73,128 | +1 |
| ElasticNet | 72,832 | 72,840 | −8 |
| BayesianRidge | 72,770 | 72,727 | +43 |

sklearn's RidgeCV / LassoCV are already near-optimal for single-parameter linear
models. Optuna's value was in confirming ElasticNet collapses to Ridge
(l1_ratio = 0.023) — a region outside sklearn's fixed grid of [0.1, 0.3, 0.5, 0.7, 0.9].

---

### Residual Diagnostics — Poly(2)+Ridge

| Statistic | Value | Ideal | Interpretation |
|-----------|:-----:|:-----:|----------------|
| Mean | 0.0 | ≈ 0 | ✓ Unbiased |
| Std | 0.6 | — | — |
| Skew | 0.987 | ≈ 0 | ✗ Underestimates expensive homes |
| Kurtosis | 4.972 | ≈ 0 | ✗ Heavy tails from $500K ceiling |

Skew and kurtosis reflect the structural $500K cap in California Housing — not
a modelling error. The log1p transform reduces but cannot eliminate this artifact.

---

### HuberRegressor — Note on Results

HuberRegressor with RobustScaler produced RMSE $85,923 — worse than all other
models. The cause: Huber's robustness to large residuals is its strength for
RMSE, but the RobustScaler on the engineered feature matrix (which already
includes log-compressed features) introduced scaling inconsistencies. The
best epsilon found was 2.0 — at the edge of the search range — suggesting
the optimal epsilon may lie outside [1.1, 2.0] for this specific feature
matrix, or that StandardScaler is a better pairing with Huber here.
MAE ($43,637) is comparable to Ridge ($43,447) — confirming the model's
point predictions are reasonable; the RMSE penalty comes from a few
large errors that Huber's linear loss still cannot eliminate on this dataset.

---

### SHAP Feature Importance — Ridge (Engineered)

| Rank | Feature | Mean \|SHAP\| | Direction |
|------|---------|:------------:|-----------|
| 1 | `MedInc` | 0.075 | High income → price UP |
| 2 | `income_rooms` | 0.048 | Wealth × space → price UP |
| 3 | `log_ave_occup` | 0.035 | High occupancy → price DOWN |
| 4 | `Latitude` | 0.031 | Higher latitude (NorCal) → price UP |
| 5 | `rooms_per_person` | 0.026 | More space per person → price UP |
| 6 | `dist_to_LA` | 0.025 | Far from LA → price DOWN |
| 7 | `Longitude` | 0.022 | Coastal → price UP |
| 8 | `HouseAge` | 0.018 | Older homes in desirable areas → price UP |

`income_rooms` (engineered) ranks 2nd above all geographic features —
confirms the explicit interaction term captured signal beyond raw `MedInc` alone.

### Poly(2)+Ridge vs Ridge — Feature Amplification

| Feature | Ratio | Interpretation |
|---------|:-----:|----------------|
| `dist_to_SF` | **3.78×** | Bay Area premium is strongly non-linear |
| `dist_to_LA` | **1.93×** | Distance-to-coast effect has a threshold |
| `AveRooms` | **1.86×** | Room count interacts with other features |
| `HouseAge` | **1.66×** | Age matters more in context of location |
| `MedInc` | 0.95× | Already near-linear — polynomial adds little |

---

### Known Limitation — Sample 2530

| Feature | Value |
|---------|-------|
| `AveOccup` | 0.69 (fewer residents than units) |
| `AveRooms` | 28.6 |
| `Population` | 27 |
| `MedHouseVal` | $83,000 |
| Ridge predicted | ~$770,000 |

A sparse vacation/seasonal property block — large homes, almost no permanent
residents, low price. Post-scaling `rooms_per_person` reaches z = 46.65,
catastrophically outside the training distribution. The linear model
extrapolates the learned relationship (high rooms → high price) to an absurd
degree. This is irreducible error for any linear model and the primary
motivation for tree-based models in P3, which isolate this region in a leaf node.

---

## Key Takeaways

| Topic | Finding |
|-------|---------|
| **Feature engineering** | ~$1,300 RMSE gain for linear models — modest but meaningful |
| **Polynomial expansion** | 22.4% RMSE reduction — single largest gain in the project |
| **Regularisation** | All linear models cluster within $500 — feature matrix matters more than model choice |
| **Lasso selection** | 1 of 26 zeroed (`geo_cluster_7`) — engineered matrix is efficient |
| **Bayesian coverage** | 93.8% vs 95% target — weakest at decile 8 due to $500K ceiling |
| **Delta method** | Correct form: `y_std * (y_pred + 1)` — explicit about log1p +1 offset |
| **Optuna** | Negligible gain for linear models — confirmed ElasticNet collapses to Ridge |
| **Huber** | RMSE worse than Ridge — MAE comparable; RobustScaler pairing needs revisiting |
| **SHAP** | `dist_to_SF` 3.78× poly amplification — strongest non-linearity signal |
| **Irreducible error** | Vacation-block sample cannot be fixed linearly — motivates P3 |
