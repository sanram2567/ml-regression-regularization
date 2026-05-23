# P2 — Finance Regression: Regularization & Uncertainty

**Layer 1 · Classical ML | Dataset: California Housing | Week 1–2**

> Regression pipeline progressing from EDA through regularised models, Bayesian uncertainty quantification, polynomial feature expansion, Optuna hyperparameter tuning, and SHAP interpretability — all built on a custom feature engineering pipeline.

---

## Project Structure

```
P2/
├── notebooks/
│   └── finance_regression_updated.ipynb   ← main notebook (43 cells)
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
│   └── 17_shap_comparison.png
└── README.md
```

---

## Environment Setup

```bash
python -m venv ml-regression-regularization
source ml-regression-regularization/bin/activate        # Windows: p2_env\Scripts\activate

pip install scikit-learn numpy pandas matplotlib seaborn scipy statsmodels
pip install optuna shap
```

**Tested on:** Python 3.10+ · scikit-learn 1.4+ · optuna 3.x · shap 0.45+

---

## Dataset

**`sklearn.datasets.fetch_california_housing()`** — 20,640 rows × 8 features (1990 California census).

| Feature | Description |
|---------|-------------|
| `MedInc` | Median income in block group ($10,000s) |
| `HouseAge` | Median house age (years) |
| `AveRooms` | Average rooms per household |
| `AveBedrms` | Average bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average household members |
| `Latitude` / `Longitude` | Block group coordinates |
| `MedHouseVal` *(target)* | Median house value ($100,000s) — **log1p-transformed** |

### Outlier Removal Policy

Only physically impossible values removed — not IQR outliers indiscriminately:

| Filter | Rows removed | Reason |
|--------|-------------|--------|
| `AveOccup >= 50` | 7 | Max was 1,243 people/household — census data entry error |
| `AveRooms >= 50` | 9 | Max was 141.9 rooms/household average — impossible |
| `AveBedrms >= 20` | 2 | Max was 34.1 bedrooms/household average — impossible |
| **Total** | **16 rows (0.08%)** | Remaining 20,624 rows kept |

High IQR counts in `AveBedrms` (6.9%) and `Population` (5.8%) are **not removed** — they reflect real skewed distributions, not data corruption. Skew is handled downstream by log-transforms and ratio features.

---

## Feature Engineering Pipeline

Custom `CaliforniaFeatureEngineer` transformer (sklearn-compatible `BaseEstimator` + `TransformerMixin`) adds 7 new features before modelling. KMeans geo-cluster is fit on training data only — no leakage.

| Feature | Formula / Method | Rationale |
|---------|-----------------|-----------|
| `bedroom_ratio` | `AveBedrms / AveRooms` | Breaks 0.73 collinearity between AveRooms & AveBedrms |
| `rooms_per_person` | `AveRooms / AveOccup` | Crowding index — captures density signal |
| `income_rooms` | `MedInc × AveRooms` | Explicit interaction — 4th-highest SHAP feature |
| `dist_to_LA` | Euclidean to (34.05, -118.24) | LA coastal premium |
| `dist_to_SF` | Euclidean to (37.77, -122.42) | Bay Area premium |
| `log_population` | `log1p(Population)` | Compresses right-skewed population outliers |
| `log_ave_occup` | `log1p(AveOccup)` | Same — 3rd-highest SHAP feature |
| `geo_cluster` | KMeans(k=12) on lat/lon → OHE | 12 spatial price zones → 11 dummy columns |

**Result:** 8 raw features → 26 engineered features (15 numeric + 11 OHE geo-cluster dummies).

---

## Results

### Baseline vs Engineered Features

| Model | Baseline RMSE ($) | Engineered RMSE ($) | Improvement |
|-------|:-----------------:|:-------------------:|:-----------:|
| OLS | 73,912 | 69,938 | −$3,974 |
| Ridge (RidgeCV) | 73,941 | 70,443 | −$3,498 |
| Lasso (LassoCV) | 73,897 | 70,695 | −$3,202 |
| ElasticNet | 73,899 | 70,276 | −$3,623 |

> Feature engineering alone reduced OLS RMSE by **~$4,000** without touching the model.

---

### Full Model Comparison (Engineered Features)

| Model | RMSE ($) | MAE ($) | R² | Notes |
|-------|:--------:|:-------:|:--:|-------|
| OLS (baseline) | 69,938 | 44,552 | 0.630 | No regularisation, all 26 features |
| Ridge (RidgeCV) | 70,443 | 44,569 | 0.624 | α = 3.45, L2 shrinkage |
| Lasso (LassoCV) | 70,695 | 44,593 | 0.622 | α = 0.0001, zeroed 1 feature (`geo_cluster_7`) |
| ElasticNet | 70,276 | 44,561 | 0.626 | α = 0.0001, l1_ratio = 0.019 |
| BayesianRidge | 70,169 | 44,559 | 0.621 | 95% PI coverage = **93.8%** |
| **Poly(2)+Ridge** | **56,919** | **38,688** | **0.755** | 135 poly + 11 OHE = 146 features, α = 27.05 |

> **Poly(2)+Ridge** delivers a **19.2% RMSE reduction** over linear Ridge — the single biggest model-level gain.

---

### Lasso Feature Selection

At the LassoCV-chosen alpha (0.0001), only **1 of 26 features** was zeroed: `geo_cluster_7`.
All 15 engineered numeric features and 10 of 11 geo-cluster dummies remained active —
confirming the feature engineering step produced genuinely useful signal throughout.

---

### BayesianRidge — 95% Prediction Interval Coverage

| Metric | Value |
|--------|-------|
| Empirical coverage | **93.8%** |
| Target | 95.0% |
| Gap | −1.2% (slightly overconfident) |

**Coverage by prediction decile** — miscalibration is concentrated in the upper half of predictions (expensive homes):

| Decile | Coverage | Interpretation |
|--------|:--------:|---------------|
| 0 (cheapest) | 94.4% | Well calibrated |
| 1–4 | 95.6–97.6% | Slightly wide intervals |
| 5–6 | 91.3–92.5% | Mild overconfidence |
| 7–8 | 88.1–90.6% | Meaningful overconfidence |
| 9 (most expensive) | 94.2% | Recovers at very top |

> The model is most uncertain about mid-to-high value homes, likely due to the $500K cap in the dataset creating a spike that the Gaussian posterior struggles to model.

---

### Residual Diagnostics (Poly(2)+Ridge)

| Statistic | Value | Ideal |
|-----------|-------|-------|
| Residual mean | 0.0 | ≈ 0 ✓ |
| Residual std | 0.6 | — |
| Residual skew | 0.987 | ≈ 0 ✗ |
| Residual kurtosis | 4.972 | ≈ 0 ✗ |

Positive skew and heavy kurtosis reflect the **$500K ceiling effect** in California Housing — a structural data truncation artifact, not a modelling failure. The log1p transform reduces but cannot eliminate this.

---

### Optuna Hyperparameter Tuning

80 trials × 5-fold CV per model using TPE (Tree-structured Parzen Estimator) sampler.

| Model | sklearn CV RMSE ($) | Optuna RMSE ($) | Gain |
|-------|:-------------------:|:---------------:|:----:|
| Ridge | 70,443 | 70,418 | +$25 |
| Lasso | 70,695 | 70,697 | −$2 |
| ElasticNet | 70,276 | 70,477 | −$201 |
| BayesianRidge | 70,169 | 70,170 | −$2 |

**Finding:** sklearn's RidgeCV / LassoCV already explore the hyperparameter space efficiently for single-parameter models. Optuna's advantage is most visible for **ElasticNet** (2 params) and more complex architectures. For linear models with a well-chosen alpha grid, sklearn CV is already near-optimal.

**Best Optuna parameters:**

| Model | Parameter | Value |
|-------|-----------|-------|
| Ridge | alpha | 3.273 |
| Lasso | alpha | 0.0001 |
| ElasticNet | alpha | 0.000213 |
| ElasticNet | l1_ratio | 0.019 |
| BayesianRidge | alpha_1 | 9.82e-02 |
| BayesianRidge | alpha_2 | 1.08e-07 |
| BayesianRidge | lambda_1 | 9.99e-02 |
| BayesianRidge | lambda_2 | 5.52e-06 |

---

### SHAP Feature Importance

`LinearExplainer` (exact, no sampling) on Optuna-tuned Ridge and Poly(2)+Ridge.

#### Ridge — Top features by mean |SHAP|

| Rank | Feature | Mean \|SHAP\| | Direction |
|------|---------|:------------:|-----------|
| 1 | `MedInc` | 0.0751 | High income → price UP |
| 2 | `income_rooms` | 0.0467 | Engineered: wealth × space → price UP |
| 3 | `log_ave_occup` | 0.0346 | High occupancy → price DOWN |
| 4 | `Latitude` | 0.0312 | Higher latitude (NorCal) → price UP |
| 5 | `dist_to_LA` | 0.0246 | Far from LA → price DOWN |
| 6 | `rooms_per_person` | 0.0235 | More space per person → price UP |
| 7 | `Longitude` | 0.0220 | Coastal (more negative) → price UP |
| 8 | `HouseAge` | 0.0186 | Older homes in desirable areas → price UP |

#### Poly(2)+Ridge vs Ridge — Amplification ratios

| Feature | Ridge \|SHAP\| | Poly \|SHAP\| | Ratio |
|---------|:-------------:|:------------:|:-----:|
| `bedroom_ratio` | 0.0135 | 0.0327 | **2.42×** — largest amplification |
| `dist_to_LA` | 0.0246 | 0.0464 | **1.89×** |
| `AveRooms` | 0.0113 | 0.0213 | **1.89×** |
| `HouseAge` | 0.0186 | 0.0297 | **1.60×** |
| `AveOccup` | 0.0170 | 0.0256 | **1.51×** |
| `Latitude` | 0.0312 | 0.0433 | 1.39× |
| `MedInc` | 0.0751 | 0.0702 | 0.93× *(linear form better)* |

> Features with ratio > 1 gain explanatory power through polynomial interactions.
> `bedroom_ratio` (2.42×) and `dist_to_LA` (1.89×) benefit most — their non-linear
> and interaction effects matter more than their linear signals alone.
> `MedInc` (0.93×) is already near-linear in its relationship with price.

---

## Key Takeaways

| Topic | Finding |
|-------|---------|
| **Feature engineering** | Reduced OLS RMSE by $4K (~5%) before touching any model — biggest single lever |
| **Regularisation** | Lasso zeroed only 1 of 26 features — engineered matrix is clean, low-redundancy |
| **Polynomial expansion** | 19.2% RMSE reduction over linear Ridge — strongest model improvement |
| **Bayesian uncertainty** | 93.8% empirical coverage vs 95% target — slight overconfidence in mid-high price range |
| **Optuna vs sklearn CV** | Negligible gain for single-param linear models; increases in value with model complexity |
| **SHAP** | `MedInc` is the dominant predictor; engineered features `income_rooms` and `log_ave_occup` rank 2nd and 3rd — validates the feature engineering choices |