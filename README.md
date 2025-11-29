# Customer Churn Prediction and Segmentation

## Overview

This project predicts customer churn and segments customers for targeted retention strategies. It provides:

- A training and evaluation pipeline for churn prediction (RandomForest, XGBoost, LightGBM).
- K-Means segmentation including churn probability as a clustering feature.
- An interactive Streamlit dashboard for exploration and suggested retention actions.

## Files & Key Components

- [app.py](app.py) — Streamlit dashboard. Key functions include:
  - [`app.load_data`](app.py) — Loads `Data/customer_data_with_clusters.csv`.
  - [`app.cluster_summary`](app.py) — Generates per-cluster aggregated metrics and churn lift.
  - [`app.retention_priority_score`](app.py) — Heuristic priority score.
  - [`app.strategy_for_cluster`](app.py) — Suggested actions for each cluster.
- [Customer Churn Prediction and Segmentation.ipynb](Customer Churn Prediction and Segmentation.ipynb) — Notebook with data cleaning, feature engineering, modeling, and clustering steps. It creates:
  - `Data/customer_data_with_clusters.csv` (used by the dashboard)
  - Saved models in `Model_Artifacts/` (e.g., LightGBM and KMeans models).
- Data:
  - [`Data/WA_Fn-UseC_-Telco-Customer-Churn.csv`](Data/WA_Fn-UseC_-Telco-Customer-Churn.csv) — Original dataset (Telco Customer Churn).
  - [`Data/customer_data_with_clusters.csv`](Data/customer_data_with_clusters.csv) — Processed dataset with `Churn_Probability` and `Cluster` columns.
- `Model_Artifacts/` — Model files (e.g., `LightGBM_churn_model.pkl`, `KMeans_churn_clusters.pkl`).

## How to run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. If you want to create the processed dataset and models, open and run the notebook:

- [Customer Churn Prediction and Segmentation.ipynb](Customer Churn Prediction and Segmentation.ipynb)

3. Launch the dashboard (it expects `Data/customer_data_with_clusters.csv` to be present):

```bash
streamlit run app.py
```

- Local link: http://localhost:8501
- Deployed link: <<PLACEHOLDER FOR DEPLOYED STREAMLIT APP URL>>

## Dashboard features

The Streamlit app ([app.py](app.py)) provides:

- Global KPIs: Selected customers, average churn probability, avg monthly revenue, avg tenure.
- Cluster summary and churn lift computed with [`app.cluster_summary`](app.py).
- Priority scoring per cluster using:
  $$
  score = avg\_churn \times avg\_monthly \times \log(1 + cluster\_size)
  $$
  (Implemented by [`app.retention_priority_score`](app.py))
- Suggested retention actions per cluster (via [`app.strategy_for_cluster`](app.py)).
- Visual insights: cluster bar chart + churn line, MonthlyCharges vs Churn scatter, contract vs payment heatmaps, feature differences vs global mean, churn by tenure buckets.
- A top-customers list (risk x value) and CSV download for the selected filters.

## Model comparison

All metrics below are from the notebook experiments in the repository. Use them as a reference — metric values are based on the notebook's runs.

| Model        | Test Accuracy | Train Accuracy | Recall (Churn=1) |    AUC Score |
| ------------ | ------------: | -------------: | ---------------: | -----------: |
| RandomForest |  0.7270788913 |   0.7500444444 |             0.81 | 0.8303044971 |
| XGBoost      |  0.7007818053 |   0.7239111111 |             0.83 | 0.8249439616 |
| LightGBM     |  0.7277896233 |   0.7575111111 |             0.82 | 0.8364363698 |

Recommendation:

- LightGBM shows the best trade-off (highest AUC and highest test accuracy) and is selected as the primary model in the notebook.
- XGBoost had the best recall (slightly higher recall), which may be preferred if the business objective prioritizes identifying as many churners as possible.

## Key findings

- LightGBM performs best overall (AUC ≈ 0.836).
- High-risk clusters (per notebook) are often characterized by:
  - Month-to-month contracts
  - Electronic check payment method
  - Fiber optic internet service
  - Lower tenure and total charges
- Clustering (KMeans, $K=4$) helps prioritize groups for retention campaigns.
- Priority score weights churn probability, average monthly ARPU, and cluster size logarithmically; it helps identify high-impact clusters to focus on.

## Reproducibility notes

- The notebook writes out `Data/customer_data_with_clusters.csv` and saves model artifacts to `Model_Artifacts/`. If you want an identical dashboard experience, ensure these outputs are present or re-run the notebook to regenerate them.
- The dashboard ([app.py](app.py)) reads `Data/customer_data_with_clusters.csv` via [`app.load_data`](app.py).

## Next steps & improvements

- Add model explainability (SHAP or per-customer feature importances) to the dashboard.
- Add model calibration and threshold selection to optimize retention budgets.
- Add an automated CI step to regenerate `Data/customer_data_with_clusters.csv` and re-save models after retraining.
- Deploy the Streamlit app, then put the public URL at the "Deployed link" location above.

## License & Credits

- Dataset: Telco Customer Churn dataset (Kaggle).
- This project combines model training/selection and clustering to support targeted retention actions, with a Streamlit-based operational dashboard.
