# %% [markdown]
# # 02 — Clustering Exploration
#
# Run PCA and KMeans clustering, visualize PC pairs,
# inspect cluster quality metrics, and compare balanced vs standard clusters.

# %%
import sys
sys.path.insert(0, "..")

import pandas as pd
from pathlib import Path

from trading_crab.config import (
    SNAPSHOT_DATE_PREPARED, CLUSTERING_FEATURES, configure_logging,
)
from trading_crab.data_ingestion import load_checkpoint
from trading_crab.clustering import run_pca, cluster_regimes, evaluate_kmeans
from trading_crab.plotting import (
    plot_pc_pairs, plot_kmeans_scores,
)

configure_logging(verbose=True)
DATA_DIR = Path("../data")

# %% [markdown]
# ## Load prepared data and run PCA

# %%
quarterly_df = load_checkpoint(DATA_DIR, "prepared_quarterly_data_smoothed", SNAPSHOT_DATE_PREPARED)
reduced_df, pca, market_codes = run_pca(quarterly_df)

print(f"PCA explained variance: {pca.explained_variance_ratio_.round(3)}")
print(f"Total explained: {pca.explained_variance_ratio_.sum():.1%}")
reduced_df.head()

# %% [markdown]
# ## Cluster quality sweep

# %%
from sklearn.preprocessing import StandardScaler
import numpy as np

X_cluster = StandardScaler().fit_transform(reduced_df.values)
scores = evaluate_kmeans(X_cluster, range(2, 12))
plot_kmeans_scores(scores)
scores.sort_values("silhouette", ascending=False)

# %% [markdown]
# ## Apply clustering

# %%
reduced_df, scores, best_k = cluster_regimes(reduced_df)
print(f"Best k = {best_k}")
print(f"Cluster sizes (standard):  {reduced_df['cluster'].value_counts().sort_index().to_dict()}")
print(f"Cluster sizes (balanced):  {reduced_df['balanced_cluster'].value_counts().sort_index().to_dict()}")

# %% [markdown]
# ## Visualize PC pairs colored by Grok labels vs our clusters

# %%
plot_pc_pairs(reduced_df, market_codes, title_prefix="Grok labels: ")

# %%
plot_pc_pairs(reduced_df, reduced_df["balanced_cluster"].values,
              title_prefix="Balanced clusters: ")
