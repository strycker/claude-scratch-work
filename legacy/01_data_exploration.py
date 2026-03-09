# %% [markdown]
# # 01 — Data Exploration
#
# Load the scraped/checkpointed data and explore it visually.
# This notebook assumes you've already run the pipeline at least once
# with `--recompute` to generate pickle checkpoints.
#
# ```bash
# python -m trading_crab.pipeline --recompute -v
# ```

# %%
import sys
sys.path.insert(0, "..")

import pandas as pd
from pathlib import Path

from trading_crab.config import (
    SNAPSHOT_DATE_MULTPL, SNAPSHOT_DATE_FRED, SNAPSHOT_DATE_PREPARED,
    CLUSTERING_FEATURES, INITIAL_FEATURES, configure_logging,
)
from trading_crab.data_ingestion import load_checkpoint
from trading_crab.plotting import plot_raw_series, plot_time_series

configure_logging(verbose=False)
DATA_DIR = Path("../data")

# %% [markdown]
# ## Load checkpointed data

# %%
multpl_df = load_checkpoint(DATA_DIR, "multpl_datasets_snapshot", SNAPSHOT_DATE_MULTPL)
print(f"multpl_df: {multpl_df.shape}")
multpl_df.head()

# %%
prepared_df = load_checkpoint(DATA_DIR, "prepared_quarterly_data_smoothed", SNAPSHOT_DATE_PREPARED)
print(f"prepared_df: {prepared_df.shape}")
print(f"NaN count: {prepared_df.isna().sum().sum()}")
prepared_df.head()

# %% [markdown]
# ## Plot raw multpl.com series

# %%
# Plot a subset of the raw scraped data
plot_raw_series(multpl_df, columns=["sp500", "cpi", "gdp_growth", "us_infl", "10yr_ustreas"])

# %% [markdown]
# ## Inspect the prepared (post-derivative) dataset

# %%
print("Clustering features:")
for i, f in enumerate(CLUSTERING_FEATURES):
    print(f"  {i+1:2d}. {f}")

# %%
prepared_df[CLUSTERING_FEATURES].describe().round(4)
