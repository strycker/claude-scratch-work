# %% [markdown]
# # 03 — Regime Analysis, Prediction & Portfolio
#
# Profile regimes, train supervised models, analyze asset returns,
# and generate portfolio recommendations.

# %%
import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
from pathlib import Path

from trading_crab.config import (
    SNAPSHOT_DATE_PREPARED, SNAPSHOT_DATE_MULTPL, SNAPSHOT_DATE_FRED,
    SNAPSHOT_DATE_GROK, CLUSTERING_FEATURES, configure_logging,
)
from trading_crab.data_ingestion import load_checkpoint, merge_all_sources
from trading_crab.clustering import run_pca, cluster_regimes
from trading_crab.regime_analysis import (
    profile_regimes, name_regimes,
    compute_transition_matrix, compute_forward_probabilities,
)
from trading_crab.supervised import (
    prepare_supervised_data, train_decision_tree,
    train_random_forest, generate_classification_report,
)
from trading_crab.asset_returns import (
    compute_quarterly_returns, returns_by_regime, best_assets_per_regime,
)
from trading_crab.portfolio import (
    simple_regime_portfolio, blended_regime_portfolio,
    generate_recommendation,
)
from trading_crab.plotting import (
    plot_regime_profile_heatmap, plot_transition_matrix,
    plot_feature_importances, plot_portfolio_recommendation,
)

configure_logging(verbose=True)
DATA_DIR = Path("../data")

# %% [markdown]
# ## Setup: load data and cluster

# %%
quarterly_df = load_checkpoint(DATA_DIR, "prepared_quarterly_data_smoothed", SNAPSHOT_DATE_PREPARED)
reduced_df, pca, market_codes = run_pca(quarterly_df)
reduced_df, scores, best_k = cluster_regimes(reduced_df)
cluster_labels = reduced_df["balanced_cluster"].values

# %% [markdown]
# ## Regime Profiling

# %%
profile = profile_regimes(quarterly_df, cluster_labels)
regime_names = name_regimes(profile)
print("Regime names:", regime_names)

# %%
plot_regime_profile_heatmap(profile, regime_names)

# %% [markdown]
# ## Transition Matrix

# %%
trans = compute_transition_matrix(cluster_labels)
print(trans.round(2))
plot_transition_matrix(trans, regime_names)

# %%
fwd = compute_forward_probabilities(cluster_labels, horizons=[1, 4, 8])
for h, probs in fwd.items():
    print(f"\nForward probabilities (horizon = {h} quarters):")
    print(probs.round(2))

# %% [markdown]
# ## Supervised Prediction

# %%
X, y, feat_names = prepare_supervised_data(quarterly_df, cluster_labels)

dt_model, dt_metrics = train_decision_tree(X, y, feat_names)
print(f"\nDecision Tree CV: {dt_metrics['mean_cv_accuracy']:.3f} ± {dt_metrics['std_cv_accuracy']:.3f}")

rf_model, rf_metrics = train_random_forest(X, y, feat_names)
print(f"Random Forest CV:  {rf_metrics['mean_cv_accuracy']:.3f} ± {rf_metrics['std_cv_accuracy']:.3f}")

# %%
print(generate_classification_report(rf_model, X, y, regime_names))
plot_feature_importances(rf_metrics["feature_importances"])

# %% [markdown]
# ## Asset Returns by Regime

# %%
# Load full (pre-derivative) data for return computation
multpl_df = load_checkpoint(DATA_DIR, "multpl_datasets_snapshot", SNAPSHOT_DATE_MULTPL)
fred_df = load_checkpoint(DATA_DIR, "fred_api_datasets_snapshot", SNAPSHOT_DATE_FRED)
grok_df = load_checkpoint(DATA_DIR, "grok_quarter_classifications", SNAPSHOT_DATE_GROK)
full_df = merge_all_sources(multpl_df, fred_df, grok_df)

returns = compute_quarterly_returns(full_df)
aligned_labels = cluster_labels[1:len(returns) + 1]
regime_ret = returns_by_regime(returns, aligned_labels)
rankings = best_assets_per_regime(regime_ret)
print(rankings.round(4))

# %% [markdown]
# ## Portfolio Construction

# %%
current_regime = int(cluster_labels[-1])
print(f"Current regime: {current_regime} ({regime_names.get(current_regime, '?')})")

simple = simple_regime_portfolio(regime_ret, current_regime)
print("\nSimple portfolio:")
print(simple)

# %%
if 1 in fwd and current_regime in fwd[1].index:
    regime_probs = fwd[1].loc[current_regime]
    blended = blended_regime_portfolio(regime_ret, regime_probs)
    print("\nBlended portfolio:")
    print(blended)

    rec = generate_recommendation(current_weights=None, target_weights=blended)
    print("\nTrade recommendation:")
    print(rec)
    plot_portfolio_recommendation(rec)
