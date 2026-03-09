"""
Grok label ingestion — load the externally generated LLM quarter classifications.

The grok pickle contains a DataFrame with a ``primary_class`` categorical column
indexed by quarter-end dates.  We convert the categories to integer codes so the
values can serve as a ``market_code`` column in the feature DataFrame.

This is the initial seed for market_code — a coarse LLM-assisted classification
used primarily for visual overlay in notebooks.  It is **not** used for model
training.  The market_code column it produces can later be replaced by:

  - balanced_cluster labels from unsupervised KMeans (step 3)
  - predicted labels from the supervised classifier (step 5)

All market_code variants are stored via CheckpointManager under the name
``market_code_{source}`` (e.g. ``market_code_grok``, ``market_code_clustered``).

Usage:
    from market_regime.ingestion.grok import load_grok_labels
    mc = load_grok_labels(data_dir)   # pd.Series or None
"""

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Pattern used to search for the grok pickle file in data_dir
GROK_GLOB = "grok_quarter_classifications_*.pickle"


def load_grok_labels(data_dir: Path) -> pd.Series | None:
    """
    Locate and load the grok classification pickle, return a ``market_code`` Series.

    Args:
        data_dir: root data directory (typically DATA_DIR from market_regime.__init__)

    Returns:
        pd.Series of integer market_code values indexed by quarter-end dates,
        or None if no grok file is found.

    The Series is named ``market_code`` and has a DatetimeIndex aligned to
    quarter-end dates (the same freq as the rest of the pipeline).
    """
    candidates = sorted(data_dir.glob(GROK_GLOB))
    if not candidates:
        log.warning(
            "No grok pickle found in %s (pattern: %s) — market_code unavailable",
            data_dir,
            GROK_GLOB,
        )
        return None

    path = candidates[-1]  # take the most recent one (last alphabetically by date)
    log.info("Loading grok labels from %s", path)

    try:
        df = pd.read_pickle(path)
    except Exception as exc:
        log.error("Failed to load grok pickle %s: %s", path, exc)
        return None

    if "primary_class" not in df.columns:
        log.error(
            "Grok pickle %s has no 'primary_class' column — found: %s",
            path,
            list(df.columns),
        )
        return None

    # Convert categorical to integer codes (0-based)
    cat = df["primary_class"]
    if not hasattr(cat, "cat"):
        cat = cat.astype("category")
    codes = cat.cat.codes.astype(int)

    # Align index to quarter-end dates
    codes.index = pd.to_datetime(codes.index)
    codes = codes.resample("QE").last()
    codes.name = "market_code"

    # Report category mapping so users can see what each code means
    cat_map = dict(enumerate(cat.cat.categories))
    log.info(
        "Grok labels loaded: %d quarters, %d categories  %s",
        len(codes),
        len(cat_map),
        cat_map,
    )
    return codes
