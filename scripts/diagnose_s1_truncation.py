"""Causal-invariance test for 08-08's S-1 halt — writes only to the --out directory given.

Runs the l2 joint leg on inputs physically truncated at T (monthly_features and
monthly_raw cut BEFORE build_inputs, so every derived frame and frozen list is
rebuilt from truncated data), then compares the filtered belief for months <= T
against the committed full-run belief matrices. Exact equality at every month
<= T means no information dated after T reached any belief value up to T.

Usage: python scripts/diagnose_s1_truncation.py 2012-08-31 /tmp/out
Registry: runs under NO_REGISTRY; appends nothing. Holdout: T <= 2020-12-31.
Run 2026-09-23 at T = 1982-08-31, 1995-03-31, 2012-08-31 — see 08-CHURN.md §5.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/user/claude-scratch-work")
sys.path.insert(0, str(REPO / "scripts"))
import run_joint_lift as R  # noqa: E402 — needs the sys.path line above

from trading_crab_lib.platform.evaluation.churn import write_probability_matrix  # noqa: E402

T = pd.Timestamp(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

real_cm = R.get_platform_checkpoint_manager()
class TruncCM:
    def load(self, name):
        df = real_cm.load(name)
        return df.loc[df.index <= T]
    def __getattr__(self, a):
        return getattr(real_cm, a)
R.get_platform_checkpoint_manager = lambda: TruncCM()

cfg = R.load_platform_config()
inp = R.build_inputs(cfg)
t0 = time.time()
curve, meta = R.run_joint_backtest(
    inp["features_1"], inp["asset_returns"],
    blend_weight_1=float(cfg.get("allocation", {}).get("blend_weight_1", 0.50)),
    trial_tag="truncation-diagnostic", cfg=cfg, features_2=inp["features_2"],
    frozen_features_1=inp["frozen_1"], frozen_features_2=inp["frozen_2"],
    routing=R.ROUTING_L2_NOWCAST, cash_returns=inp["cash_returns"],
    registry_path=R.NO_REGISTRY,
)
res = {"T": str(T.date()), "seconds": round(time.time() - t0, 1),
       "frozen_1": inp["frozen_1"], "frozen_2": inp["frozen_2"],
       "last_step": str(curve.index.max().date())}
for c in (1, 2):
    p = out / f"belief_{c}.parquet"
    write_probability_matrix(meta[f"per_step_belief_{c}"], p)
    trunc = pd.read_parquet(p)
    full = pd.read_parquet(REPO / f"outputs/reports/platform/joint_lift/joint_lift_belief_{c}_l2.parquet")
    full = full.loc[full.index <= T]
    same_idx = trunc.index.equals(full.index)
    cols = sorted(set(trunc.columns) | set(full.columns))
    a = trunc.reindex(columns=cols, fill_value=0.0)
    b = full.reindex(index=a.index, columns=cols, fill_value=0.0)
    d = float(np.nanmax(np.abs(a.to_numpy() - b.to_numpy()))) if len(a) else float("nan")
    res[f"classifier_{c}"] = {"rows_trunc": len(trunc), "rows_full_le_T": len(full),
                              "index_equal": bool(same_idx), "max_abs_diff": d,
                              "bit_identical": bool(same_idx and d == 0.0)}
(out / "result.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
