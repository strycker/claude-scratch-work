---
phase: 7
slug: regime-representation
scope: phase-wave 1 ONLY (07-CONTEXT.md D-09)
status: complete
verdict: accept-with-caveats
created: 2026-09-15
signed_off: 2026-09-15
tests_total: 3
tests_passed: 3
tests_pending: 0
---

# Phase 7 (Wave 1) — UAT

Three items reached human verification. All three are closed. **Verdict:
accept-with-caveats** — matching how Phase 6's P3 cold-start gate was signed, and honest to
what was measured.

## UAT-1 — 07-03's measurement checkpoint ✅ PASSED (approved with a binding condition)

**Why human:** plan 07-03 is `autonomous: false`. Its `checkpoint:human-verify` task puts the
wave-1 numbers to a person before the ADR commits to them. An agent marking this satisfied is
the exact evidence failure `UAT-AUDIT-2026-09-09` indicts, so the executor stopped and
surfaced the table inline rather than self-approving.

**Approved**, with one binding condition attached and discharged: the criterion-3
window-narrowing must be **foregrounded inline**, at the point the numbers appear, in both the
report and ADR-0001 — never as a footnote. Verified in the rendered artifact:

```
| Labeling disagreement (`pct_disagree`, `n_compared`) | 82.77% (389/470; 1974-02 -> 2020-12) | 80.90% (288/356; 1974-02 -> 2017-05) |
```

Both denominators and both date windows sit in the same table cell as the percentages. A
reader cannot see `82.77% → 80.90%` without also seeing `470` vs `356` and `2020-12` vs
`2017-05`.

## UAT-2 — the criterion-8 claim ✅ PASSED (claim corrected; not a wave-1 defect)

**Why human:** `07-VERIFICATION.md` returned `status: human_needed` on one item — whether the
blanket "platform/ imports nothing from the legacy library" claim was an acceptable
out-of-scope inaccuracy or a sign-off blocker.

**Resolved by correcting the claim, not by softening it.** The claim was false, and so was its
evidence: `MIGRATION-PLAN.md`'s exit check ends in `| grep -v platform`, and every match line
begins with a path containing `platform`, so the filter discarded every violation — the check
**could not fail**. An AST scan finds **31 real legacy import sites**.

All 31 predate Phase 7 and **wave 1 added none**, so this is pre-existing coupling a broken
check hid, not a wave-1 regression — hence PASSED rather than a blocker. Corrected in
`ROADMAP.md` (criteria 7.8 and 8.1), `MIGRATION-PLAN.md`, `07-CONTEXT.md`, and `STATE.md`, and
now enforced by `tests/unit/test_platform_legacy_import_ratchet.py` (11 tests).

## UAT-3 — overall wave-1 sign-off ✅ PASSED (accept-with-caveats)

| Criterion | Verdict |
|---|---|
| 1 — one documented feature policy, test fails on divergence, ADR | ✅ verified by live re-derivation |
| 2 — §5.4 ratio interpretable with resolved-transition count | ✅ **0.60145, 7/7 resolved** |
| 3 — disagreement measured against the 82.8% baseline | ⚠️ satisfied as worded, **named limitation** |
| 4 — ablation delta on both axes, in band | ✅ +0.377847 / −0.066124 |

### Accepted caveats — recorded as limitations, not defects

1. **Criterion 3's comparison window.** The frozen policy degraded **232/588** L2 steps vs
   118/588, so the post-fix number rests on 356 steps ending 2017-05 against a baseline of 470
   ending 2020-12. The cause is understood: L2 degrades when any class in a training window has
   fewer than `n_splits` examples (`driver.py:161-163`), and freezing L1 changed the label
   sequence. Verified **not** a wiring defect — `frozen_l1_features` reaches only `_refit_l1`
   (`:461`, `:473`); `_refit_l2` (`:266-271`) takes no such parameter. Resolving it needs an L2
   CV design decision → **wave 2, with its own ADR**.

2. **Drawdown got worse, and the record says so.** `dd_delta` moved −0.014364 → **−0.066124**
   (4.6×) and strategy max DD −21.24% → **−26.42%**. Both in band; D-06 makes measurement the
   gate, not the sign, so neither blocks. But the regime layer buys **less** drawdown protection
   than the void pre-fix numbers suggested, and Faber (6.3726 / −18.94%) still beats the
   strategy on both §23.1 axes. The strategy remains **last of five legs**.

3. **Four `[ASSUMED]` bands remain unconfirmed.** Correct for wave 1 — D-07 makes them advisory
   flags, so nothing turned on their values. They become load-bearing at criterion 7. Confirm or
   revise before wave 2 leans on them.

### What wave 1 claimed, and did

Its job was to make the labeler's feature policy **singular** and §5.4 **interpretable**. Both
done: driver and report now fit on one computed-once column list, and the §5.4 ratio resolves
every transition (7/7). **A13 — the top open audit item — is resolved by cause**, and the caveat
came off only because the cause was fixed, with a test asserting the caveat names its own
licensing artifacts (D-08).

It did **not** claim to make the strategy good. That is wave 2's axis and beyond.
