# Stale remote branches — inventory 2026-09-10

**Status: NOT deleted.** The operator authorised deletion after PR #145 merged, but this
session's git credentials cannot delete remote branches — `git push origin --delete` returns
**HTTP 403** for all 51, and the GitHub MCP server exposes `create_branch` but no delete
equivalent. The inventory below is kept so the cleanup can be done from a machine with push
rights, and so nothing is lost if it is.

## To delete them (run locally, with push rights)

```bash
git fetch origin --prune
# review first:
git branch -r | grep -v 'origin/main$' | grep -v HEAD
# then delete every remote branch except main:
git branch -r | grep -v 'origin/main$' | grep -v HEAD | sed 's#\s*origin/##' \
  | xargs -I{} git push origin --delete {}
```

GitHub's web UI also offers **Branches → delete**, and its "restore" button works for a while
after deletion.

## Recovery

Any branch here can be recreated while GitHub retains the objects:

```bash
git push origin <sha>:refs/heads/<branch-name>
```

Neither ancestry (`git branch --merged`) nor patch-id (`git cherry`) could prove these were
contained in main — expected for squash-merged PRs, whose squashed commit has a different
patch-id than the commits it replaced. Absence of proof is not proof of absence, so the SHAs
are recorded rather than relying on that judgement.

Only these 5 were provably merged (0 commits ahead of main): `claude/gsd-phase-5-closeout-vy3fk1`,
`claude/gsd-review-phase-5-vy3fk1`, `claude/gsd-review-phase-5-vy3fk1-status`,
`claude/gsd-discuss-phase-6-8wne0z`, `claude/phase-6-planning-fymr21`. The rest carry commits
not in main's history; for the 2026-03/04 branches that is largely divergence from the old
pre-platform era rather than unique work.

`main` at time of inventory: `0c4cab30ddaffe949b3b868dbfc05353233ec45b`

| branch | sha | last commit | ahead of main |
|---|---|---|---|
| `claude/analyze-and-plan-5JtZM` | `5c399ccfb3caa6a111caa2986dded9fe31dc7451` | 2026-03-27 | 213 |
| `claude/analyze-repo-setup-wm4hl` | `a19f569574b9cf7f866704e4571c8fe702d44a92` | 2026-03-20 | 170 |
| `claude/audit-and-align-d7jMa` | `5ddfd56223b87a13a797a817d54cccaaf196ad07` | 2026-03-10 | 33 |
| `claude/audit-and-plan-M6KzQ` | `84a7e207e83e7cc99fa384955b133117595fd196` | 2026-03-17 | 99 |
| `claude/bug-fixes-and-enhancements-d7jMa` | `08ff5ea0e00fbaa7960287ad0a888fef2c683ba5` | 2026-03-11 | 64 |
| `claude/check-linter-setup-8IktK` | `92ed28dba2a8f0e1bc43cee9da57294ade1e3063` | 2026-04-21 | 291 |
| `claude/compare-repos-plan-wm4hl` | `f5c8b3ed1cc266fa2a1e60cdd1a6ea7974ceeb13` | 2026-03-19 | 160 |
| `claude/egress-test-script-ywzqmy` | `b603a2c6d5cd446e54047e9a2095afca675b678b` | 2026-07-23 | 21 |
| `claude/expand-features-data-sources-d7jMa` | `bca7a4c58a86d46c61e65a30ae5e2c7a56c2a57f` | 2026-03-10 | 38 |
| `claude/expand-pipeline-monitoring-gMB4X` | `2a815eee94cfe2d852e686d7df57d9b61845b684` | 2026-03-24 | 180 |
| `claude/fix-asset-prices-checkpoint-Sqot2` | `69d4b58750fff88d0173ba709aed292e3930f2b4` | 2026-04-01 | 273 |
| `claude/fix-nat-strftime-Sqot2` | `b801d7e304f285845e9e955f4dc82c25003cc93f` | 2026-04-01 | 277 |
| `claude/fix-py310-ci` | `6d2f38ec9afd6c197d81a0d6aede1b10e6f8340b` | 2026-07-22 | 7 |
| `claude/fix-skipped-constraint-tests-Sqot2` | `b372c58e79c3db335bcc19866553e9aaa61a9f91` | 2026-04-01 | 275 |
| `claude/fix-ssl-requirements-d7jMa` | `601243437d558fe60c9ac66980835ef05cb906d5` | 2026-03-09 | 27 |
| `claude/fix-yfinance-fred-d7jMa` | `d76ec32815a097915e4ba502618184919b9a4ac7` | 2026-03-11 | 70 |
| `claude/fred-domain-whitelist-3o5n7q` | `c2dd554a4064ceefeeeae9c07f5a7347edf346af` | 2026-07-23 | 18 |
| `claude/full-pipeline-buildout-d7jMa` | `0c405c6998eb1967591e040cb912bffa2ee5d213` | 2026-03-09 | 22 |
| `claude/gsd-discuss-phase-6-8wne0z` | `bb5966683d82e65455ac796424c2b3e9506f3f3a` | 2026-09-09 | 0 |
| `claude/gsd-phase-5-closeout-vy3fk1` | `eb7f95478538b7db1fc92064b550dea080c68d21` | 2026-08-04 | 0 |
| `claude/gsd-review-phase-5-vy3fk1` | `d9c1759792b9ef8d4b3ca6859699dd31433ccaca` | 2026-08-04 | 0 |
| `claude/gsd-review-phase-5-vy3fk1-status` | `658f147ba6ea852a908bcccbc48fed17cf17062b` | 2026-08-04 | 0 |
| `claude/implement-phase-a2-SmHEA` | `ab8481d6116a76ccd2f4200c1047e361c54a0fcc` | 2026-03-30 | 240 |
| `claude/implement-phase-c6-mprNG` | `27000b1e67c19b1b6534aa83d0ca65b341cabd94` | 2026-03-26 | 196 |
| `claude/implement-phase-d2-G5aFz` | `cc5a40e152af14bfe993e32d6a3d069890cda460` | 2026-03-27 | 205 |
| `claude/improve-pca-kmeans-output-d7jMa` | `ecf3a6803ce98ca69cafb8f80876466452552c59` | 2026-03-10 | 43 |
| `claude/new-dev-branch-ls5Ya` | `0149305a8a4cc5eb7dbb6ce6fd600217c13ecc00` | 2026-04-03 | 287 |
| `claude/next-changes-86877` | `7814583939fb538ec03b3a71a0796252e8353477` | 2026-04-21 | 297 |
| `claude/notebook-improvements-d7jMa` | `229011b0e7d581cea58bfd6e0e83707b0df17fb9` | 2026-03-10 | 53 |
| `claude/p5-doc-updates-1774982009` | `94ccd158344c2054a3080c20fbf681f808c53fdf` | 2026-04-01 | 268 |
| `claude/phase-4-execution` | `05bab51340a6af75134a9d68757ca759f4e0e969` | 2026-07-23 | 5 |
| `claude/phase-5-honest-backtest` | `0f2ee4018327d44d4db788abfe19d81b27377658` | 2026-07-24 | 29 |
| `claude/phase-6-planning-fymr21` | `ee518d8f224da34c9c566f147008e969edc8104e` | 2026-09-10 | 0 |
| `claude/phase-a-simplify-47473` | `723040abbdd97a33b39f3494a2bce9f5048ded93` | 2026-03-27 | 224 |
| `claude/phase-b-decompose` | `cd0b85ab8fa05be56e828fc441c33bc5c97071a3` | 2026-03-28 | 233 |
| `claude/phase-b-implementation-V4l7I` | `abc50525f1bf481ff2f0a8173dc0ea1a6dd056ac` | 2026-03-24 | 181 |
| `claude/phase-c1-implementation-3YFDk` | `30464c599d6e1288d300ea2769eaae49a04f1785` | 2026-03-25 | 189 |
| `claude/phase-e-implementation-rQww6` | `15677fc8a500000a5444b2d91b40d1ee2bcae25f` | 2026-03-27 | 210 |
| `claude/phases-e-k4-59138` | `d61039b45fbe8fec3e7ea7e024f645c18d2389a2` | 2026-04-02 | 281 |
| `claude/phases-k1-k3-64024` | `45d0cac12ad8995372a15cf0d0c1a1811c279c3a` | 2026-04-02 | 283 |
| `claude/post-release-v0.1.2-ahfh5` | `2466352078f6381abb3b57cc16b4415bcebde342` | 2026-04-02 | 279 |
| `claude/refresh-submodule-analysis-1Icoo` | `c85bebcb841359b6fa854359b9ef782bad47e0ae` | 2026-03-27 | 220 |
| `claude/review-gsd-updates-d7jMa` | `b31dcad35bfba078bf73d341cc6f90955cea865c` | 2026-03-16 | 86 |
| `claude/review-meta-plan-Sqot2` | `007d56e86cb6fe652051e216f61c29a79691b48b` | 2026-03-31 | 249 |
| `claude/review-phase-f6-TuNeE` | `375e555268e1e691166fbc796ac2c598c0d6f770` | 2026-04-02 | 285 |
| `claude/s1-migration-prep-8IktK` | `9742ca0fc633077f4d39376d9365336ea33c1eb2` | 2026-04-21 | 299 |
| `claude/setup-project-structure-d7jMa` | `bea0d3f765e15d376b7b25f6bbc26e2caffc0c49` | 2026-03-17 | 93 |
| `claude/simplify-repo-structure-d7jMa` | `c45cefab7747b244e76759a56176a3a42821832c` | 2026-03-17 | 96 |
| `claude/tier2-improvements-wm4hl` | `109718ff49842e62516185247c17e2d832654796` | 2026-03-18 | 135 |
| `claude/trading-crab-repo-strategy-u5s9lu` | `b41256822ab62d3212497b78713be299f4484444` | 2026-07-22 | 2 |
| `develop` | `08e5876c162804c41788fbfaba3c5f1a3936ba83` | 2026-03-17 | 106 |
