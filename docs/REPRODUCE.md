# Reproduction guide

The canonical guide is **[../README.md](../README.md)**. This file is a short index.

## Offline (paper tables + figures)

```bash
bash scripts/setup.sh
bash scripts/check_setup.sh
bash scripts/run_stats.sh
```

**Output:** `output/reports/paper_benchmark_tables.xlsx` · **Charts:** `output/reports/charts/`（与仓库根目录 `README.md` 离线复现小节一致）。

**Pre-GitHub:** `bash scripts/preflight_release.sh`

## Conventions

- Primary score: `mean(1_score, 3_score)` (GPT-5.4)
- PK cohort: 1000 items (public 4×200 + DISCBench first 200)
- Constants: `scripts/lib/cif_stats_common.py`

## More

| Topic | Doc |
|-------|-----|
| Stage 1–3 commands | [PIPELINE.md](PIPELINE.md) |
| Chart file names | [HEATMAP_GUIDE.md](HEATMAP_GUIDE.md) |
| GitHub / anonymous URL | [GITHUB_PUBLISH.md](GITHUB_PUBLISH.md) |
| Data files | [../data/README.md](../data/README.md) |
