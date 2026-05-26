# Charts and heatmaps (PK cohort)

> Data: `data/questions.jsonl` + `data/replies.jsonl`  
> Command: `bash scripts/run_stats.sh`

## Conventions

- **PK cohort**: public 4×200 + DISCBench (Ours) first 200 = 1000 items
- **Primary score**: `mean(1_score, 3_score)` (GPT-5.4, all sources)
- **Discrimination D**: (high 27% − low 27% model means) / range; ≥4 models required
- **Display**: Ours → DISCBench

## Output locations

| Type | Path |
|------|------|
| Paper workbook | `output/reports/paper_benchmark_tables.xlsx` |
| Full stats | `output/reports/comprehensive_benchmark_stats.xlsx` |
| PNG figures | `output/reports/charts/` |

## Active chart files (`charts/`)

| File | Description |
|------|-------------|
| `01_difficulty_tier_mean_score_by_source.png` | Difficulty tier × dataset — item count + mean score heatmaps |
| `01b_difficulty_tier_disc_by_source.png` | Difficulty tier × dataset — mean discrimination D |
| `01c_difficulty_tier_mean_score_lines.png` | Mean score by difficulty tier (line chart; English axes) |
| `02_discrimination_by_difficulty.png` | Fine-grained difficulty bins vs mean D |
| `03_difficulty_vs_checkpoint_all.png` | Checkpoint count bin × difficulty tier |
| `04_query_length_by_source.png` | Mean instruction length by dataset |
| `05_discbench_12_models_mean.png` | DISCBench 500 items · 12-model means |
| `06_models_x_sources.png` | 12 models × 5 datasets |
| `07_discbench_l1_x_12models.png` | DISCBench L1 × 12 models |
| `07b_l1_x_8models_pk.png` | PK cohort L1 × 8 models |
| `09_source_constraint_pct.png` | Dataset × six constraint types (%) |
| `pk_difficulty_vs_mean_score.png` | PK · difficulty tier × mean-score tier counts |

> Older PNG/XLSX directly under `output/reports/` (e.g. `pk_discrimination_*.png`, `compared_difficulty_disc_heatmap.xlsx`) are **not** produced by the current pipeline; safe to delete locally.

## Reproduce

```bash
cd DISCBench
bash scripts/setup.sh
bash scripts/run_stats.sh
```
