# Scripts index

## Entry points (use these)

| Command | Purpose |
|---------|---------|
| `bash scripts/setup.sh` | Install Python deps, scaffold `config.py`, mirror data layout |
| `bash scripts/check_setup.sh` | Preflight before stats or stages |
| `bash scripts/preflight_release.sh` | **Pre-GitHub push**: setup + data audit + verify + run_stats |
| **`bash scripts/run_stats.sh`** | 离线统计：生成 **`paper_benchmark_tables.xlsx`** + **`output/reports/charts/*.png`**；合并后删除 `comprehensive_benchmark_stats.xlsx` 与 `benchmark_source_model_summary.xlsx`，不保留中间 Excel |
| `bash scripts/prepare_for_stats.sh` | Enrich question table → verify → stats |
| `bash scripts/run_all_stages.sh` | Online: stage1 → stage2 → stage3 → stats |

## Statistics (`scripts/stats/`)

| Script | Output |
|--------|--------|
| `generate_comprehensive_benchmark_stats.py` | （由 `run_stats.sh` 调用）写临时 comprehensive xlsx → 图表 PNG；最终被 `run_stats.sh` 删除临时文件 |
| `generate_benchmark_source_summary_report.py` | （同上）临时 benchmark summary xlsx，合并后即删 |
| `generate_paper_benchmark_workbook.py` | 合并为上表的 **`paper_benchmark_tables.xlsx`**（唯一提交的统计簿） |

**单次统计产物（入库推荐）：** 仅 **`output/reports/paper_benchmark_tables.xlsx`** + **`output/reports/charts/*.png`**（见仓库根 `.gitignore`）。

Shared logic: `lib/cif_stats_common.py`, `lib/paths.py`.

## Data maintenance (optional)

| Script | Purpose |
|--------|---------|
| `enrich_questions_table.py` | `query_len`, constraints, L1/L2/L3 from schema |
| `verify_data_for_stats.py` | 统计前覆盖检查；默认可写审计表 `--output`，`--no-excel` 仅终端输出（preflight 使用） |
| `import_seven_models_jsonl.py` | Merge `output/replies/七个模型/*.jsonl` into replies xlsx |
| `prepare_stage1_round2_sheets.py` | Dual-round IQ sheet setup |
| `merge_iq_dual_sheet_avg.py` | Average round1+round2 → `difficulty_score` |
| `export_discbench_open_jsonl.py` | Export `data/DISCbench_data.jsonl` (PK 200; optional `--ranked-subset 100`) |
| `export_public_open_jsonl.py` | Export `data/public_benchmark_data.jsonl` (4×200 public cohort) |
| `rebuild_open_datasets.sh` | Consolidate + export both open bundles + encoding fix |
| `consolidate_discbench_data.py` | Legacy xlsx → `questions.jsonl` + `replies.jsonl` + `discbench_master.xlsx` |
| `bootstrap_from_open_jsonl.py` | `DISCbench_data.jsonl` → canonical flat jsonl |
| `fix_open_jsonl_encoding.py` | Repair encoding in open nested JSONL (`--input` path) |
| `stage_sync.sh` | `push` / `pull` between JSONL and `data/_staging/*.xlsx` |

## Stage shells

`run_stage1.sh`, `run_stage2.sh`, `run_stage3.sh`, `run_stage1_round2.sh`, `wait_round2_then_merge.sh` — wrap `python3 -m evaluation.main`.
