# Data directory

## Canonical store (JSONL-first, full bank)

| File | Description |
|------|-------------|
| `questions.jsonl` | All sources (1300 rows: 4×200 public + 500 Ours) |
| `replies.jsonl` | 12 models — one row per `(qid, model)` with replies + judge scores |
| `discbench_master.xlsx` | Optional Ours-only workbook (Questions + Replies sheets) |

**PK / stats cohort (1000 items):** CFbench, infobench, ComplexBench, advancedif (200 each) + Ours first 200 by `qid`.

Rebuild from legacy Excel:

```bash
python3 scripts/consolidate_discbench_data.py
```

## Open-release bundles (nested JSONL)

| File | Items | Models | Content |
|------|-------|--------|---------|
| `DISCbench_data.jsonl` | **200** | 12 | DISCBench PK cohort (Ours前200题): instruction + IQ + 12 model replies/scores |
| `public_benchmark_data.jsonl` | **800** | 8 | Four public benchmarks (200 each): same nested schema |

One-command rebuild (consolidate + export + encoding fix):

```bash
bash scripts/rebuild_open_datasets.sh
```

Or step by step:

```bash
python3 scripts/consolidate_discbench_data.py
python3 scripts/export_discbench_open_jsonl.py      # 200 PK items
python3 scripts/export_public_open_jsonl.py         # 800 public items
python3 scripts/fix_open_jsonl_encoding.py --input data/DISCbench_data.jsonl --no-bootstrap
python3 scripts/fix_open_jsonl_encoding.py --input data/public_benchmark_data.jsonl --no-bootstrap
```

Legacy 100-item high-difficulty subset (optional):

```bash
python3 scripts/export_discbench_open_jsonl.py --ranked-subset 100
```

## Legacy import sources

| File | Role |
|------|------|
| `论文数据_all.xlsx` | Question bank (sheet `数据对齐`) |
| `replies_compared_all.xlsx` | Compared replies (12 models) |

## Other

| File | Role |
|------|------|
| `models.xlsx` | **唯一**模型路由表；仅收录 Compared 评测使用的 12 个 `模型名称` + Aimux「供应商」列（见 `scripts/lib/cif_stats_common.py` · `CANONICAL_12`）。其中 `doubao-seed-2-0-pro`、`GLM-4.7-flash` 等为与 JSONL/`replies.jsonl` 列 `model` 一致的逻辑 id，`供应商` 仍与 Aimux 侧路由对齐。 |
| `schema.xlsx` | L1/L2/L3 alignment |
| `sysprompts/*.txt`, `config/stage*.json` | Stage configs |
| `data/_staging/*.xlsx` | Generated for evaluation stages (`scripts/stage_sync.sh`) |
