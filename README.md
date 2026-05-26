# DISCBench Compared Benchmark — Reproduction Package

Four-stage pipeline with **JSONL as the canonical data store**. Evaluation stages use a thin Excel staging bridge; statistics read JSONL directly.

**Working directory:** Open a terminal in **`DISCBench/`** (the repo root you clone), then run `bash scripts/...`. Individual scripts `cd` to that root internally, but paths in docs assume this root.

```
Stage 1  instruction quality   →  updates questions.jsonl
Stage 2  reply generation      →  updates replies.jsonl
Stage 3  reply evaluation      →  updates replies.jsonl (judge scores)
Stage 4  statistics & charts   →  output/reports/  (offline)
```

---

## Quick start (offline stats)

```bash
bash scripts/setup.sh
bash scripts/check_setup.sh
bash scripts/run_stats.sh
```

**Output:** `output/reports/paper_benchmark_tables.xlsx` · **Charts:** `output/reports/charts/`

If `data/questions.jsonl` is missing but legacy xlsx exists, `setup.sh` runs `consolidate_discbench_data.py` automatically.

**Before pushing to GitHub:** `bash scripts/preflight_release.sh` (see [docs/GITHUB_PUBLISH.md](docs/GITHUB_PUBLISH.md)).

---

## Data layout

| Path | Role |
|------|------|
| `data/questions.jsonl` | Question bank (all compared sources) |
| `data/replies.jsonl` | 12-model replies + GPT-5.4 scores |
| `data/discbench_master.xlsx` | Optional single workbook (Ours + 12 models) |
| `data/DISCbench_data.jsonl` | Open PK 200 (DISCBench, nested JSONL) |
| `data/public_benchmark_data.jsonl` | Open 800 (4 public benchmarks, nested JSONL) |
| `data/_staging/` | Auto-generated xlsx for stages (do not edit manually) |

**Migrate from old split xlsx:**

```bash
python3 scripts/consolidate_discbench_data.py
python3 scripts/consolidate_discbench_data.py --prune   # remove legacy tables & output/questions|replies
```

---

## Four stages (online, needs API)

```bash
cp config.example.py config.py
bash scripts/run_stage1.sh    # push jsonl→staging → IQ → pull staging→jsonl
bash scripts/run_stage2.sh    # reply generation
bash scripts/run_stage3.sh    # LLM judge
bash scripts/run_stats.sh       # stage 4
```

Or: `bash scripts/run_all_stages.sh`

Stage configs: `data/config/stage*.json` → `data/_staging/questions.xlsx` / `replies.xlsx`.

---

## Statistical conventions

| Concept | Definition |
|---------|------------|
| Primary score | `mean(1_score, 3_score)` (GPT-5.4) |
| PK cohort | Public 4×200 + DISCBench first 200 = 1000 items |
| 12 models | DISCBench full evaluation set |
| Display | `Ours` → **DISCBench** |

See `scripts/lib/cif_stats_common.py`.

---

## Project structure

```
DISCBench/
├── data/
│   ├── questions.jsonl          # canonical questions
│   ├── replies.jsonl            # canonical replies (12 models)
│   ├── DISCbench_data.jsonl     # open PK 200 export
│   ├── public_benchmark_data.jsonl  # open 4×200 public export
│   ├── discbench_master.xlsx    # optional consolidated excel
│   └── config/                  # stage1–3 JSON configs
├── evaluation/                  # stages 1–3 engine
├── scripts/
│   ├── run_stats.sh             # stage 4
│   ├── stage_sync.sh            # jsonl ↔ staging
│   ├── consolidate_discbench_data.py
│   ├── export_discbench_open_jsonl.py
│   ├── lib/jsonl_store.py
│   └── stats/
└── output/reports/              # generated (gitignored)
```

---

## Scripts index

See [scripts/README.md](scripts/README.md).

---

## Docs

| Doc | Topic |
|-----|--------|
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | Short reproduction index |
| [docs/PIPELINE.md](docs/PIPELINE.md) | Command cheat sheet |
| [docs/GITHUB_PUBLISH.md](docs/GITHUB_PUBLISH.md) | Anonymous GitHub release |
| [data/README.md](data/README.md) | Data files |

---

## Requirements

Python 3.9+, `pip install -r requirements.txt`. Offline stats need no `config.py`.
