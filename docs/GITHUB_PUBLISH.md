# Publishing DISCBench to GitHub (anonymous supplementary)

Use this checklist when creating a **standalone** public repo for double-blind review or supplementary material.

**Root directory** = contents of `DISCBench/` only (not the parent mono-repo unless you intentionally nest it).

---

## 1. What reviewers need

| Goal | Required files |
|------|----------------|
| **Open benchmark data** (1000 PK items, nested JSONL) | `data/DISCbench_data.jsonl` (200) + `data/public_benchmark_data.jsonl` (800) |
| **Reproduce paper stats** (tables + charts) | `data/questions.jsonl` + `data/replies.jsonl` **or** legacy `data/论文数据_all.xlsx` + `data/replies_compared_all.xlsx` |
| **Run stages 1–3** (optional) | `config.py`, `data/models.xlsx`（仅此一张，含Compared 12 路由）, sysprompts, stage configs |

Stats logic is unchanged: PK cohort = 4×200 public + Ours first 200 by `qid`; primary score = `mean(1_score, 3_score)` (GPT-5.4).

---

## 2. Files to commit

### Include

| Path | Role |
|------|------|
| `data/DISCbench_data.jsonl` | Open DISCBench PK 200 (nested schema) |
| `data/public_benchmark_data.jsonl` | Open public 4×200 (nested schema) |
| `data/questions.jsonl`, `data/replies.jsonl` | Canonical flat store for stats (recommended) |
| `data/models.xlsx`, `data/schema.xlsx` | Stage（模型路由仅存 `models.xlsx`） / enrich |
| `data/sysprompts/`, `data/config/` | Stage configs |
| `evaluation/`, `clients/`, `scripts/`, `requirements.txt` | Code |
| `README.md`, `docs/`, `config.example.py` | Docs |

### Optional (smaller clone without full flat JSONL)

| Path | Role |
|------|------|
| `data/论文数据_all.xlsx`, `data/replies_compared_all.xlsx` | Rebuild flat JSONL via `consolidate_discbench_data.py` |

### Exclude (`.gitignore`)

| Path | Reason |
|------|--------|
| `config.py` | API secrets |
| `output/` | Generated reports |
| `.pydeps/` | Local vendored libs |
| `data/_staging/` | Generated staging xlsx |
| `*.xlsx.bak*`, `data/*.bak*` | Backups |

---

## 3. Git LFS (large data)

Track Excel and JSONL data files:

```bash
cd DISCBench
git lfs install
git lfs track "*.xlsx"
git lfs track "*.jsonl"
git add .gitattributes
```

Approximate sizes: `replies.jsonl` ~83 MB, `public_benchmark_data.jsonl` ~19 MB, `DISCbench_data.jsonl` ~14 MB, legacy xlsx ~25 MB.

---

## 4. Pre-push verification (local)

```bash
bash scripts/preflight_release.sh
```

This runs: `check_setup` → data/open JSONL audit → `verify_data_for_stats` → `run_stats.sh`.

Quick data-only check:

```bash
bash scripts/check_setup.sh
python3 scripts/verify_data_for_stats.py
bash scripts/run_stats.sh
```

Rebuild open bundles from canonical JSONL:

```bash
python3 scripts/consolidate_discbench_data.py   # if only legacy xlsx present
python3 scripts/export_discbench_open_jsonl.py
python3 scripts/export_public_open_jsonl.py
python3 scripts/fix_open_jsonl_encoding.py --no-bootstrap
python3 scripts/fix_open_jsonl_encoding.py --input data/public_benchmark_data.jsonl --no-bootstrap
```

---

## 5. Initialize and push

```bash
cd DISCBench
git init
git add .
git commit -m "DISCBench compared benchmark: open JSONL + stats reproduction"
git branch -M main
git remote add origin git@github.com:<org-or-anon-account>/<repo>.git
git push -u origin main
```

**Anonymous tips:** neutral repo name; no real names in commit messages; reviewers only need `run_stats.sh` (no API keys).

---

## 6. Paper reproducibility text

```text
Code and data: https://github.com/<account>/<repo>
Open PK subset (1000 items): data/DISCbench_data.jsonl + data/public_benchmark_data.jsonl
Reproduce tables/figures: bash scripts/setup.sh && bash scripts/run_stats.sh
Primary artifact: output/reports/paper_benchmark_tables.xlsx
```

---

## 7. Post-push smoke test (clean clone)

```bash
git clone https://github.com/<account>/<repo>.git
cd <repo>
bash scripts/setup.sh
bash scripts/preflight_release.sh
# or: bash scripts/run_stats.sh && test -f output/reports/paper_benchmark_tables.xlsx
```

---

## 8. Architecture notes (JSONL-first)

- **Stats** read `data/questions.jsonl` + `data/replies.jsonl` via `lib/jsonl_store.py` (not raw xlsx).
- **Stages 1–3** use `data/_staging/*.xlsx` synced by `scripts/stage_sync.sh`.
- **Open export** uses nested schema in `scripts/lib/open_jsonl_export.py`.
- Removed from package: `scripts/legacy/*`, `sync_replies_from_projects_cif.py`.
