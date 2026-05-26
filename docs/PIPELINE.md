# Pipeline 速查

所有命令在 **DISCBench 根目录**执行。

## 离线统计（无需 API）

```bash
bash scripts/setup.sh
bash scripts/run_stats.sh
```

输出：**`output/reports/paper_benchmark_tables.xlsx`**（仅此一份统计簿） + **`output/reports/charts/*.png`**（另有总册内 Charts sheet）。

## 数据准备 + 统计

```bash
bash scripts/prepare_for_stats.sh
```

## Stage 1–3（需 config.py + API）

```bash
bash scripts/run_stage1.sh    # 指令质量 → difficulty_score
bash scripts/run_stage2.sh    # 多模型回复 → data/replies.jsonl（经 staging 桥接）
bash scripts/run_stage3.sh    # LLM 裁判 → 1~4_score
bash scripts/run_stats.sh
```

## Stage1 双轮难度（可选）

```bash
bash scripts/run_stage1_round2.sh
python3 scripts/merge_iq_dual_sheet_avg.py --sync-all
python3 scripts/enrich_questions_table.py --sync-all
bash scripts/run_stats.sh
```

## 统计脚本（唯一入口：`run_stats.sh`）

| 脚本 | 作用 |
|------|------|
| `stats/generate_comprehensive_benchmark_stats.py` | 综合统计 + charts |
| `stats/generate_benchmark_source_summary_report.py` | 模型排名 / 裁判 |
| `stats/generate_paper_benchmark_workbook.py` | 合并论文总册 |

合并完成后，`run_stats.sh` 会删除 `comprehensive_benchmark_stats.xlsx` 与 `benchmark_source_model_summary.xlsx`，磁盘上只保留 **`paper_benchmark_tables.xlsx`** 与 **`charts/*.png`**。
