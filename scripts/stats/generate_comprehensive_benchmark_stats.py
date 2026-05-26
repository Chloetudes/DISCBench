#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DISCBench / Compared benchmark 综合统计（论文复现用）。

输出：
  - comprehensive_benchmark_stats.xlsx（分主题 sheet，表内纵向堆叠）
  - paper_benchmark_tables.xlsx（论文用总册，推荐直接引用）
  - output/reports/charts/*.png

统计逻辑（10 项）：
  1. 难度档 × 各数据集题级均分热力图 + 难度分/均分线性相关
  2. 难度档 × 题均区分度 D（中等难度区分度最高）
  3. 难度档 × 考点数分箱热力图 + 相关
  4. 题目表全量 query 长度（1300 题）
  5. 综合对比表（来源维度）
  6. PK 对比：8 模型 × 5 数据集（各 200 题 cohort，无额外模型行）
  7. DISCBench：12 模型总榜（500 题）+ L1 × 12（仅自建集）
  8. 数据集元信息（难度/约束/任务覆盖）
  9. 数据集 × 六类约束个数热力图
 10. 展示名：Ours → DISCBench，排在末尾
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
for p in (str(_ROOT), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lib.cif_stats_common import (
    CANONICAL_8,
    CANONICAL_12,
    CHECKPOINT_LABELS,
    CONSTRAINT_COLS,
    CONSTRAINT_LABELS,
    CONSTRAINT_LABELS_EN,
    GPT54_SCORE_COLS,
    MIN_MODELS,
    OURS_SOURCE,
    PK_OURS_N,
    PRIMARY_SCORE_LABEL,
    PUBLIC_SOURCES,
    QUESTIONS_SHEET,
    resolve_questions_path,
    resolve_replies_path,
    SOURCE_ORDER,
    SOURCE_ORDER_DISPLAY,
    TIER_LABELS,
    TIER_SHORT,
    assign_tier_bin,
    apply_display_source,
    corr_pair,
    cross_tab_count,
    display_source,
    embed_images_in_excel,
    format_paper_excel_sheets,
    write_stacked_tables,
    load_questions,
    load_replies_with_scores,
    plot_bar_simple,
    plot_disc_by_difficulty_fine,
    plot_heatmap,
    prepare_analysis_tables,
    reply_row_is_success,
    score_col_for_source,
    plot_tier_count_and_mean,
    plot_tier_mean_score_lines,
    tier_count_pivot_questions,
    tier_count_pivot,
    tier_mean_score_pivot,
    tier_mean_disc_pivot,
    tier_disc_count_pivot,
    to_discbench_model,
)


def _constraint_stats(qsub: pd.DataFrame) -> Tuple[int, float]:
    """约束总数与均值（优先 constraint_n，否则六类 iq_count 求和）。"""
    if "constraint_n" in qsub.columns:
        cn = pd.to_numeric(qsub["constraint_n"], errors="coerce")
        if cn.notna().any():
            return int(cn.sum()), round(float(cn.mean()), 2)
    parts = []
    for c in CONSTRAINT_COLS:
        if c in qsub.columns:
            parts.append(pd.to_numeric(qsub[c], errors="coerce").fillna(0))
    if parts:
        total_row = sum(parts)
        return int(total_row.sum()), round(float(total_row.mean()), 2)
    return 0, np.nan


def _dataset_master_summary_table(
    items: pd.DataFrame,
    questions_all: pd.DataFrame,
    questions_cohort: pd.DataFrame,
) -> pd.DataFrame:
    """
    各数据集总表：
      - 题库字段：来自题目表全量（如 DISCBench 500）
      - 评估字段：来自 PK cohort（公开 200 + DISCBench 前 200）
    """
    rows: List[Dict] = []
    for src in SOURCE_ORDER:
        q_full = questions_all[questions_all["source"].astype(str) == src]
        q_eval = questions_cohort[questions_cohort["source"].astype(str) == src]
        sub = items[items["source"] == src]
        scored = sub[sub["mean_score"].notna()]
        d_valid = scored.dropna(subset=["disc"])

        cp_full = pd.to_numeric(q_full.get("checkpoint_n"), errors="coerce")
        cp_sum = int(cp_full.sum()) if cp_full.notna().any() else 0
        cn_sum, cn_mean = _constraint_stats(q_full)

        disc_zero_n = int((d_valid["disc"] == 0).sum()) if len(d_valid) else 0
        disc_zero_pct = round(100.0 * disc_zero_n / len(d_valid), 1) if len(d_valid) else np.nan

        rows.append({
            "数据来源": display_source(src),
            "题库题数": len(q_full),
            "评估题数": len(q_eval),
            "考点总数": cp_sum,
            "平均考点数": round(float(cp_full.mean()), 2) if cp_full.notna().any() else np.nan,
            "约束总数": cn_sum,
            "平均约束数": cn_mean,
            "平均指令长度": round(float(q_full["query_len"].mean()), 1) if q_full["query_len"].notna().any() else np.nan,
            "平均难度分": round(float(q_full["difficulty_score"].mean()), 2) if q_full["difficulty_score"].notna().any() else np.nan,
            "平均模型得分": round(float(scored["mean_score"].mean()), 2) if len(scored) else np.nan,
            "平均区分度D": round(float(d_valid["disc"].mean()), 3) if len(d_valid) else np.nan,
            "区分度D_中位数": round(float(d_valid["disc"].median()), 3) if len(d_valid) else np.nan,
            "D=0题数": disc_zero_n,
            "D=0题占比_%": disc_zero_pct,
            "题均分满分率_%": round(float(scored["item_perfect_mean"].mean()) * 100, 1) if len(scored) else np.nan,
            "至少一模型满分率_%": round(float(scored["any_model_perfect"].mean()) * 100, 1) if len(scored) else np.nan,
            "全模型满分率_%": round(float(scored["all_models_perfect"].mean()) * 100, 1) if len(scored) else np.nan,
            "L1覆盖数": int(q_full["L1"].nunique()) if "L1" in q_full.columns else np.nan,
            f"≥{MIN_MODELS}模型可统计题数": len(scored),
            "评估口径": "PK·公开200" if src != OURS_SOURCE else f"PK·DISCBench前{PK_OURS_N}",
            "计分": PRIMARY_SCORE_LABEL,
        })

    df = pd.DataFrame(rows)
    scored_all = items[items["mean_score"].notna()]
    d_all = scored_all.dropna(subset=["disc"])
    disc_zero_all = int((d_all["disc"] == 0).sum()) if len(d_all) else 0
    cp_all = pd.to_numeric(questions_all.get("checkpoint_n"), errors="coerce")
    cn_all_sum, cn_all_mean = _constraint_stats(questions_all)
    total = {
        "数据来源": "PK合计(1000题评估)",
        "题库题数": len(questions_all),
        "评估题数": len(questions_cohort),
        "考点总数": int(cp_all.sum()) if cp_all.notna().any() else 0,
        "平均考点数": round(float(cp_all.mean()), 2) if cp_all.notna().any() else np.nan,
        "约束总数": cn_all_sum,
        "平均约束数": cn_all_mean,
        "平均指令长度": round(float(questions_all["query_len"].mean()), 1),
        "平均难度分": round(float(questions_all["difficulty_score"].mean()), 2),
        "平均模型得分": round(float(scored_all["mean_score"].mean()), 2) if len(scored_all) else np.nan,
        "平均区分度D": round(float(d_all["disc"].mean()), 3) if len(d_all) else np.nan,
        "区分度D_中位数": round(float(d_all["disc"].median()), 3) if len(d_all) else np.nan,
        "D=0题数": disc_zero_all,
        "D=0题占比_%": round(100.0 * disc_zero_all / len(d_all), 1) if len(d_all) else np.nan,
        "题均分满分率_%": round(float(scored_all["item_perfect_mean"].mean()) * 100, 1) if len(scored_all) else np.nan,
        "至少一模型满分率_%": round(float(scored_all["any_model_perfect"].mean()) * 100, 1) if len(scored_all) else np.nan,
        "全模型满分率_%": round(float(scored_all["all_models_perfect"].mean()) * 100, 1) if len(scored_all) else np.nan,
        "L1覆盖数": int(questions_all["L1"].nunique()) if "L1" in questions_all.columns else np.nan,
        f"≥{MIN_MODELS}模型可统计题数": len(scored_all),
        "评估口径": "公开4×200+DISCBench前200",
        "计分": PRIMARY_SCORE_LABEL,
    }
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)


def _source_summary_table(items: pd.DataFrame, questions: pd.DataFrame) -> pd.DataFrame:
    """兼容旧名：等同数据集总表（仅 cohort 题目数）。"""
    return _dataset_master_summary_table(items, questions, questions)


def _query_length_table(questions_all: pd.DataFrame) -> pd.DataFrame:
    """题目表全量（含 Ours 500 题）query 长度。"""
    rows = []
    for src in SOURCE_ORDER + ["全表"]:
        sub = questions_all if src == "全表" else questions_all[questions_all["source"].astype(str) == src]
        ql = sub["query_len"].dropna()
        label = display_source(src) if src != "全表" else "全表"
        rows.append({
            "范围": label,
            "题目数": len(sub),
            "平均长度": round(float(ql.mean()), 1) if len(ql) else np.nan,
            "中位数": round(float(ql.median()), 1) if len(ql) else np.nan,
            "P25": round(float(ql.quantile(0.25)), 1) if len(ql) else np.nan,
            "P75": round(float(ql.quantile(0.75)), 1) if len(ql) else np.nan,
            "最短": int(ql.min()) if len(ql) else np.nan,
            "最长": int(ql.max()) if len(ql) else np.nan,
        })
    return pd.DataFrame(rows)


def _difficulty_mean_corr(items: pd.DataFrame) -> pd.DataFrame:
    rows_corr = []
    valid = items.dropna(subset=["difficulty_score", "mean_score"])
    rows_corr.append({"范围": "全表", **corr_pair(valid["difficulty_score"], valid["mean_score"])})
    for src in SOURCE_ORDER:
        sub = valid[valid["source"] == src]
        if len(sub) >= 3:
            rows_corr.append({"范围": display_source(src), **corr_pair(sub["difficulty_score"], sub["mean_score"])})
    return pd.DataFrame(rows_corr)


def _difficulty_checkpoint_heatmaps(
    questions: pd.DataFrame, out_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_corr = []
    ct_all = None
    for src in SOURCE_ORDER + ["全表"]:
        sub = questions if src == "全表" else questions[questions["source"].astype(str) == src]
        sub = sub.dropna(subset=["difficulty_tier", "checkpoint_bin"])
        if sub.empty:
            continue
        ct = pd.crosstab(sub["checkpoint_bin"], sub["difficulty_tier"], dropna=False)
        for lab in CHECKPOINT_LABELS:
            if lab not in ct.index:
                ct.loc[lab] = 0
        ct = ct.reindex(index=CHECKPOINT_LABELS, columns=TIER_LABELS, fill_value=0)
        label = "全表" if src == "全表" else display_source(src)
        if src == "全表":
            ct_all = ct.copy()
            plot_heatmap(
                ct.astype(float),
                out_dir / "03_difficulty_vs_checkpoint_all.png",
                "All items · checkpoint count bin × difficulty tier · item count",
                mask_empty=False,
            )
        dsub = sub.dropna(subset=["difficulty_score", "checkpoint_n"])
        if len(dsub) >= 3:
            rows_corr.append({"范围": label, **corr_pair(dsub["difficulty_score"], dsub["checkpoint_n"])})
    return pd.DataFrame(rows_corr), ct_all if ct_all is not None else pd.DataFrame()


def _model_source_pivot(replies: pd.DataFrame) -> pd.DataFrame:
    """8 模型 × 5 数据来源 → 均分（PK cohort CANONICAL_8 回复行）。不含 DISCBench 特有 4 模型。"""
    r = replies[replies["logical_model"].isin(CANONICAL_8)].copy()
    rows: List[Dict] = []
    for src in SOURCE_ORDER:
        sub = r[(r["source"] == src)].dropna(subset=["score"])
        dlabel = display_source(src)
        for model in CANONICAL_8:
            g = sub[sub["logical_model"] == model]
            rows.append({
                "模型": model,
                "数据来源": dlabel,
                "均分": round(float(g["score"].mean()), 2) if len(g) else np.nan,
                "有效条数": len(g),
                "覆盖题数": int(g["qid"].nunique()) if len(g) else 0,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    pv = df.pivot_table(index="模型", columns="数据来源", values="均分", aggfunc="first")
    pv = pv.reindex(index=CANONICAL_8, columns=SOURCE_ORDER_DISPLAY)
    return pv


def _model_scores_table(replies: pd.DataFrame) -> pd.DataFrame:
    """长表：模型 × 来源均分。"""
    rows: List[Dict] = []
    for src in SOURCE_ORDER:
        sub = replies[replies["source"] == src].dropna(subset=["score"])
        sub = sub[sub["logical_model"].isin(CANONICAL_8)]
        dlabel = display_source(src)
        for model in CANONICAL_8:
            g = sub[sub["logical_model"] == model]
            rows.append({
                "数据来源": dlabel,
                "模型": model,
                "计分列": PRIMARY_SCORE_LABEL,
                "均分": round(float(g["score"].mean()), 2) if len(g) else np.nan,
                "有效条数": len(g),
            })
    return pd.DataFrame(rows)


def _l1_model_table(replies: pd.DataFrame, questions: pd.DataFrame) -> pd.DataFrame:
    """L1 × 模型均分长表（全 cohort，canonical 8 模型）。"""
    l1_map = questions.set_index("qid")["L1"].to_dict()
    r = replies.copy()
    r["L1"] = r["qid"].map(l1_map)
    r = r.dropna(subset=["score", "L1"])
    r = r[r["logical_model"].isin(CANONICAL_8)]
    rows = []
    for l1 in sorted(r["L1"].unique()):
        for model in CANONICAL_8:
            sub = r[(r["L1"] == l1) & (r["logical_model"] == model)]
            if sub.empty:
                continue
            rows.append({
                "L1": l1,
                "模型": model,
                "均分": round(float(sub["score"].mean()), 2),
                "题数": int(sub["qid"].nunique()),
            })
    return pd.DataFrame(rows)


def _l1_model_pivot(l1_df: pd.DataFrame, *, models: List[str] = None) -> pd.DataFrame:
    """纵轴模型、横轴 L1。"""
    if l1_df.empty:
        return pd.DataFrame()
    idx = models if models is not None else CANONICAL_8
    pv = l1_df.pivot_table(index="模型", columns="L1", values="均分", aggfunc="first")
    pv = pv.reindex(index=idx)
    return pv.sort_index(axis=1)


def _discbench_replies_12(replies: pd.DataFrame, questions_ours: pd.DataFrame) -> pd.DataFrame:
    """DISCBench 全量 500 题 × 12 模型回复（有分即计入）。"""
    ours_qids = set(questions_ours["qid"].astype(str))
    r = replies[
        (replies["source"] == OURS_SOURCE)
        & (replies["qid"].astype(str).isin(ours_qids))
    ].copy()
    r = r[r.apply(reply_row_is_success, axis=1)]
    r["model_12"] = r["model"].map(to_discbench_model)
    r = r[r["model_12"].astype(str).str.len() > 0]
    return r.dropna(subset=["score"])


def _discbench_12_model_scores(replies: pd.DataFrame, questions_all: pd.DataFrame) -> pd.DataFrame:
    """12 模型在 DISCBench 500 题上的均分与排名。"""
    ours_q = questions_all[questions_all["source"].astype(str) == OURS_SOURCE].copy()
    sub = _discbench_replies_12(replies, ours_q)
    n_total = len(ours_q)
    rows: List[Dict] = []
    for model in CANONICAL_12:
        g = sub[sub["model_12"] == model]
        n_q = int(g["qid"].nunique()) if len(g) else 0
        rows.append({
            "模型": model,
            "数据来源": "DISCBench",
            "计分列": PRIMARY_SCORE_LABEL,
            "均分": round(float(g["score"].mean()), 2) if len(g) else np.nan,
            "有效回复数": len(g),
            "覆盖题数": n_q,
            "题库题数": n_total,
            "覆盖率_%": round(100.0 * n_q / n_total, 1) if n_total else np.nan,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("均分", ascending=False, na_position="last").reset_index(drop=True)
    df.insert(0, "排名", range(1, len(df) + 1))
    return df


def _discbench_12_model_pivot(scores_df: pd.DataFrame) -> pd.DataFrame:
    if scores_df.empty:
        return pd.DataFrame()
    s = scores_df.set_index("模型")["均分"].reindex(CANONICAL_12)
    return pd.DataFrame({"DISCBench": s})


def _discbench_l1_model_table(replies: pd.DataFrame, questions_all: pd.DataFrame) -> pd.DataFrame:
    """DISCBench 500 题 · L1 × 12 模型均分。"""
    ours_q = questions_all[questions_all["source"].astype(str) == OURS_SOURCE].copy()
    l1_map = ours_q.set_index("qid")["L1"].to_dict()
    sub = _discbench_replies_12(replies, ours_q)
    sub["L1"] = sub["qid"].map(l1_map)
    sub = sub.dropna(subset=["L1"])
    rows = []
    for l1 in sorted(sub["L1"].unique()):
        for model in CANONICAL_12:
            g = sub[(sub["L1"] == l1) & (sub["model_12"] == model)]
            if g.empty:
                continue
            rows.append({
                "L1": l1,
                "模型": model,
                "均分": round(float(g["score"].mean()), 2),
                "题数": int(g["qid"].nunique()),
            })
    return pd.DataFrame(rows)


def _discbench_l1_pivot(l1_df: pd.DataFrame) -> pd.DataFrame:
    return _l1_model_pivot(l1_df, models=CANONICAL_12)


def _dataset_meta_table(questions: pd.DataFrame) -> pd.DataFrame:
    q = questions.copy()
    q["difficulty_score"] = pd.to_numeric(q["difficulty_score"], errors="coerce")
    q["difficulty_tier"] = assign_tier_bin(q["difficulty_score"])
    rows = []
    for src in SOURCE_ORDER:
        sub = q[q["source"].astype(str) == src]
        tier_counts = {
            lab: int((sub["difficulty_tier"].astype(str) == lab).sum()) for lab in TIER_LABELS
        }
        constraint_sums = {}
        for col, label in zip(CONSTRAINT_COLS, CONSTRAINT_LABELS):
            if col in sub.columns:
                constraint_sums[label] = int(pd.to_numeric(sub[col], errors="coerce").sum())
        rows.append({
            "数据来源": display_source(src),
            "题目数": len(sub),
            **{f"难度_{lab}": v for lab, v in tier_counts.items()},
            "L1覆盖数": int(sub["L1"].nunique()) if "L1" in sub.columns else np.nan,
            "L2覆盖数": int(sub["L2"].nunique()) if "L2" in sub.columns else np.nan,
            "L3覆盖数": int(sub["L3"].nunique()) if "L3" in sub.columns else np.nan,
            **{f"约束_{k}_总数": v for k, v in constraint_sums.items()},
        })
    return pd.DataFrame(rows)


def _constraint_heatmap_data(questions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (各来源约束总数, 各来源六类约束占比 %)。"""
    count_mat = pd.DataFrame(index=SOURCE_ORDER_DISPLAY, columns=CONSTRAINT_LABELS, dtype=float)
    pct_mat = pd.DataFrame(index=SOURCE_ORDER_DISPLAY, columns=CONSTRAINT_LABELS, dtype=float)
    for src in SOURCE_ORDER:
        sub = questions[questions["source"].astype(str) == src]
        dsrc = display_source(src)
        totals = []
        for col, label in zip(CONSTRAINT_COLS, CONSTRAINT_LABELS):
            v = int(pd.to_numeric(sub[col], errors="coerce").sum()) if col in sub.columns else 0
            count_mat.loc[dsrc, label] = v
            totals.append(v)
        denom = sum(totals)
        for label, v in zip(CONSTRAINT_LABELS, totals):
            pct_mat.loc[dsrc, label] = round(100.0 * v / denom, 1) if denom > 0 else 0.0
    return count_mat, pct_mat


def _scope_summary_row(sub: pd.DataFrame, scope: str) -> Dict:
    d = sub.dropna(subset=["difficulty_score", "disc"])
    row: Dict = {
        "范围": scope,
        "题数": len(sub),
        "有D": len(d),
        "难度分_均值": round(float(sub["difficulty_score"].mean()), 2)
        if sub["difficulty_score"].notna().any()
        else np.nan,
        "模型均分_均值": round(float(sub["mean_score"].mean()), 2)
        if sub["mean_score"].notna().any()
        else np.nan,
        "D_均值": round(float(d["disc"].mean()), 3) if len(d) else np.nan,
        "D=0占比_%": round(100.0 * (d["disc"] == 0).mean(), 1) if len(d) else np.nan,
    }
    if len(d) >= 3:
        row["Pearson_难度↔均分"] = round(
            float(d[["difficulty_score", "mean_score"]].corr().iloc[0, 1]), 4
        )
    return row


def run(
    questions_xlsx: Path,
    replies_xlsx: Path,
    output_xlsx: Path,
    *,
    sheet: str = "数据对齐",
) -> None:
    charts_dir = output_xlsx.parent / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    questions_all = load_questions(questions_xlsx, sheet=sheet)
    replies_all = load_replies_with_scores(replies_xlsx, questions_all)
    coverage, questions, replies, items = prepare_analysis_tables(
        questions_xlsx, replies_xlsx, sheet=sheet
    )
    items_scored = items[items["mean_score"].notna()].copy()

    meta = pd.DataFrame([
        {"项": "题目表", "值": str(questions_xlsx)},
        {"项": "回复表", "值": str(replies_xlsx)},
        {"项": "分析cohort", "值": "公开4×200 + DISCBench前200 = 1000题"},
        {"项": "query长度口径", "值": "题目表全量 1300 题"},
        {"项": "统计主分", "值": f"{PRIMARY_SCORE_LABEL} = mean({', '.join(GPT54_SCORE_COLS)})"},
        {"项": "主分裁判", "值": "GPT-5.4 第1轮 + 第3轮"},
        {"项": "题级统计门槛", "值": f"≥{MIN_MODELS} 个模型有评估分"},
        {"项": "预设难度", "值": "difficulty_score（Stage1 指令质量分）"},
        {"项": "难度档", "值": "difficulty_score → D(0-20) C(20-40) B(40-60) A(60-80) S(80-100)"},
        {"项": "区分度D", "值": "高/低27%模型均分差 ÷ 极差（每项按模型分列取分后再算）"},
        {"项": "区分度D_模型集合(PK)", "值": "各题仅用 CANONICAL_8（含 DISCBench），与公开四集及表6题级D口径一致"},
        {"项": "PK回复行", "值": "cohort 内仅保留 CANONICAL_8 行；不参与 PK 的多余模型行不参与任何跨集表/图"},
        {"项": "12模型口径", "值": "仅 DISCBench 全库500题 · 总分排名与表4-1 L1×12；不混入 PK 与其他四集矩阵"},
        {"项": "数据集总表", "值": "01_数据集总表：题库全量字段 + PK cohort 评估字段"},
        {"项": "D=0占比", "值": "题级区分度 D=0（各模型分数完全一致，极差=0）"},
    ])

    summary = _dataset_master_summary_table(items, questions_all, questions)
    query_len = _query_length_table(questions_all)
    corr_diff_mean = _difficulty_mean_corr(items_scored)
    corr_diff_cp, cp_ct_all = _difficulty_checkpoint_heatmaps(questions, charts_dir)
    model_scores = _model_scores_table(replies)
    model_source_pv = _model_source_pivot(replies)
    discbench_12 = _discbench_12_model_scores(replies_all, questions_all)
    discbench_12_pv = _discbench_12_model_pivot(discbench_12)
    discbench_l1 = _discbench_l1_model_table(replies_all, questions_all)
    discbench_l1_pv = _discbench_l1_pivot(discbench_l1)
    l1_model = _l1_model_table(replies, questions)
    l1_pv = _l1_model_pivot(l1_model)
    tier_mean = tier_mean_score_pivot(items_scored)
    tier_disc = tier_mean_disc_pivot(items_scored)
    tier_disc_count = tier_disc_count_pivot(items_scored)
    tier_count = tier_count_pivot_questions(questions_all)
    tier_count_pk = tier_count_pivot(items_scored)
    meta_table = _dataset_meta_table(questions_all)
    constraint_count, constraint_pct = _constraint_heatmap_data(questions)

    q_pk = items_scored
    compare_pk = pd.DataFrame([
        _scope_summary_row(q_pk[q_pk["source"].isin(PUBLIC_SOURCES)], "PK·公开800"),
        _scope_summary_row(q_pk[q_pk["source"] == OURS_SOURCE], f"PK·DISCBench{PK_OURS_N}"),
    ])

    # --- 绘图 ---
    chart_manifest: List[Tuple[str, Path]] = []

    # 1. 难度档 → 各数据集题数 + 题级均分
    p1 = charts_dir / "01_difficulty_tier_mean_score_by_source.png"
    plot_tier_count_and_mean(
        tier_count,
        tier_mean,
        p1,
        title="Item count (full bank) & mean model score (PK cohort) by difficulty tier",
    )
    chart_manifest.append(("1. Difficulty tier × dataset: item count + mean score", p1))

    # 1c. line chart: mean score by difficulty tier
    p1c = charts_dir / "01c_difficulty_tier_mean_score_lines.png"
    plot_tier_mean_score_lines(tier_mean, p1c)
    chart_manifest.append(("1c. Difficulty tier × dataset: mean score (line chart)", p1c))

    # 1b. mean discrimination D by difficulty tier
    if not tier_disc.empty:
        p1b = charts_dir / "01b_difficulty_tier_disc_by_source.png"
        plot_heatmap(
            tier_disc,
            p1b,
            "Mean discrimination D by difficulty tier (PK cohort, ≥4 models scored)",
            fmt=".3f",
            cbar_label="Mean D",
            xlabel="Difficulty tier",
            ylabel="Dataset",
        )
        chart_manifest.append(("1b. Difficulty tier × dataset: mean discrimination D", p1b))

    # 2. fine-grained difficulty vs discrimination
    disc_fine_df = plot_disc_by_difficulty_fine(
        items_scored,
        charts_dir / "02_discrimination_by_difficulty.png",
        bin_width=4.0,
        min_count=5,
    )
    chart_manifest.append(("2. Preset difficulty vs. mean discrimination D (fine-grained)", charts_dir / "02_discrimination_by_difficulty.png"))

    # 3. difficulty × checkpoint (generated in _difficulty_checkpoint_heatmaps)
    chart_manifest.append(("3. Checkpoint count bin × difficulty tier: item count", charts_dir / "03_difficulty_vs_checkpoint_all.png"))

    # 4. query length
    ql_labels = [display_source(s) for s in SOURCE_ORDER]
    ql_vals = [
        float(questions_all.loc[questions_all["source"] == s, "query_len"].mean())
        for s in SOURCE_ORDER
    ]
    p4 = charts_dir / "04_query_length_by_source.png"
    plot_bar_simple(ql_labels, ql_vals, p4, "Mean instruction length by dataset (full question bank)", ylabel="Characters", fmt=".0f")
    chart_manifest.append(("4. Mean instruction length by dataset", p4))

    # 5. DISCBench · 12 models
    if not discbench_12_pv.empty:
        p5 = charts_dir / "05_discbench_12_models_mean.png"
        order = discbench_12.sort_values("均分", ascending=False)["模型"].tolist()
        pv5 = discbench_12_pv.reindex(order)
        plot_heatmap(
            pv5,
            p5,
            "DISCBench · mean score by model (500 items, mean(1_score,3_score))",
            fmt=".1f",
            cbar_label="Mean score",
            xlabel="",
            ylabel="Model",
        )
        chart_manifest.append(("5. DISCBench · 12-model mean score ranking", p5))

    # 6. 8 models × 5 datasets (PK cohort only)
    if not model_source_pv.empty:
        p6 = charts_dir / "06_models_x_sources.png"
        plot_heatmap(
            model_source_pv,
            p6,
            "8 models × 5 datasets · PK cohort mean score (CANONICAL_8, 200/source, mean(1,3))",
            fmt=".1f",
            cbar_label="Mean score",
            xlabel="Dataset",
            ylabel="Model",
        )
        chart_manifest.append(("6. PK · 8 models × 5 datasets: mean score", p6))

    # 7. DISCBench · L1 × 12 models
    if not discbench_l1_pv.empty:
        p7 = charts_dir / "07_discbench_l1_x_12models.png"
        plot_heatmap(
            discbench_l1_pv,
            p7,
            "DISCBench · L1 task type × 12 models · mean score",
            fmt=".1f",
            cbar_label="Mean score",
            xlabel="L1 task type",
            ylabel="Model",
        )
        chart_manifest.append(("7. DISCBench · L1 × 12 models: mean score", p7))

    # 7b. PK cohort · L1 × 8 models
    if not l1_pv.empty:
        p7b = charts_dir / "07b_l1_x_8models_pk.png"
        plot_heatmap(
            l1_pv,
            p7b,
            "PK cohort · L1 task type × 8 models · mean score (1000 items)",
            fmt=".1f",
            cbar_label="Mean score",
            xlabel="L1 task type",
            ylabel="Model",
        )
        chart_manifest.append(("7b. PK cohort · L1 × 8 models: mean score", p7b))

    # legacy 07 filename (same as 7b)
    if not l1_pv.empty:
        plot_heatmap(
            l1_pv,
            charts_dir / "07_l1_x_models.png",
            "PK cohort · L1 task type × 8 models · mean score (1000 items)",
            fmt=".1f",
            cbar_label="Mean score",
            xlabel="L1 task type",
            ylabel="Model",
        )

    # 9. constraint type share
    constraint_pct_en = constraint_pct.rename(
        columns=dict(zip(CONSTRAINT_LABELS, CONSTRAINT_LABELS_EN))
    )
    p9 = charts_dir / "09_source_constraint_pct.png"
    plot_heatmap(
        constraint_pct_en,
        p9,
        "Six constraint types · share (%) by dataset",
        fmt=".1f",
        mask_empty=False,
        cbar_label="Share (%)",
        xlabel="Constraint type",
        ylabel="Dataset",
    )
    chart_manifest.append(("9. Dataset × six constraint types: share (%)", p9))

    # 相关：难度×均分 cross-tab（数据 sheet 用）
    ct_all = cross_tab_count(
        items_scored.dropna(subset=["difficulty_tier", "mean_score_tier"]),
        "difficulty_tier",
        "mean_score_tier",
    )
    if not ct_all.empty:
        ct_plot = ct_all.copy()
        ct_plot.index = [str(i) for i in ct_plot.index]
        ct_plot.columns = [str(c) for c in ct_plot.columns]
        p_pk = charts_dir / "pk_difficulty_vs_mean_score.png"
        plot_heatmap(
            ct_plot.astype(float),
            p_pk,
            "PK cohort · difficulty tier × mean-score tier · item count",
        )
        chart_manifest.append(("2b. PK cohort · difficulty tier × mean-score tier: item count", p_pk))

    coverage_disp = apply_display_source(coverage, "数据来源") if "数据来源" in coverage.columns else coverage
    items_out = apply_display_source(items.sort_values(["source", "difficulty_score"], ascending=[True, False]))

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as w:
        write_stacked_tables(w, "00_说明与覆盖", [
            ("【说明】统计口径与配置", meta),
            ("【数据覆盖】各来源题目与回复覆盖", coverage_disp),
        ])
        write_stacked_tables(w, "表1_数据集概览", [
            ("表1-1  各数据集总览（题库全量 + PK cohort 评估）", summary),
            ("表1-2  各数据集指令长度分布", query_len),
            ("表1-3  各数据集元信息（任务/难度/约束覆盖）", meta_table),
            ("表1-4  六类约束总数", constraint_count),
            ("表1-5  六类约束占比（%）", constraint_pct),
        ])
        sec2: List[Tuple[str, pd.DataFrame]] = [
            ("表2-0  难度档 × 数据集 → 题数（题目表全量 · difficulty_score）", tier_count),
            ("表2-0b 难度档 × 数据集 → 题数（PK cohort 评估子集）", tier_count_pk),
            ("表2-1  难度档 × 数据集 → 题级模型均分（PK cohort）", tier_mean),
            ("表2-1b 难度档 × 数据集 → 平均区分度 D（PK cohort）", tier_disc),
            ("表2-1c 难度档 × 数据集 → 有 D 的题数（PK cohort）", tier_disc_count),
            ("表2-2  难度档 × 模型均分档 → 题数（PK cohort）", ct_all),
            ("表2-3  难度分与模型均分相关（按来源）", corr_diff_mean),
        ]
        if not disc_fine_df.empty:
            sec2.append(("表2-4  细粒度难度分 × 题均区分度 D", disc_fine_df))
        if not cp_ct_all.empty:
            sec2.append(("表2-5  考点数分箱 × 难度档 → 题数", cp_ct_all))
        sec2.extend([
            ("表2-6  难度分与考点数相关（按来源）", corr_diff_cp),
            ("表2-7  PK 公开集 vs DISCBench 对比", compare_pk),
        ])
        write_stacked_tables(w, "表2_难度与区分度", sec2)
        sec3: List[Tuple[str, pd.DataFrame]] = [
            ("表3-1  8 模型 × 5 数据集均分矩阵（PK cohort×200/source，CANONICAL_8）", model_source_pv),
            ("表3-2  8 模型 × 来源均分明细（长表）", model_scores),
        ]
        if not discbench_12.empty:
            sec3.append(("表3-3  DISCBench · 12 模型均分与排名（500 题）", discbench_12))
        if not discbench_12_pv.empty:
            sec3.append(("表3-4  DISCBench · 12 模型均分（透视）", discbench_12_pv))
        write_stacked_tables(w, "表3_模型得分", sec3)
        sec4: List[Tuple[str, pd.DataFrame]] = []
        if not discbench_l1_pv.empty:
            sec4.append(("表4-1  DISCBench · L1 × 12 模型均分", discbench_l1_pv))
        if not l1_pv.empty:
            sec4.append(("表4-2  PK cohort · L1 × 8 模型均分", l1_pv))
        if sec4:
            write_stacked_tables(w, "表4_L1任务类型", sec4)
        write_stacked_tables(w, "附录_题级明细", [
            ("附录 A  PK cohort 题级统计明细（1000 题）", items_out),
        ])

    format_paper_excel_sheets(output_xlsx)
    embed_images_in_excel(output_xlsx, chart_manifest, sheet_name="Charts")

    print(f"✓ Excel: {output_xlsx}")
    print(f"✓ 统计图表 sheet 嵌入 {len(chart_manifest)} 张图")
    print(f"✓ PNG → {charts_dir}/")
    print(f"  分析cohort: {len(questions)} 题 | 可统计题级均分: {len(items_scored)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="DISCBench Compared benchmark 综合统计")
    ap.add_argument("--questions", default=str(resolve_questions_path()))
    ap.add_argument("--questions-sheet", default=QUESTIONS_SHEET)
    ap.add_argument("--replies", default=str(resolve_replies_path()))
    ap.add_argument("--output", default=str(_ROOT / "output/reports/comprehensive_benchmark_stats.xlsx"))
    args = ap.parse_args()
    run(
        Path(args.questions),
        Path(args.replies),
        Path(args.output),
        sheet=args.questions_sheet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
