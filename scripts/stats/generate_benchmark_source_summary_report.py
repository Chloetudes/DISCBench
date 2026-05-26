#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按评测集(source)汇总：8 模型均分/排名、三轮裁判一致性、题级区分度、难度等。
输出 Excel：`output/reports/benchmark_source_model_summary.xlsx`
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.cif_stats_common import (
    CANONICAL_8,
    GPT54_JUDGE_MODEL,
    GPT54_MEAN_SCORE_COL,
    GPT54_SCORE_COLS,
    MIN_MODELS,
    OURS_SOURCE,
    PK_OURS_N,
    PRIMARY_SCORE_LABEL,
    PUBLIC_ANALYSIS_N,
    PUBLIC_SOURCES,
    resolve_questions_path,
    resolve_replies_path,
    SOURCE_ORDER,
    analysis_cohort_qids,
    gpt54_mean_score,
    load_questions,
    load_replies_with_scores,
    normalize_qid,
    primary_score_col,
    reply_row_is_success,
    safe_str,
    to_canonical,
    verify_cohort_completeness,
    format_paper_excel_sheets,
    write_stacked_tables,
)

SOURCE_ORDER = list(PUBLIC_SOURCES) + [OURS_SOURCE, "公开四集合计", "PK_cohort"]
REAL_SOURCES = list(PUBLIC_SOURCES) + [OURS_SOURCE]

SCORE_COLS = {
    "第1轮_GPT54": "1_score",
    "第2轮_Claude46": "2_score",
    "第3轮_GPT54": "3_score",
    "第4轮_GPT54": "4_score",
}

PRIMARY_SCORE_LABEL = "主分析_GPT54"

JUDGE_BY_COL = {
    "eval_compared_infobench": GPT54_JUDGE_MODEL,
    "1_score": GPT54_JUDGE_MODEL,
    "2_score": "Claude Opus 4.6",
    "3_score": GPT54_JUDGE_MODEL,
    "4_score": GPT54_JUDGE_MODEL,
}


def _primary_col(source: str) -> str:
    return GPT54_MEAN_SCORE_COL


def discrimination_index(scores: pd.Series, min_n: int = 4) -> float:
    s = pd.to_numeric(scores, errors="coerce").dropna()
    if len(s) < min_n:
        return np.nan
    sorted_scores = s.sort_values()
    n_27 = max(1, int(len(s) * 0.27))
    hg = float(sorted_scores.tail(n_27).mean())
    lg = float(sorted_scores.head(n_27).mean())
    rng = float(s.max() - s.min())
    return (hg - lg) / rng if rng > 0 else 0.0


def pair_metrics(df: pd.DataFrame, a: str, b: str) -> Dict[str, Any]:
    sub = df.dropna(subset=[a, b])
    if len(sub) < 5:
        return {"n_pairs": len(sub)}
    s1, s2 = sub[a].values, sub[b].values
    d = s2 - s1
    pr = scipy_stats.pearsonr(s1, s2)
    return {
        "n_pairs": int(len(sub)),
        "Pearson_r": round(float(pr.statistic), 4),
        "MAE": round(float(np.abs(d).mean()), 2),
        "同分率_%": round(float((s1 == s2).mean() * 100), 1),
        "均分差_B减A": round(float(d.mean()), 2),
        "judge_A": JUDGE_BY_COL.get(a, a),
        "judge_B": JUDGE_BY_COL.get(b, b),
    }


def load_data(replies_path: Path, questions_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, Optional[str], pd.DataFrame]:
    qs = load_questions(questions_path, sheet="数据对齐")
    cohort_qids = analysis_cohort_qids(qs)
    scored_replies = load_replies_with_scores(replies_path, qs)
    coverage = verify_cohort_completeness(qs, scored_replies)

    qs = qs[qs["qid"].astype(str).isin(cohort_qids)].copy()
    from lib.jsonl_store import load_replies_df

    replies = load_replies_df(replies_path, canonical_only=False)
    replies["qid"] = replies["qid"].astype(str).str.strip().map(normalize_qid)
    qid_src = dict(zip(qs["qid"].astype(str), qs["source"].astype(str)))
    replies["source"] = replies["qid"].map(qid_src)
    replies["logical_model"] = replies["model"].astype(str).map(to_canonical)
    replies = replies[replies["logical_model"].astype(str).str.len() > 0]
    replies = replies[replies.apply(reply_row_is_success, axis=1)]
    for col in SCORE_COLS.values():
        if col in replies.columns:
            replies[col] = pd.to_numeric(replies[col], errors="coerce")
    replies[GPT54_MEAN_SCORE_COL] = replies.apply(gpt54_mean_score, axis=1)
    replies = replies[replies["qid"].astype(str).isin(cohort_qids)].copy()
    # 跨数据集对比与综合统计一致：只保留 8 个可比模型行，不混入 DISCBench 额外 4 模型
    replies = replies[replies["logical_model"].isin(CANONICAL_8)].copy()

    diff_col = "difficulty_score" if "difficulty_score" in qs.columns else None
    return replies, qs, diff_col, coverage


def source_mask(replies: pd.DataFrame, source: str) -> pd.Series:
    if source == "PK_cohort":
        return pd.Series(True, index=replies.index)
    if source == "公开四集合计":
        return replies["source"].isin(PUBLIC_SOURCES)
    return replies["source"] == source


def build_model_scores_table(replies: pd.DataFrame, source: str) -> pd.DataFrame:
    sub = replies[source_mask(replies, source)].copy()
    rows: List[Dict[str, Any]] = []
    rounds = dict(SCORE_COLS)
    if source in REAL_SOURCES:
        rounds[PRIMARY_SCORE_LABEL] = _primary_col(source)
    for score_label, col in rounds.items():
        if col not in sub.columns:
            continue
        g = sub.dropna(subset=[col]).groupby("logical_model")[col]
        for model in CANONICAL_8:
            if model not in g.groups:
                rows.append({
                    "评测集": source,
                    "分数轮次": score_label,
                    "裁判": JUDGE_BY_COL.get(col, ""),
                    "模型": model,
                    "均分": np.nan,
                    "有效条数": 0,
                })
                continue
            s = g.get_group(model)
            rows.append({
                "评测集": source,
                "分数轮次": score_label,
                "裁判": JUDGE_BY_COL.get(col, ""),
                "模型": model,
                "均分": round(float(s.mean()), 2),
                "有效条数": int(len(s)),
                "标准差": round(float(s.std(ddof=1)), 2) if len(s) > 1 else 0,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # 排名：同评测集+分数轮次内按均分降序
    df["排名"] = df.groupby(["评测集", "分数轮次"])["均分"].rank(
        ascending=False, method="min"
    )
    return df


def build_composite_rank(replies: pd.DataFrame, source: str, score_col: str) -> pd.DataFrame:
    """每题对 8 模型排名后取平均名次（仅含 8 模型均有分的题）。"""
    sub = replies[source_mask(replies, source)].dropna(subset=[score_col])
    rank_rows = []
    for qid, g in sub.groupby("qid"):
        pivot = g.groupby("logical_model")[score_col].first()
        if not all(m in pivot.index for m in CANONICAL_8):
            continue
        scores = pivot.reindex(CANONICAL_8)
        ranks = scores.rank(ascending=False, method="min")
        for model in CANONICAL_8:
            rank_rows.append({"qid": qid, "模型": model, "题内排名": ranks[model]})
    if not rank_rows:
        return pd.DataFrame()
    rdf = pd.DataFrame(rank_rows)
    agg = rdf.groupby("模型")["题内排名"].agg(["mean", "count"]).reset_index()
    agg.columns = ["模型", "平均题内排名", "计入题数"]
    agg = agg.sort_values("平均题内排名")
    agg["综合排名"] = range(1, len(agg) + 1)
    agg.insert(0, "评测集", source)
    agg.insert(1, "分数列", score_col)
    return agg


def build_source_discrimination(
    replies: pd.DataFrame, qs: pd.DataFrame, diff_col: Optional[str], source: str, score_col: str
) -> Dict[str, Any]:
    sub = replies[source_mask(replies, source)].dropna(subset=[score_col])
    qmeta = qs.drop_duplicates("qid").set_index("qid") if "qid" in qs.columns else pd.DataFrame()
    item_rows = []
    for qid, g in sub.groupby("qid"):
        sc = g.groupby("logical_model")[score_col].first()
        sc8 = sc.reindex([m for m in CANONICAL_8 if m in sc.index]).dropna()
        if len(sc8) < 4:
            continue
        mean_v = float(sc8.mean())
        item_rows.append({
            "qid": qid,
            "n_models": len(sc8),
            "题均分": mean_v,
            "spread": float(sc8.max() - sc8.min()),
            "mad": float((sc8 - mean_v).abs().mean()),
            "D": discrimination_index(sc8),
            "item_difficulty": 100 - mean_v,
        })
    idf = pd.DataFrame(item_rows)
    if idf.empty:
        return {"评测集": source, "分数列": score_col, "有效题数": 0}
    out: Dict[str, Any] = {
        "评测集": source,
        "分数列": score_col,
        "有效题数": len(idf),
        "题均分_均值": round(idf["题均分"].mean(), 2),
        "题均分_中位数": round(idf["题均分"].median(), 2),
        "spread_均值": round(idf["spread"].mean(), 2),
        "spread_中位数": round(idf["spread"].median(), 2),
        "MAD_均值": round(idf["mad"].mean(), 2),
        "区分度D_均值": round(idf["D"].mean(), 3),
        "区分度D_中位数": round(idf["D"].median(), 3),
        "spread≥40_题占比_%": round((idf["spread"] >= 40).mean() * 100, 1),
        "spread≥50_题占比_%": round((idf["spread"] >= 50).mean() * 100, 1),
        "item_difficulty_均值": round(idf["item_difficulty"].mean(), 2),
    }
    if diff_col and diff_col in qmeta.columns and source in (OURS_SOURCE, "PK_cohort", "公开四集合计"):
        m = idf.copy()
        m["difficulty_score"] = m["qid"].map(
            lambda q: pd.to_numeric(qmeta.loc[q, diff_col], errors="coerce")
            if q in qmeta.index
            else np.nan
        )
        d = m.dropna(subset=["difficulty_score"])
        if len(d) >= 10:
            r_sp, _ = scipy_stats.pearsonr(d["difficulty_score"], d["spread"])
            r_mn, _ = scipy_stats.pearsonr(d["difficulty_score"], d["题均分"])
            out["难度分_vs_spread_r"] = round(float(r_sp), 3)
            out["难度分_vs_题均分_r"] = round(float(r_mn), 3)
            out["难度相关_n题"] = len(d)
    return out


def build_judge_consistency(replies: pd.DataFrame, source: str) -> pd.DataFrame:
    sub = replies[source_mask(replies, source)]
    col1 = SCORE_COLS["第1轮_GPT54"]
    col2 = SCORE_COLS["第2轮_Claude46"]
    col3 = SCORE_COLS["第3轮_GPT54"]
    col4 = SCORE_COLS.get("第4轮_GPT54", "4_score")
    pairs = [
        ("第1轮 vs 第2轮", col1, col2),
        ("第2轮 vs 第3轮（当前）", col2, col3),
        ("第1轮 vs 第3轮（主信度）", col1, col3),
    ]
    if source == OURS_SOURCE and col4 in sub.columns and sub[col4].notna().sum() >= 5:
        pairs.append(("第3轮 vs 第4轮", col3, col4))
    rows = []
    for label, col_a, col_b in pairs:
        if col_a not in sub.columns or col_b not in sub.columns:
            continue
        m = pair_metrics(sub, col_a, col_b)
        m["评测集"] = source
        m["对比"] = label
        rows.append(m)
    return pd.DataFrame(rows)


PERFECT_SCORE = 100.0


def build_perfect_rate_summary(replies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按评测集汇总满分率；主分析列 = mean(1_score, 3_score)。"""
    source_rows: List[Dict[str, Any]] = []
    model_rows: List[Dict[str, Any]] = []

    for src in SOURCE_ORDER:
        sub = replies[source_mask(replies, src)]
        primary_col = GPT54_MEAN_SCORE_COL
        rounds = dict(SCORE_COLS)
        if src in REAL_SOURCES:
            rounds[PRIMARY_SCORE_LABEL] = primary_col

        for score_label, col in rounds.items():
            if col not in sub.columns:
                continue
            scored = sub.dropna(subset=[col])
            n = len(scored)
            n_perfect = int((scored[col] == PERFECT_SCORE).sum()) if n else 0
            source_rows.append({
                "评测集": src,
                "分数轮次": score_label,
                "分列": col,
                "裁判": JUDGE_BY_COL.get(col, ""),
                "有效评分数": n,
                "满分条数": n_perfect,
                "满分率_%": round(n_perfect / n * 100, 2) if n else np.nan,
            })
            if score_label == PRIMARY_SCORE_LABEL and n:
                q_any = q_all8 = 0
                for _, g in scored.groupby("qid"):
                    sc = g.groupby("logical_model")[col].first()
                    if (sc == PERFECT_SCORE).any():
                        q_any += 1
                    if set(CANONICAL_8).issubset(sc.index) and (sc.reindex(CANONICAL_8) == PERFECT_SCORE).all():
                        q_all8 += 1
                n_q = scored["qid"].nunique()
                source_rows[-1]["有满分题数_任一模型"] = q_any
                source_rows[-1]["有满分题占比_%"] = round(q_any / n_q * 100, 2) if n_q else np.nan
                n_full_q = sum(
                    1
                    for _, g in scored.groupby("qid")
                    if set(CANONICAL_8).issubset(set(g["logical_model"].unique()))
                )
                source_rows[-1]["8模型齐全题数"] = n_full_q
                source_rows[-1]["8模型全满分题数"] = q_all8
                source_rows[-1]["8模型全满分题占比_%"] = (
                    round(q_all8 / n_full_q * 100, 2) if n_full_q else np.nan
                )

        if primary_col not in sub.columns or src not in REAL_SOURCES:
            continue
        scored = sub.dropna(subset=[primary_col])
        for model in CANONICAL_8:
            msub = scored[scored["logical_model"] == model]
            n = len(msub)
            n_perfect = int((msub[primary_col] == PERFECT_SCORE).sum()) if n else 0
            model_rows.append({
                "评测集": src,
                "分数轮次": PRIMARY_SCORE_LABEL,
                "分列": primary_col,
                "模型": model,
                "有效评分数": n,
                "满分条数": n_perfect,
                "满分率_%": round(n_perfect / n * 100, 2) if n else np.nan,
            })

    return pd.DataFrame(source_rows), pd.DataFrame(model_rows)


def build_coverage(replies: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src in SOURCE_ORDER:
        sub = replies[source_mask(replies, src)]
        row = {"评测集": src, "回复行数": len(sub)}
        for label, col in {**SCORE_COLS, PRIMARY_SCORE_LABEL: _primary_col(src)}.items():
            if col not in sub.columns:
                continue
            row[f"{label}_有效行"] = int(sub[col].notna().sum())
            row[f"{label}_覆盖率_%"] = round(sub[col].notna().mean() * 100, 1) if len(sub) else 0
        primary = GPT54_MEAN_SCORE_COL
        if primary in sub.columns:
            n_full = 0
            for _, g in sub.dropna(subset=[primary]).groupby("qid"):
                if set(CANONICAL_8).issubset(set(g["logical_model"].unique())):
                    n_full += 1
            row["主分析_8模型齐全题数"] = n_full
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replies", default=str(resolve_replies_path()))
    ap.add_argument("--questions", default=str(resolve_questions_path()))
    ap.add_argument(
        "--output",
        default=str(_ROOT / "output/reports/benchmark_source_model_summary.xlsx"),
    )
    args = ap.parse_args()

    replies, qs, diff_col, coverage = load_data(Path(args.replies), Path(args.questions))

    model_scores_parts = []
    composite_parts = []
    disc_parts = []
    judge_parts = []

    for src in SOURCE_ORDER:
        model_scores_parts.append(build_model_scores_table(replies, src))
        pcol = _primary_col(src)
        if pcol in replies.columns and src in REAL_SOURCES:
            composite_parts.append(build_composite_rank(replies, src, pcol))
        for label, col in SCORE_COLS.items():
            if col in replies.columns:
                disc_parts.append(build_source_discrimination(replies, qs, diff_col, src, col))
        if src in REAL_SOURCES and pcol in replies.columns:
            disc_parts.append(build_source_discrimination(replies, qs, diff_col, src, pcol))
        judge_parts.append(build_judge_consistency(replies, src))

    model_scores = pd.concat(model_scores_parts, ignore_index=True)
    composite = pd.concat(composite_parts, ignore_index=True) if composite_parts else pd.DataFrame()
    discrimination = pd.DataFrame(disc_parts)
    judge = pd.concat(judge_parts, ignore_index=True)
    coverage_detail = build_coverage(replies)
    perfect_by_source, perfect_by_model = build_perfect_rate_summary(replies)

    meta = pd.DataFrame([
        {"项": "分析cohort", "值": f"公开4×{PUBLIC_ANALYSIS_N} + Ours前{PK_OURS_N} = 1000题"},
        {"项": "回复行过滤", "值": "仅 CANONICAL_8；12 模型 DISCBench 总榜见 comprehensive 表3-3"},
        {"项": "统一裁判", "值": GPT54_JUDGE_MODEL},
        {"项": "统计主分", "值": f"{PRIMARY_SCORE_LABEL} = mean({', '.join(GPT54_SCORE_COLS)})"},
        {"项": "2_score", "值": "Claude 裁判，仅作一致性对比"},
    ])

    primary_only = model_scores[model_scores["分数轮次"] == PRIMARY_SCORE_LABEL].copy()
    pivot_mean = primary_only.pivot_table(
        index="评测集", columns="模型", values="均分", aggfunc="first"
    )
    pivot_rank = primary_only.pivot_table(
        index="评测集", columns="模型", values="排名", aggfunc="first"
    )
    pivot_mean = pivot_mean.reindex(columns=CANONICAL_8)
    pivot_rank = pivot_rank.reindex(columns=CANONICAL_8)

    pivot_perfect = pd.DataFrame()
    pm = perfect_by_model[perfect_by_model["分数轮次"] == PRIMARY_SCORE_LABEL].copy()
    if not pm.empty:
        pivot_perfect = pm.pivot_table(
            index="评测集", columns="模型", values="满分率_%", aggfunc="first"
        ).reindex(columns=CANONICAL_8)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sec5: List[Tuple[str, pd.DataFrame]] = [
        ("表5-1  各模型均分与排名（分来源 × 轮次）", model_scores),
        ("表5-2  主分析均分矩阵（8 模型 × 5 来源）", pivot_mean),
        ("表5-3  主分析排名矩阵", pivot_rank),
    ]
    if not composite.empty:
        sec5.append(("表5-4  综合题内排名", composite))
    sec5.extend([
        ("表5-5  满分率（按数据来源）", perfect_by_source),
        ("表5-6  满分率（按模型）", perfect_by_model),
    ])
    if not pivot_perfect.empty:
        sec5.append(("表5-7  主分析满分率矩阵（%）", pivot_perfect))

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        write_stacked_tables(writer, "00_说明与覆盖", [
            ("【说明】统计口径与配置", meta),
            ("【数据覆盖】各来源题目覆盖", coverage),
            ("【回复覆盖明细】各模型回复成功数", coverage_detail),
        ])
        write_stacked_tables(writer, "表5_模型排名与满分率", sec5)
        write_stacked_tables(writer, "表6_裁判与区分度", [
            ("表6-1  区分度与难度（分来源 × 计分列）", discrimination),
            ("表6-2  裁判一致性（GPT-5.4 vs Claude）", judge),
        ])

    format_paper_excel_sheets(out_path)

    print(f"✓ 已写入: {out_path}")
    print(f"  - 01_模型均分与排名: {len(model_scores)} 行")
    print(f"  - 03_区分度与难度: {len(discrimination)} 行")
    print(f"  - 04_裁判一致性: {len(judge)} 行")
    print(f"  - 07_满分率_按来源: {len(perfect_by_source)} 行")


if __name__ == "__main__":
    main()
