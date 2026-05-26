#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CIF compared benchmark 统计共用常量与工具。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_CIF_ROOT = Path(__file__).resolve().parents[2]
from lib.paths import (  # noqa: E402
    CIF_ROOT,
    QUESTIONS_JSONL,
    QUESTIONS_SHEET,
    REPLIES_JSONL,
    resolve_questions_path,
    resolve_replies_path,
)

PUBLIC_SOURCES = ("CFbench", "infobench", "ComplexBench", "advancedif")
OURS_SOURCE = "Ours"
MIN_MODELS = 4
PK_OURS_N = 200
PUBLIC_ANALYSIS_N = 200  # 每个公开集固定 200 题
PERFECT_SCORE = 100.0

# 统一裁判：GPT-5.4；统计主分 = mean(1_score, 3_score)（两轮 GPT-5.4 均分）
GPT54_JUDGE_MODEL = "gpt-5.4-2026-03-05"
GPT54_SCORE_COLS = ("1_score", "3_score")
GPT54_MEAN_SCORE_COL = "gpt54_mean_score"
PRIMARY_SCORE_LABEL = "mean(1_score,3_score)"
# 原始分列（仅作对照 / 裁判一致性，不作主统计）
PUBLIC_SCORE_COL = "1_score"
OURS_SCORE_COL = "4_score"

CANONICAL_8 = [
    "claude-sonnet-4-6",
    "deepseek-v3.2-thinking",
    "doubao-seed-2-0-pro",
    "gemini-3.1-pro-preview",
    "glm-5.1",
    "gpt-5.4-2026-03-05",
    "kimi-k2.5",
    "qwen3.6-plus",
]

# DISCBench 额外 4 模型（仅自建集 500 题评估）
DISCBENCH_EXTRA_4 = [
    "GLM-4.7-flash",
    "qwen3.5-27b",
    "step-3.5-flash",
    "Hunyuan-T1-20250822",
]
CANONICAL_12 = CANONICAL_8 + DISCBENCH_EXTRA_4

LEGACY_TO_CANONICAL = {
    "cladue-opus-4-6-thinking": "claude-sonnet-4-6",
    "doubao-seed-2.0-pro": "doubao-seed-2-0-pro",
    "gemini-3-pro-preview": "gemini-3.1-pro-preview",
    "glm-5": "glm-5.1",
    "DeepSeek-V3.2-Thinking": "deepseek-v3.2-thinking",
}

# 难度档 D/C/B/A/S ↔ [0,20) [20,40) [40,60) [60,80) [80,100]
TIER_BINS = [0, 20, 40, 60, 80, 100.01]
TIER_LABELS = ["D(0-20)", "C(20-40)", "B(40-60)", "A(60-80)", "S(80-100)"]
TIER_SHORT = TIER_LABELS  # 图表/表头统一「档(分数区间)」
# 折线图 x 轴：与 assign_tier_bin(right=False) 一致的左闭右开区间
TIER_INTERVAL_LABELS = [
    "D\n[0,20)",
    "C\n[20,40)",
    "B\n[40,60)",
    "A\n[60,80)",
    "S\n[80,100)",
]
TIER_SCORE_TICK_LABELS = ["0-20", "20-40", "40-60", "60-80", "80-100"]

# 题目表 difficulty_score（Stage1 指令质量分）
DIFFICULTY_PRIORITY = [
    "difficulty_score",
    "difficulty_score_py",
    "difficulty_score_avg",
    "提取难度分",
    "difficulty1",
]
DIFF_BINS = TIER_BINS
DIFF_LABELS = TIER_LABELS
MEAN_SCORE_BINS = TIER_BINS
MEAN_SCORE_LABELS = TIER_LABELS

CHECKPOINT_BINS = [-0.01, 4, 8, 12, 16, 10000]
CHECKPOINT_LABELS = ["≤4", "5-8", "9-12", "13-16", "17+"]

D_BINS = [0, 0.3, 0.5, 0.7, 0.85, 1.01]
D_LABELS = ["[0,0.3)", "[0.3,0.5)", "[0.5,0.7)", "[0.7,0.85)", "[0.85,1.0]"]
SOURCE_ORDER = list(PUBLIC_SOURCES) + [OURS_SOURCE]

# 展示名：自建集统一为 DISCBench，排在 SOURCE_ORDER 末尾
SOURCE_DISPLAY: Dict[str, str] = {
    "CFbench": "CFbench",
    "infobench": "infobench",
    "ComplexBench": "ComplexBench",
    "advancedif": "advancedif",
    "Ours": "DISCBench",
}
SOURCE_ORDER_DISPLAY = [SOURCE_DISPLAY[s] for s in SOURCE_ORDER]

CONSTRAINT_COLS = [
    "iq_count_教学约束",
    "iq_count_素材约束",
    "iq_count_流程步骤",
    "iq_count_格式输出",
    "iq_count_边界范围",
    "iq_count_数量篇幅",
]
CONSTRAINT_LABELS = ["教学", "素材", "流程", "格式", "边界", "数量篇幅"]
CONSTRAINT_LABELS_EN = [
    "Teaching",
    "Material",
    "Process",
    "Format",
    "Boundary",
    "Length/Count",
]

BENCHMARK_EIGHT_FAMILY_ORDER = (
    "glm", "gpt", "claude", "gemini", "doubao", "qwen", "deepseek", "kimi",
)
BENCHMARK_EIGHT_FAMILY_LABEL_ZH = {
    "glm": "智谱 GLM",
    "gpt": "OpenAI GPT",
    "claude": "Anthropic Claude",
    "gemini": "Google Gemini",
    "doubao": "字节豆包",
    "qwen": "阿里通义 Qwen",
    "deepseek": "DeepSeek",
    "kimi": "月之暗面 Kimi",
}


def safe_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        try:
            import math
            if math.isnan(value):
                return ""
        except Exception:
            pass
    return str(value)


def normalize_qid(val) -> str:
    s = str(val).strip()
    try:
        f = float(s)
        if not pd.isna(f) and f == int(f):
            return str(int(f))
    except (ValueError, TypeError):
        pass
    return s


def reply_row_is_success(row_like) -> bool:
    status = safe_str(row_like.get("status", "")).strip().lower()
    if status:
        return status == "ok"
    reply = safe_str(row_like.get("reply", "")).strip()
    return bool(reply and not reply.startswith("<error"))


def resolve_eval_column(replies_df: pd.DataFrame, eval_batch_id: str = None) -> str:
    eval_cols = [
        c for c in replies_df.columns
        if isinstance(c, str) and c.startswith("eval_") and not c.endswith("_raw")
    ]
    if eval_batch_id:
        candidate = f"eval_{eval_batch_id}"
        if candidate in replies_df.columns:
            return candidate
        alt = eval_batch_id.replace("_", "")
        alt2 = eval_batch_id.replace("batch", "batch_") if "batch" in eval_batch_id else eval_batch_id
        for c in [f"eval_{alt}", f"eval_{alt2}"]:
            if c in replies_df.columns:
                return c
    if eval_cols:
        return eval_cols[-1]
    raise ValueError(f"结果表中未找到评估列（eval_*），已有列: {list(replies_df.columns)}")


def to_canonical(model: str) -> str:
    m = str(model or "").strip()
    if m in CANONICAL_8:
        return m
    if m in LEGACY_TO_CANONICAL:
        return LEGACY_TO_CANONICAL[m]
    low = {k.lower(): v for k, v in LEGACY_TO_CANONICAL.items()}
    return low.get(m.lower(), "")


def to_discbench_model(model: str) -> str:
    """DISCBench 12 模型 id（含额外 4 个仅 Ours 评估的模型）。"""
    c = to_canonical(model)
    if c:
        return c
    m = str(model or "").strip()
    for x in DISCBENCH_EXTRA_4:
        if m.lower() == x.lower():
            return x
    return ""


def score_col_for_source(source: str) -> str:
    """统计用分列标签（全体系统一 GPT-5.4 双轮均分）。"""
    return PRIMARY_SCORE_LABEL


def score_rule_label(source: str) -> str:
    label = SOURCE_DISPLAY.get(str(source), str(source))
    return f"{label}·{PRIMARY_SCORE_LABEL}·GPT54"


def primary_score_col(source: str) -> str:
    """统计主分：mean(1_score, 3_score)，均为 GPT-5.4。"""
    return PRIMARY_SCORE_LABEL


def gpt54_mean_score(row: pd.Series) -> float:
    """GPT-5.4 第1轮与第3轮分数的算术平均；任一轮缺失则 NaN。"""
    vals = []
    for col in GPT54_SCORE_COLS:
        if col not in row.index:
            return np.nan
        v = pd.to_numeric(row.get(col), errors="coerce")
        if pd.isna(v):
            return np.nan
        vals.append(float(v))
    return round(float(np.mean(vals)), 4) if vals else np.nan


def discrimination_index(scores: pd.Series, min_n: int = MIN_MODELS) -> float:
    s = pd.to_numeric(scores, errors="coerce").dropna()
    if len(s) < min_n:
        return np.nan
    sorted_scores = s.sort_values()
    n_27 = max(1, int(len(s) * 0.27))
    hg = float(sorted_scores.tail(n_27).mean())
    lg = float(sorted_scores.head(n_27).mean())
    rng = float(s.max() - s.min())
    return (hg - lg) / rng if rng > 0 else 0.0


def qdisc_from_scores(scores: pd.Series, min_n: int = MIN_MODELS) -> Dict[str, float]:
    s = pd.to_numeric(scores, errors="coerce").dropna()
    if len(s) < min_n:
        return {}
    return {
        "disc": discrimination_index(s, min_n=min_n),
        "spread": float(s.max() - s.min()),
        "n_models": float(len(s)),
        "mean_score": float(s.mean()),
    }


def pick_difficulty(row: pd.Series) -> Tuple[float, str]:
    for c in DIFFICULTY_PRIORITY:
        if c not in row.index:
            continue
        v = pd.to_numeric(row.get(c), errors="coerce")
        if pd.notna(v):
            return float(v), c
    return np.nan, ""


def query_length(text) -> float:
    s = safe_str(text)
    return float(len(s)) if s else np.nan


def count_checkpoints_from_rubrics(text) -> float:
    s = safe_str(text)
    if not s:
        return np.nan
    numbered = re.findall(r"(?m)^\s*\d+\.\s", s)
    if numbered:
        return float(len(numbered))
    bullets = [ln for ln in s.splitlines() if ln.strip()]
    return float(len(bullets)) if bullets else np.nan


def checkpoint_count_for_row(row: pd.Series) -> float:
    if "cp_classify_n_items" in row.index:
        v = pd.to_numeric(row.get("cp_classify_n_items"), errors="coerce")
        if pd.notna(v) and v > 0:
            return float(v)
    for col in ("rubrics", "evaluation_criteria", "human_rubrics"):
        if col in row.index:
            n = count_checkpoints_from_rubrics(row.get(col))
            if pd.notna(n) and n > 0:
                return n
    if "iq_denominator" in row.index:
        v = pd.to_numeric(row.get("iq_denominator"), errors="coerce")
        if pd.notna(v) and v > 0:
            return float(v)
    return np.nan


def assign_tier_bin(values: pd.Series) -> pd.Categorical:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=TIER_BINS,
        labels=TIER_LABELS,
        right=False,
        include_lowest=True,
    )


def assign_checkpoint_bin(values: pd.Series) -> pd.Categorical:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=CHECKPOINT_BINS,
        labels=CHECKPOINT_LABELS,
        right=True,
        include_lowest=True,
    )


def corr_pair(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(df)
    out: Dict[str, float] = {"n": float(n)}
    if n < 3:
        return out
    xv = df["x"].astype(float).values
    yv = df["y"].astype(float).values
    out["Pearson_r"] = round(float(np.corrcoef(xv, yv)[0, 1]), 4)
    try:
        from scipy import stats as scipy_stats
        sr, sp = scipy_stats.spearmanr(xv, yv)
        out["Spearman_rho"] = round(float(sr), 4)
        out["Spearman_p"] = round(float(sp), 6)
    except Exception:
        pass
    return out


def _norm_oc(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip().lower()


def eval_benchmark_eight_family(provider, model, aimux_origin_code=None):
    m = safe_str(model).strip().lower()
    if not m:
        return None
    if "deepseek" in m:
        return "deepseek"
    if "qwen" in m or "通义" in safe_str(model):
        return "qwen"
    if "kimi" in m or "moonshot" in m:
        return "kimi"
    if "doubao" in m or "豆包" in safe_str(model):
        return "doubao"
    if "gemini" in m or "gemma" in m:
        return "gemini"
    if any(k in m for k in ("gpt-", "gpt_", "o1", "o3", "o4", "chatgpt")):
        return "gpt"
    if any(k in m for k in ("claude", "opus-4", "sonnet", "haiku")):
        return "claude"
    if "glm" in m or "chatglm" in m:
        return "glm"
    return None


def load_questions(questions_path: Path | None = None, sheet: str = "数据对齐") -> pd.DataFrame:
    """Load question bank (JSONL preferred; legacy xlsx supported)."""
    from lib.jsonl_store import load_questions_df

    path = Path(questions_path) if questions_path else resolve_questions_path()
    q = load_questions_df(path, source=None)
    if "source" not in q.columns:
        q["source"] = OURS_SOURCE
    picked = q.apply(pick_difficulty, axis=1, result_type="expand")
    q["difficulty_score"] = picked[0]
    q["difficulty_source_col"] = picked[1]
    stored = pd.to_numeric(q.get("query_len"), errors="coerce")
    computed = q["query"].map(query_length) if "query" in q.columns else np.nan
    q["query_len"] = stored.where(stored.notna(), computed)
    computed_cp = q.apply(checkpoint_count_for_row, axis=1)
    stored_cp = pd.to_numeric(q.get("checkpoint_n"), errors="coerce")
    q["checkpoint_n"] = stored_cp.where(stored_cp.notna() & (stored_cp > 0), computed_cp)
    q["difficulty_tier"] = assign_tier_bin(q["difficulty_score"])
    q["checkpoint_bin"] = assign_checkpoint_bin(q["checkpoint_n"])
    return q


def load_replies_with_scores(replies_path: Path | None, questions: pd.DataFrame) -> pd.DataFrame:
    from lib.jsonl_store import load_replies_df

    path = Path(replies_path) if replies_path else resolve_replies_path()
    src_map = questions[["qid", "source"]].drop_duplicates("qid").set_index("qid")["source"].astype(str).to_dict()
    r = load_replies_df(path, canonical_only=False)
    r["qid"] = r["qid"].map(normalize_qid)
    r["source"] = r["qid"].map(src_map)
    r["model"] = r["model"].astype(str).str.strip()
    r = r[~r["model"].str.lower().eq("ref")]
    r = r[~r["model"].astype(str).str.startswith("reply_slot_")]
    r = r[r.apply(reply_row_is_success, axis=1)]
    r["logical_model"] = r["model"].map(to_canonical)
    r[GPT54_MEAN_SCORE_COL] = r.apply(gpt54_mean_score, axis=1)
    r["score"] = r[GPT54_MEAN_SCORE_COL]
    r["score_col"] = PRIMARY_SCORE_LABEL
    return r


def ours_pk_qids(questions: pd.DataFrame, n: int = PK_OURS_N) -> set:
    """Ours 按 qid 排序取前 n 题（与 PK cohort 一致）。"""
    sub = questions[questions["source"].astype(str) == OURS_SOURCE].copy()
    qnum = pd.to_numeric(sub["qid"], errors="coerce")
    sub = sub.assign(_qnum=qnum).sort_values(["_qnum", "qid"]).head(n)
    return set(sub["qid"].astype(str))


def analysis_cohort_qids(questions: pd.DataFrame) -> set:
    """
    论文统计口径：
      - 公开四集：各 200 题（共 800）
      - Ours：qid 前 200 题（非全量 500）
    """
    pub = set(questions[questions["source"].astype(str).isin(PUBLIC_SOURCES)]["qid"].astype(str))
    return pub | ours_pk_qids(questions)


def filter_analysis_cohort(
    questions: pd.DataFrame, replies: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qids = analysis_cohort_qids(questions)
    q = questions[questions["qid"].astype(str).isin(qids)].copy()
    r = replies[replies["qid"].astype(str).isin(qids)].copy()
    return q, r


def _scored_rows_for_qid(replies: pd.DataFrame, qid: str, _source: str) -> pd.DataFrame:
    """PK / 跨集对比：仅统计 CANONICAL_8（向后兼容保留第三参）。"""
    g = replies[replies["qid"].astype(str) == str(qid)]
    g = g[g["logical_model"].isin(CANONICAL_8)]
    return g.dropna(subset=["score"])


def verify_cohort_completeness(questions: pd.DataFrame, replies: pd.DataFrame) -> pd.DataFrame:
    """检查各来源分析 cohort 的回复与评估覆盖（写入 Excel 00_数据覆盖）。"""
    rows: List[Dict] = []
    for src in SOURCE_ORDER:
        if src == OURS_SOURCE:
            qids = sorted(ours_pk_qids(questions))
            expected = PK_OURS_N
        else:
            qids = sorted(
                questions.loc[questions["source"].astype(str) == src, "qid"].astype(str).unique()
            )
            expected = PUBLIC_ANALYSIS_N
        n_any_reply = n_any_score = n_ge_min = n_both = 0
        for qid in qids:
            g = replies[
                (replies["qid"].astype(str) == str(qid))
                & replies["logical_model"].isin(CANONICAL_8)
            ]
            gs = _scored_rows_for_qid(replies, qid, src)
            if len(g) > 0:
                n_any_reply += 1
            if len(gs) > 0:
                n_any_score += 1
            if len(gs) >= MIN_MODELS:
                n_ge_min += 1
            if len(g) > 0 and len(gs) > 0:
                n_both += 1
        rows.append({
            "数据来源": src,
            "分析cohort题数": expected,
            "cohort内题数": len(qids),
            "有回复题数": n_any_reply,
            "有评估分题数": n_any_score,
            f"≥{MIN_MODELS}模型有分": n_ge_min,
            "回复+评估完整": n_both,
            "完整率_%": round(n_both / expected * 100, 1) if expected else np.nan,
            "计分列": score_col_for_source(src),
            "口径说明": "Ours=前200题" if src == OURS_SOURCE else "公开=全200题",
        })
    total = sum(r["分析cohort题数"] for r in rows)
    rows.append({
        "数据来源": "合计",
        "分析cohort题数": total,
        "cohort内题数": len(analysis_cohort_qids(questions)),
        "有回复题数": sum(r["有回复题数"] for r in rows),
        "有评估分题数": sum(r["有评估分题数"] for r in rows),
        f"≥{MIN_MODELS}模型有分": sum(r[f"≥{MIN_MODELS}模型有分"] for r in rows),
        "回复+评估完整": sum(r["回复+评估完整"] for r in rows),
        "完整率_%": round(sum(r["回复+评估完整"] for r in rows) / total * 100, 1) if total else np.nan,
        "计分列": PRIMARY_SCORE_LABEL,
        "口径说明": "800公开+200 Ours=1000",
    })
    return pd.DataFrame(rows)


def prepare_analysis_tables(
    questions_path: Path | None = None,
    replies_path: Path | None = None,
    *,
    sheet: str = "数据对齐",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载 → cohort 完整性（8 模型口径）→ 过滤 1000 题 cohort → **仅 CANONICAL_8 回复行** → 题级表。
    （DISCBench 特有 4 模型不计入 PK 对比；12 模型总榜另用全表 replies_all 计算。）
    返回 (coverage_report, questions_cohort, replies_cohort_pk8, items)
    """
    questions = load_questions(questions_path, sheet=sheet)
    replies = load_replies_with_scores(replies_path, questions)
    coverage = verify_cohort_completeness(questions, replies)
    questions, replies = filter_analysis_cohort(questions, replies)
    replies = replies[replies["logical_model"].isin(CANONICAL_8)].copy()
    items = build_item_level_table(questions, replies)
    return coverage, questions, replies, items


def build_item_level_table(questions: pd.DataFrame, replies: pd.DataFrame) -> pd.DataFrame:
    """题级指标（仅 questions 中 cohort 题目；需 ≥MIN_MODELS 有评估分才产出均分/D）。

    区分度 D（PK cohort · 跨数据集可比）：
      一律仅使用 ``CANONICAL_8`` 的模型分（与公开四集一致）；与
      ``generate_benchmark_source_summary_report.build_source_discrimination`` 口径对齐。
      DISCBench 12 模型专项见 ``_discbench_*`` 等表，不在此混用 12 模型算 D。
    """
    rows: List[Dict] = []
    qidx = questions.set_index("qid")

    for qid in qidx.index:
        qrow = qidx.loc[qid]
        if isinstance(qrow, pd.DataFrame):
            qrow = qrow.iloc[0]
        src = str(qrow["source"])
        g = replies[replies["qid"].astype(str) == str(qid)]
        g2 = g[g["logical_model"].isin(CANONICAL_8)].dropna(subset=["score"])
        scores = g2.groupby("model")["score"].first()
        has_reply = len(g) > 0
        has_score = len(scores) > 0
        if len(scores) < MIN_MODELS:
            metrics: Dict[str, float] = {}
        else:
            metrics = qdisc_from_scores(scores, min_n=MIN_MODELS)

        rows.append({
            "qid": qid,
            "source": src,
            "L1": qrow.get("L1", ""),
            "L2": qrow.get("L2", ""),
            "L3": qrow.get("L3", ""),
            "difficulty_score": qrow.get("difficulty_score"),
            "difficulty_tier": qrow.get("difficulty_tier"),
            "difficulty_source_col": qrow.get("difficulty_source_col"),
            "checkpoint_n": qrow.get("checkpoint_n"),
            "checkpoint_bin": qrow.get("checkpoint_bin"),
            "query_len": qrow.get("query_len"),
            "score_col_used": score_col_for_source(src),
            "has_reply": has_reply,
            "has_eval_score": has_score,
            "n_models_scored": len(scores),
            "mean_score": metrics.get("mean_score", np.nan),
            "disc": metrics.get("disc", np.nan),
            "spread": metrics.get("spread", np.nan),
            "item_perfect_mean": bool(metrics and metrics.get("mean_score") == PERFECT_SCORE),
            "any_model_perfect": bool((scores == PERFECT_SCORE).any()) if len(scores) else False,
            "all_models_perfect": bool((scores == PERFECT_SCORE).all()) if len(scores) else False,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["mean_score_tier"] = assign_tier_bin(df["mean_score"])
        df["item_difficulty"] = PERFECT_SCORE - df["mean_score"]
    return df


def cross_tab_count(sub: pd.DataFrame, row_col: str, col_col: str) -> pd.DataFrame:
    s = sub.dropna(subset=[row_col, col_col])
    if s.empty:
        return pd.DataFrame()
    ct = pd.crosstab(s[row_col], s[col_col], dropna=False)
    for lab in TIER_LABELS:
        if lab not in ct.index:
            ct.loc[lab] = np.nan
        if lab not in ct.columns:
            ct[lab] = np.nan
    ct = ct.reindex(index=TIER_LABELS, columns=TIER_LABELS)
    return ct


def display_source(source: str) -> str:
    return SOURCE_DISPLAY.get(str(source), str(source))


def apply_display_source(df: pd.DataFrame, col: str = "source") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = out[col].astype(str).map(display_source)
    return out


def tier_mean_score_pivot(items: pd.DataFrame) -> pd.DataFrame:
    """各 source × 难度档 → 题级均分的均值（无题为空；难度档来自 difficulty_score）。"""
    out = pd.DataFrame(index=SOURCE_ORDER_DISPLAY, columns=TIER_LABELS, dtype=float)
    for src in SOURCE_ORDER:
        sub = items[items["source"] == src]
        dsrc = display_source(src)
        for tier_label in TIER_LABELS:
            cell = sub[sub["difficulty_tier"].astype(str) == tier_label]
            out.loc[dsrc, tier_label] = round(float(cell["mean_score"].mean()), 2) if len(cell) else np.nan
    return out


def tier_mean_disc_pivot(items: pd.DataFrame) -> pd.DataFrame:
    """各 source × 难度档 → 题均区分度 D 的均值（disc 口径：CANONICAL_8，含 Ours，与 PK 跨集可比）。难度档来自 difficulty_score。"""
    out = pd.DataFrame(index=SOURCE_ORDER_DISPLAY, columns=TIER_LABELS, dtype=float)
    for src in SOURCE_ORDER:
        sub = items[items["source"] == src]
        dsrc = display_source(src)
        for tier_label in TIER_LABELS:
            cell = sub[
                (sub["difficulty_tier"].astype(str) == tier_label) & sub["disc"].notna()
            ]
            out.loc[dsrc, tier_label] = round(float(cell["disc"].mean()), 3) if len(cell) else np.nan
    return out


def tier_disc_count_pivot(items: pd.DataFrame) -> pd.DataFrame:
    """各 source × 难度档 → 有区分度 D 的题数（PK cohort）。"""
    out = pd.DataFrame(index=SOURCE_ORDER_DISPLAY, columns=TIER_LABELS, dtype=float)
    for src in SOURCE_ORDER:
        sub = items[items["source"] == src]
        dsrc = display_source(src)
        for tier_label in TIER_LABELS:
            n = int(
                (
                    (sub["difficulty_tier"].astype(str) == tier_label) & sub["disc"].notna()
                ).sum()
            )
            out.loc[dsrc, tier_label] = float(n) if n else np.nan
    return out


def tier_count_pivot_questions(questions: pd.DataFrame) -> pd.DataFrame:
    """各 source × 难度档 → 题数（题目表全量；统一 difficulty_score 分档）。"""
    q = questions.copy()
    q["difficulty_score"] = pd.to_numeric(q["difficulty_score"], errors="coerce")
    q["difficulty_tier"] = assign_tier_bin(q["difficulty_score"])
    out = pd.DataFrame(index=SOURCE_ORDER_DISPLAY, columns=TIER_LABELS, dtype=float)
    for src in SOURCE_ORDER:
        sub = q[q["source"].astype(str) == src]
        dsrc = display_source(src)
        for tier_label in TIER_LABELS:
            n = int((sub["difficulty_tier"].astype(str) == tier_label).sum())
            out.loc[dsrc, tier_label] = float(n) if n else np.nan
    return out


def tier_count_pivot(items: pd.DataFrame) -> pd.DataFrame:
    """各 source × 难度档 → 题数（评估 cohort；难度档来自 difficulty_score）。"""
    out = pd.DataFrame(index=SOURCE_ORDER_DISPLAY, columns=TIER_LABELS, dtype=float)
    for src in SOURCE_ORDER:
        sub = items[items["source"] == src]
        dsrc = display_source(src)
        for tier_label in TIER_LABELS:
            n = int((sub["difficulty_tier"].astype(str) == tier_label).sum())
            out.loc[dsrc, tier_label] = float(n) if n else np.nan
    return out


# 全项目热力图统一配色：浅蓝 = 低值，深蓝 = 高值
HEATMAP_CMAP = "Blues"


def plot_tier_count_and_mean(
    count_mat: pd.DataFrame,
    mean_mat: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Item count & mean model score by difficulty tier and dataset",
) -> None:
    """Dual panel: left=item count, right=mean item-level model score."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print(f"  ⚠️ 跳过 PNG（未安装 matplotlib）: {out_path.name}")
        return

    if count_mat is None or count_mat.empty or mean_mat is None or mean_mat.empty:
        return

    sns.set_theme(style="white", font="Arial Unicode MS", font_scale=0.95)
    n_rows, n_cols = count_mat.shape
    fig, axes = plt.subplots(1, 2, figsize=(max(14.0, 1.35 * n_cols + 8), max(4.5, 0.65 * n_rows + 1.8)))

    cnt = count_mat.astype(float)
    mean = mean_mat.astype(float)
    cnt_mask = cnt.isna() | (cnt <= 0)
    mean_mask = mean.isna()

    sns.heatmap(
        cnt,
        annot=True,
        fmt=".0f",
        cmap=HEATMAP_CMAP,
        ax=axes[0],
        mask=cnt_mask,
        linewidths=0.6,
        linecolor="#e8eef5",
        cbar_kws={"shrink": 0.82, "label": "Item count"},
        annot_kws={"size": 9},
    )
    axes[0].set_title("Item count (full question bank · difficulty_score bins)", fontsize=10, pad=8)
    axes[0].set_xlabel("Difficulty tier")
    axes[0].set_ylabel("Dataset")

    sns.heatmap(
        mean,
        annot=True,
        fmt=".1f",
        cmap=HEATMAP_CMAP,
        ax=axes[1],
        mask=mean_mask,
        linewidths=0.6,
        linecolor="#e8eef5",
        cbar_kws={"shrink": 0.82, "label": "Mean score"},
        annot_kws={"size": 9},
    )
    axes[1].set_title("Mean item-level model score (PK cohort · scored items)", fontsize=10, pad=8)
    axes[1].set_xlabel("Difficulty tier")
    axes[1].set_ylabel("")

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# 折线图：各数据集用高对比配色（与热力图蓝色系区分）
SOURCE_LINE_COLORS = {
    "CFbench": "#E63946",
    "infobench": "#1D3557",
    "ComplexBench": "#2A9D8F",
    "advancedif": "#F77F00",
    "DISCBench": "#8338EC",
}
SOURCE_LINE_MARKERS = {
    "CFbench": "o",
    "infobench": "s",
    "ComplexBench": "^",
    "advancedif": "D",
    "DISCBench": "P",
}


def _stagger_score_label_offsets(
    y_values: List[float],
    *,
    eps: float = 2.0,
    base_dy: float = 10.0,
) -> List[Tuple[float, float]]:
    """Return (dx, dy) point offsets for score labels that overlap at one x."""
    n = len(y_values)
    if n == 0:
        return []
    if n == 1:
        return [(0.0, base_dy)]

    order = sorted(range(n), key=lambda i: y_values[i], reverse=True)
    levels = [0] * n
    for rank, i in enumerate(order):
        yi = y_values[i]
        level = 0
        for j in order[:rank]:
            if abs(yi - y_values[j]) < eps:
                level = max(level, levels[j] + 1)
        levels[i] = level

    offsets: List[Tuple[float, float]] = []
    for level in levels:
        dy = base_dy + level * 12.0
        if level == 0:
            dx = 0.0
        else:
            sign = -1.0 if level % 2 else 1.0
            dx = sign * (8.0 + (level - 1) * 5.0)
        offsets.append((dx, dy))
    return offsets


def plot_tier_mean_score_lines(
    mean_mat: pd.DataFrame,
    out_path: Path,
    *,
    title: str | None = None,
    y_pad: float = 4.0,
) -> None:
    """Multi-dataset line chart: x=difficulty score bin, y=mean model score (adaptive y-range)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print(f"  ⚠️ 跳过 PNG（未安装 matplotlib）: {out_path.name}")
        return

    if mean_mat is None or mean_mat.empty:
        return

    mean = mean_mat.reindex(columns=TIER_LABELS).astype(float)
    vals = mean.values.flatten()
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return

    y_min = float(np.floor(min(vals) - y_pad))
    y_max = float(np.ceil(max(vals) + y_pad))
    y_min = max(0.0, y_min)
    y_max = min(100.0, y_max)

    sns.set_theme(style="white", font="Arial Unicode MS", font_scale=0.95)
    x = np.arange(len(TIER_LABELS))
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    annotations: List[Tuple[int, float, str]] = []

    for src_label, row in mean.iterrows():
        label = str(src_label)
        y = row.values.astype(float)
        mask = ~np.isnan(y)
        if not mask.any():
            continue
        color = SOURCE_LINE_COLORS.get(label, "#333333")
        marker = SOURCE_LINE_MARKERS.get(label, "o")
        ax.plot(
            x[mask],
            y[mask],
            color=color,
            marker=marker,
            linewidth=2.6,
            markersize=9,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1.2,
            label=label,
            zorder=3,
        )
        for xi, yi in zip(x[mask], y[mask]):
            annotations.append((int(xi), float(yi), color))

    from collections import defaultdict

    by_x: dict[int, List[Tuple[float, str, int]]] = defaultdict(list)
    for idx, (xi, yi, color) in enumerate(annotations):
        by_x[xi].append((yi, color, idx))

    label_offsets: dict[int, Tuple[float, float]] = {}
    for xi, group in by_x.items():
        ys = [g[0] for g in group]
        offsets = _stagger_score_label_offsets(ys)
        for (_, _, idx), (dx, dy) in zip(group, offsets):
            label_offsets[idx] = (dx, dy)

    for idx, (xi, yi, color) in enumerate(annotations):
        dx, dy = label_offsets.get(idx, (0.0, 10.0))
        ax.annotate(
            f"{yi:.1f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            fontsize=8,
            color=color,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(TIER_SCORE_TICK_LABELS, fontsize=9)
    ax.set_xlabel("Difficulty score", fontsize=10)
    ax.set_ylabel("Model Performance(mean)", fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, pad=12, fontweight="bold")
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.15, len(TIER_LABELS) - 0.85)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=min(5, len(mean)),
        fontsize=9,
        frameon=False,
    )
    ax.grid(axis="y", alpha=0.35, linestyle="--", color="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(
    mat: pd.DataFrame,
    out_path: Path,
    title: str,
    fmt: str = ".0f",
    cmap: str = HEATMAP_CMAP,
    mask_empty: bool = True,
    cbar_label: str = "Value",
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print(f"  ⚠️ 跳过 PNG（未安装 matplotlib）: {out_path.name}")
        return

    if mat is None or mat.empty:
        return
    plot_mat = mat.astype(float)
    mask = plot_mat.isna() if mask_empty else None
    sns.set_theme(style="white", font="Arial Unicode MS", font_scale=0.95)
    w = max(7.0, 0.9 * plot_mat.shape[1] + 2.5)
    h = max(4.0, 0.6 * plot_mat.shape[0] + 1.5)
    fig, ax = plt.subplots(figsize=(w, h))
    sns.heatmap(
        plot_mat,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        ax=ax,
        mask=mask,
        linewidths=0.6,
        linecolor="#e8eef5",
        cbar_kws={"shrink": 0.82, "label": cbar_label},
        annot_kws={"size": 9},
    )
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, pad=10)
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)


def plot_disc_by_difficulty_fine(
    items: pd.DataFrame,
    out_path: Path,
    *,
    bin_width: float = 4.0,
    min_count: int = 5,
    smooth_sigma: float = 1.2,
    title: str = "Preset difficulty vs. mean discrimination D (fine-grained bins)",
) -> pd.DataFrame:
    """Fine-grained difficulty bins → mean discrimination D with smoothed curve."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        print(f"  ⚠️ 跳过 PNG（未安装 matplotlib/scipy）: {out_path.name}")
        return pd.DataFrame()

    valid = items.dropna(subset=["difficulty_score", "disc"]).copy()
    if valid.empty:
        return pd.DataFrame()

    edges = np.arange(0, 100 + bin_width, bin_width)
    valid["diff_bin"] = pd.cut(
        valid["difficulty_score"], bins=edges, right=False, include_lowest=True
    )
    rows = []
    for interval, g in valid.groupby("diff_bin", observed=True):
        if len(g) < min_count:
            continue
        lo = float(interval.left)
        hi = float(interval.right)
        rows.append({
            "难度分区间": f"[{lo:.0f},{hi:.0f})",
            "区间中点": round((lo + hi) / 2, 1),
            "题数": len(g),
            "题均区分度D": round(float(g["disc"].mean()), 4),
            "区分度D_中位数": round(float(g["disc"].median()), 4),
        })
    df = pd.DataFrame(rows).sort_values("区间中点")
    if len(df) < 3:
        return df

    x = df["区间中点"].astype(float).values
    y = df["题均区分度D"].astype(float).values
    n = df["题数"].astype(float).values
    y_smooth = gaussian_filter1d(y, sigma=smooth_sigma)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(
        x, y, width=bin_width * 0.85, alpha=0.35, color="#9ecae1",
        edgecolor="#6baed6", linewidth=0.6, label="Mean D per bin",
    )
    ax.plot(x, y_smooth, color="#08519c", linewidth=2.5, label="Smoothed curve")
    ax.fill_between(x, y_smooth, alpha=0.18, color="#3182bd")
    ax.scatter(x, y, s=np.clip(n * 1.5, 12, 80), c="#2171b5", alpha=0.75, zorder=3, label="Item count (point size)")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Preset difficulty score (instruction quality)")
    ax.set_ylabel("Mean discrimination D")
    ax.set_title(title, fontsize=11, pad=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)
    return df


def plot_disc_by_difficulty_tier(
    items: pd.DataFrame,
    out_path: Path,
    title: str = "Mean discrimination D by difficulty tier",
) -> pd.DataFrame:
    """Difficulty tier → mean discrimination D (bar + line)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  ⚠️ 跳过 PNG（未安装 matplotlib）: {out_path.name}")
        return pd.DataFrame()

    valid = items.dropna(subset=["difficulty_tier", "disc"])
    rows = []
    for lab, short in zip(TIER_LABELS, TIER_SHORT):
        sub = valid[valid["difficulty_tier"] == lab]
        rows.append({
            "难度档": short,
            "题数": len(sub),
            "题均区分度D": round(float(sub["disc"].mean()), 3) if len(sub) else np.nan,
            "区分度D_中位数": round(float(sub["disc"].median()), 3) if len(sub) else np.nan,
        })
    df = pd.DataFrame(rows)
    if df["题均区分度D"].notna().sum() == 0:
        return df

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(TIER_SHORT))
    vals = df["题均区分度D"].astype(float).values
    bars = ax1.bar(x, vals, color="#9ecae1", edgecolor="#3182bd", linewidth=0.8, label="Mean discrimination D")
    ax1.plot(x, vals, color="#08519c", marker="o", linewidth=2, markersize=7, label="Trend")
    ax1.set_xticks(x)
    ax1.set_xticklabels(TIER_SHORT)
    ax1.set_xlabel("Difficulty tier (preset difficulty score)")
    ax1.set_ylabel("Mean discrimination D")
    ax1.set_ylim(0, max(0.01, np.nanmax(vals) * 1.15))
    ax1.set_title(title, fontsize=11, pad=10)
    ax1.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)
    return df


def plot_bar_simple(
    labels: List[str],
    values: List[float],
    out_path: Path,
    title: str,
    ylabel: str = "Value",
    fmt: str = ".1f",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  ⚠️ 跳过 PNG（未安装 matplotlib）: {out_path.name}")
        return

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.1), 4.2))
    x = np.arange(len(labels))
    vals = [float(v) if pd.notna(v) else 0 for v in values]
    ax.bar(x, vals, color="#6baed6", edgecolor="#2171b5", linewidth=0.8)
    for i, v in enumerate(vals):
        if pd.notna(values[i]):
            ax.text(i, v, format(v, fmt), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, pad=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)


def prep_table_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """透视表/带 index 的表导出前 reset_index。"""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if not isinstance(out.index, pd.RangeIndex):
        out = out.reset_index()
    return out


def write_stacked_tables(
    writer: pd.ExcelWriter,
    sheet_name: str,
    sections: List[Tuple[str, pd.DataFrame]],
    *,
    gap_rows: int = 2,
) -> int:
    """同一 sheet 内纵向写入多张表，每表前有节标题行（论文排版）。"""
    startrow = 0
    written = 0
    for title, df in sections:
        out = prep_table_for_export(df)
        if out.empty:
            continue
        if written > 0:
            startrow += gap_rows
        pd.DataFrame([[title]]).to_excel(
            writer, sheet_name=sheet_name, startrow=startrow, index=False, header=False
        )
        startrow += 1
        out.to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False)
        startrow += len(out) + 1
        written += 1
    if written == 0:
        pd.DataFrame([["（本节无数据）"]]).to_excel(
            writer, sheet_name=sheet_name, index=False, header=False
        )
    return written


def format_paper_excel_sheets(
    xlsx_path: Path,
    *,
    skip_sheets: Optional[Tuple[str, ...]] = None,
) -> None:
    """加粗节标题行与表头行，便于直接复制到论文。"""
    try:
        from openpyxl import load_workbook
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError:
        return

    skip = set(skip_sheets or ())
    skip.add("Charts")
    skip.add("统计图表")

    wb = load_workbook(xlsx_path)
    title_font = Font(bold=True, size=11, color="1F4E79")
    title_fill = PatternFill("solid", fgColor="D9E2F3")
    header_font = Font(bold=True, size=10)
    header_fill = PatternFill("solid", fgColor="F2F2F2")

    for ws in wb.worksheets:
        if ws.title in skip:
            continue
        title_rows: set = set()
        header_rows: set = set()
        for r in range(1, ws.max_row + 1):
            val = ws.cell(r, 1).value
            if isinstance(val, str) and (
                val.startswith("表") or val.startswith("【") or val.startswith("附录")
            ):
                title_rows.add(r)
                if r + 1 <= ws.max_row:
                    header_rows.add(r + 1)

        for r in title_rows:
            for c in range(1, ws.max_column + 1):
                cell = ws.cell(r, c)
                cell.font = title_font
                cell.fill = title_fill
                cell.alignment = Alignment(vertical="center")
            ws.row_dimensions[r].height = 22

        for r in header_rows:
            if r in title_rows:
                continue
            for c in range(1, ws.max_column + 1):
                cell = ws.cell(r, c)
                if cell.value is not None:
                    cell.font = header_font
                    cell.fill = header_fill

    wb.save(xlsx_path)


def embed_images_in_excel(
    xlsx_path: Path,
    images: List[Tuple[str, Path]],
    sheet_name: str = "Charts",
) -> None:
    """将 PNG 嵌入 Excel 单一 sheet（纵向排列）。"""
    try:
        from openpyxl import load_workbook
        from openpyxl.drawing.image import Image as XLImage
        from openpyxl.styles import Font
    except ImportError as e:
        print(f"  ⚠️ 跳过图表嵌入（需 Pillow: {e}）")
        return

    wb = load_workbook(xlsx_path)
    if sheet_name in wb.sheetnames:
        del wb[sheet_name]
    ws = wb.create_sheet(sheet_name)
    ws.column_dimensions["A"].width = 100
    row = 1
    for title, img_path in images:
        if not Path(img_path).is_file():
            continue
        ws.cell(row=row, column=1, value=title).font = Font(bold=True, size=12)
        row += 1
        img = XLImage(str(img_path))
        max_w = 760
        if img.width > max_w:
            scale = max_w / img.width
            img.width = int(img.width * scale)
            img.height = int(img.height * scale)
        ws.add_image(img, f"A{row}")
        row += int(img.height / 18) + 4
    wb.save(xlsx_path)
