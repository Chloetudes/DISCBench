#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补全 论文数据_all.xlsx · 数据对齐 sheet：
  - query_len / constraint_n / checkpoint_n
  - L1/L2/L3（需 data/schema.xlsx；缺失则跳过）
  - difficulty_score / iq_count_* 等（从 instruction_quality_raw 本地重算）
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
for p in (str(_ROOT), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lib.schema_label_align import (
    align_questions_labels,
    default_schema_path,
    fill_missing_questions_labels,
)
from lib.cif_stats_common import checkpoint_count_for_row, count_checkpoints_from_rubrics, query_length
from lib.jsonl_store import load_questions_df, save_questions_jsonl
from lib.paths import LEGACY_QUESTIONS_XLSX, QUESTIONS_JSONL, QUESTIONS_SHEET
from evaluation.core.instruction_quality_scoring import (
    CONSTRAINT_TYPE_WEIGHTS,
    build_instruction_quality_row,
    extract_constraints_from_raw_regex,
    score_to_desc,
    score_to_grade,
)

IQ_COUNT_COLS = [f"iq_count_{k}" for k in CONSTRAINT_TYPE_WEIGHTS]
STAT_TYPE_MAP = {
    "教学约束": "iq_count_教学约束",
    "素材约束": "iq_count_素材约束",
    "流程步骤": "iq_count_流程步骤",
    "格式输出": "iq_count_格式输出",
    "边界范围": "iq_count_边界范围",
    "数量篇幅": "iq_count_数量篇幅",
}

FILL_IQ_COLS = [
    "instruction_quality_raw",
    "difficulty_score",
    "difficulty_score_py",
    "difficulty_score_model",
    "difficulty_score_delta",
    "difficulty_level",
    "difficulty_desc",
    "iq_numerator",
    "iq_denominator",
    "iq_constraint_json",
    "instruction_quality_parsed_json",
    "iq_qualified",
    "iq_filter_reason",
    "iq_parse_ok",
    "iq_status",
    "iq_error",
] + IQ_COUNT_COLS


def _is_empty(val) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return True
    s = str(val).strip()
    return s in ("", "nan", "None", "NaT")


def extract_constraints_from_raw(raw: str) -> List[Tuple[str, float]]:
    return [
        (str(c["类型"]), float(c["难度分"]))
        for c in extract_constraints_from_raw_regex(raw)
        if c.get("类型") and c.get("难度分") is not None
    ]


def extract_stats_from_raw(raw: str) -> Dict[str, Any]:
    """从统计段提取 iq_count / 总约束数等；不提取百分制作为主难度分。"""
    s = str(raw)
    out: Dict[str, Any] = {}
    m = re.search(r"总加权分\s*:\s*([\d.]+)", s)
    if m:
        out["iq_numerator"] = float(m.group(1))
    for label, col in STAT_TYPE_MAP.items():
        m = re.search(rf"^\s*{label}\s*:\s*(\d+)", s, re.MULTILINE)
        if m:
            out[col] = int(m.group(1))
    if any(k in out for k in IQ_COUNT_COLS):
        denom = sum(
            int(out.get(c, 0) or 0)
            for c in ("iq_count_流程步骤", "iq_count_格式输出", "iq_count_边界范围", "iq_count_数量篇幅")
        )
        if denom > 0:
            out["iq_denominator"] = denom
    m = re.search(r"^\s*总约束数\s*:\s*(\d+)", s, re.MULTILINE)
    if m:
        out["_total_constraints_stat"] = int(m.group(1))
    m = re.search(r"是否合格\s*:\s*(\w+)", s)
    if m:
        out["iq_qualified"] = m.group(1).lower() in ("true", "yes", "1")
    m = re.search(r"等级\s*:\s*([^\n]+)", s)
    if m:
        out["difficulty_level"] = m.group(1).strip()
    m = re.search(r"难度描述\s*:\s*([^\n]+)", s)
    if m:
        out["difficulty_desc"] = m.group(1).strip().strip('"')
    return out


def compute_from_constraints(constraints: List[Tuple[str, float]]) -> Dict[str, Any]:
    from evaluation.core.instruction_quality_scoring import compute_difficulty_score_py

    cons = [{"类型": t, "难度分": d} for t, d in constraints]
    calc = compute_difficulty_score_py({"约束列表": cons})
    score = calc["difficulty_score_py"]
    return {
        "difficulty_score": score,
        "difficulty_score_py": score,
        "iq_numerator": calc["numerator"],
        "iq_denominator": calc["denominator"],
        **{f"iq_count_{k}": v for k, v in calc["counts_by_type"].items()},
    }


def rescoring_row(raw: str) -> Dict[str, Any]:
    if _is_empty(raw):
        return {}

    built = build_instruction_quality_row(str(raw))
    if built.get("difficulty_score") is not None:
        built["iq_status"] = "rescored_py"
        built["iq_parse_ok"] = True
        return built

    # 仅约束块正则计分；禁止用评估结果里的百分制得分回填 difficulty_score
    cons = extract_constraints_from_raw(raw)
    computed = compute_from_constraints(cons) if cons else {}
    stats = extract_stats_from_raw(raw)

    out: Dict[str, Any] = dict(computed)
    for k, v in stats.items():
        if k.startswith("_"):
            continue
        if k in ("difficulty_score", "difficulty_score_py"):
            continue
        if not _is_empty(v) and _is_empty(out.get(k)):
            out[k] = v

    if out.get("difficulty_score") is not None:
        score = float(out["difficulty_score"])
        if _is_empty(out.get("difficulty_level")):
            out["difficulty_level"] = score_to_grade(score)
        if _is_empty(out.get("difficulty_desc")):
            out["difficulty_desc"] = score_to_desc(score)
        out["iq_parse_ok"] = bool(cons)
        out["iq_status"] = "rescored_regex"
        out["iq_error"] = "" if cons else "约束块为空"
    elif cons:
        out["iq_parse_ok"] = True
        out["iq_status"] = "rescored_partial"
        out["iq_error"] = "分母为 0; 已解析约束但未计分"
    return out


def constraint_n_for_row(row: pd.Series) -> float:
    if all(c in row.index for c in IQ_COUNT_COLS):
        vals = [pd.to_numeric(row.get(c), errors="coerce") for c in IQ_COUNT_COLS]
        if any(pd.notna(v) for v in vals):
            return float(sum(v for v in vals if pd.notna(v)))

    raw = row.get("instruction_quality_raw")
    if not _is_empty(raw):
        stats = extract_stats_from_raw(str(raw))
        if stats.get("_total_constraints_stat") is not None:
            return float(stats["_total_constraints_stat"])
        cons = extract_constraints_from_raw(str(raw))
        if cons:
            return float(len(cons))

    cp = pd.to_numeric(row.get("cp_classify_n_items"), errors="coerce")
    if pd.notna(cp) and cp > 0:
        return float(cp)

    rub = checkpoint_count_for_row(row)
    if pd.notna(rub) and rub > 0:
        return float(rub)
    return 0.0


def rubrics_checkpoint_n_for_row(row: pd.Series) -> float:
    for col in ("rubrics", "evaluation_criteria", "human_rubrics"):
        if col in row.index:
            n = count_checkpoints_from_rubrics(row.get(col))
            if pd.notna(n) and n > 0:
                return float(n)
    cp = pd.to_numeric(row.get("cp_classify_n_items"), errors="coerce")
    if pd.notna(cp) and cp > 0:
        return float(cp)
    return np.nan


def _set_cell(df: pd.DataFrame, idx, col: str, val) -> None:
    """写入单元格并尽量匹配列 dtype（避免 bool→float 报错）。"""
    if col not in df.columns or _is_empty(val):
        return
    series = df[col]
    if pd.api.types.is_bool_dtype(series.dtype):
        if isinstance(val, str):
            val = val.strip().lower() in ("true", "1", "yes", "是")
        df.at[idx, col] = bool(val)
    elif pd.api.types.is_numeric_dtype(series.dtype):
        if isinstance(val, bool):
            val = 1 if val else 0
        num = pd.to_numeric(val, errors="coerce")
        if pd.notna(num):
            df.at[idx, col] = float(num) if series.dtype != "int64" else int(round(float(num)))
    else:
        df.at[idx, col] = val


def _assign_cell(df: pd.DataFrame, idx, col: str, val) -> bool:
    if col not in df.columns:
        return False
    if _is_empty(val):
        return False
    cur = df.at[idx, col]
    if _is_empty(cur) or str(cur) != str(val):
        _set_cell(df, idx, col, val)
        return True
    return False


def _iq_counts_missing(row: pd.Series) -> bool:
    vals = [row.get(c) for c in IQ_COUNT_COLS if c in row.index]
    if not vals:
        return True
    if all(_is_empty(v) for v in vals):
        return True
    if not _is_empty(row.get("instruction_quality_raw")):
        total = sum(float(v) for v in vals if pd.notna(v))
        if total <= 0:
            return True
    return False


def enrich_dataframe(df: pd.DataFrame, *, refresh_from_raw: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    for col in ("query_len", "constraint_n", "checkpoint_n"):
        if col not in df.columns:
            df[col] = np.nan

    schema_path = default_schema_path(_ROOT)
    label_before = df["L1"].notna().sum() if "L1" in df.columns else 0
    if schema_path.is_file():
        if refresh_from_raw:
            df, unmapped = align_questions_labels(
                df,
                schema_xlsx=schema_path,
                keep_legacy="L3_legacy" in df.columns,
            )
            filled = int(df["L1"].notna().sum()) - label_before
            print(f"  刷新 L1/L2/L3（全表重对齐）: {df['L1'].notna().sum()}/{len(df)} 行有标签")
        else:
            df, filled, unmapped = fill_missing_questions_labels(df, schema_xlsx=schema_path)
            if filled:
                print(f"  补 L1/L2/L3: {filled} 行（原有 {label_before} 行）")
        if not unmapped.empty:
            print("  ⚠ 未映射 L3:")
            print(unmapped.to_string(index=False))
    else:
        print(f"  ⚠ 跳过 L1/L2/L3（无 {schema_path.relative_to(_ROOT)}）")
        unmapped = pd.DataFrame()

    log_rows: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        qid = row.get("qid")
        src = row.get("source")
        changes: List[str] = []
        had_l1 = pd.notna(row.get("L1"))

        qlen = query_length(row.get("query"))
        if not _is_empty(qlen):
            if _is_empty(row.get("query_len")) or float(row["query_len"]) != float(qlen):
                df.at[idx, "query_len"] = float(qlen)
                changes.append("query_len")

        cn = constraint_n_for_row(row)
        if _is_empty(row.get("constraint_n")) or float(row.get("constraint_n") or -1) != cn:
            df.at[idx, "constraint_n"] = cn
            changes.append("constraint_n")

        cpn = rubrics_checkpoint_n_for_row(row)
        if pd.notna(cpn):
            cur_cpn = pd.to_numeric(row.get("checkpoint_n"), errors="coerce")
            if _is_empty(cur_cpn) or float(cur_cpn) != float(cpn):
                df.at[idx, "checkpoint_n"] = float(cpn)
                changes.append("checkpoint_n")

        raw = row.get("instruction_quality_raw")
        if refresh_from_raw and not _is_empty(raw):
            need_iq = True
        else:
            need_iq = _is_empty(row.get("difficulty_score")) or _iq_counts_missing(row)
        if need_iq and not _is_empty(raw):
            rescored = rescoring_row(str(raw))
            for col in FILL_IQ_COLS:
                if col == "instruction_quality_raw":
                    continue
                if col in rescored:
                    if refresh_from_raw:
                        _set_cell(df, idx, col, rescored[col])
                        changes.append(col)
                    elif _assign_cell(df, idx, col, rescored[col]):
                        changes.append(col)

            iq_sum = sum(
                float(pd.to_numeric(df.at[idx, c], errors="coerce") or 0) for c in IQ_COUNT_COLS
            )
            if iq_sum > 0:
                df.at[idx, "constraint_n"] = iq_sum
                changes.append("constraint_n")
            elif not _is_empty(rescored.get("_total_constraints_stat")):
                df.at[idx, "constraint_n"] = float(rescored["_total_constraints_stat"])
                changes.append("constraint_n")

            cpn2 = rubrics_checkpoint_n_for_row(df.loc[idx])
            if pd.notna(cpn2):
                df.at[idx, "checkpoint_n"] = float(cpn2)
                changes.append("checkpoint_n")

        if not had_l1 and pd.notna(df.at[idx, "L1"]):
            changes.extend(["L1", "L2", "L3"])

        if changes:
            log_rows.append({"qid": qid, "source": src, "更新字段": ",".join(sorted(set(changes)))})

    return df, pd.DataFrame(log_rows)


def apply_sheet1_difficulty_scores(path: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Sheet1.score 为 instruction_quality 原始算分结果，写回 difficulty_score 供统计分档。"""
    xl = pd.ExcelFile(path)
    if "Sheet1" not in xl.sheet_names:
        return df
    s1 = pd.read_excel(path, sheet_name="Sheet1")
    if "qid" not in s1.columns or "score" not in s1.columns:
        return df

    s1 = s1[["qid", "score"]].copy()
    s1["score"] = pd.to_numeric(s1["score"], errors="coerce")
    s1 = s1.dropna(subset=["score"])
    score_map = dict(zip(s1["qid"].astype(str), s1["score"].astype(float)))
    if not score_map:
        return df

    out = df.copy()
    updated = 0
    for idx, row in out.iterrows():
        qid = str(row.get("qid", "")).strip()
        if qid not in score_map:
            continue
        score = float(score_map[qid])
        out.at[idx, "difficulty_score"] = score
        if "difficulty_score_py" in out.columns:
            out.at[idx, "difficulty_score_py"] = score
        if "difficulty_level" in out.columns:
            out.at[idx, "difficulty_level"] = score_to_grade(score)
        if "difficulty_desc" in out.columns:
            out.at[idx, "difficulty_desc"] = score_to_desc(score)
        updated += 1

    ours_mask = out["source"].astype(str).isin(("Ours", "DISCBench"))
    s_cnt = int((pd.to_numeric(out.loc[ours_mask, "difficulty_score"], errors="coerce") >= 80).sum())
    print(f"  Sheet1 → difficulty_score: {updated} 题 | Ours S档(≥80): {s_cnt}")
    return out


def write_questions_xlsx(path: Path, sheet: str, enriched: pd.DataFrame) -> None:
    xl = pd.ExcelFile(path)
    sheets = {name: pd.read_excel(path, sheet_name=name) for name in xl.sheet_names}
    sheets[sheet] = enriched
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, sdf in sheets.items():
            sdf.to_excel(w, sheet_name=name, index=False)


def print_coverage(df: pd.DataFrame) -> None:
    print("\n=== 补全后覆盖 ===")
    print(f"query_len: {df['query_len'].notna().sum()}/{len(df)}")
    print(f"constraint_n: {df['constraint_n'].notna().sum()}/{len(df)}")
    print(f"checkpoint_n: {df['checkpoint_n'].notna().sum()}/{len(df)}")
    if "L1" in df.columns:
        print(f"L1/L2/L3: {df['L1'].notna().sum()}/{len(df)}")
    print(f"difficulty_score: {df['difficulty_score'].notna().sum()}/{len(df)}")
    print("\n按来源字段覆盖:")
    for src, g in df.groupby("source"):
        l1 = int(g["L1"].notna().sum()) if "L1" in g.columns else 0
        cpn = int(g["checkpoint_n"].notna().sum()) if "checkpoint_n" in g.columns else 0
        miss_d = int(g["difficulty_score"].isna().sum())
        print(f"  {src}: L1={l1}/{len(g)}, checkpoint_n={cpn}/{len(g)}, 缺难度={miss_d}")


def enrich_one_path(
    path: Path,
    sheet: str,
    log_path: Path,
    *,
    refresh_from_raw: bool = False,
    apply_sheet1_scores: bool = True,
) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        df = load_questions_df(path, source=None)
        sheet1_path = LEGACY_QUESTIONS_XLSX if LEGACY_QUESTIONS_XLSX.is_file() else path
    else:
        df = pd.read_excel(path, sheet_name=sheet)
        sheet1_path = path
    enriched, log = enrich_dataframe(df, refresh_from_raw=refresh_from_raw)
    if apply_sheet1_scores and sheet1_path.suffix.lower() == ".xlsx":
        enriched = apply_sheet1_difficulty_scores(sheet1_path, enriched)
    if path.suffix.lower() == ".jsonl":
        save_questions_jsonl(path, enriched)
    else:
        write_questions_xlsx(path, sheet, enriched)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    agg: Dict[str, Tuple[str, Any]] = {
        "题数": ("qid", "count"),
        "query_len": ("query_len", lambda s: int(s.notna().sum())),
        "constraint_n": ("constraint_n", lambda s: int(s.notna().sum())),
        "checkpoint_n": ("checkpoint_n", lambda s: int(s.notna().sum())),
        "difficulty_score": ("difficulty_score", lambda s: int(s.notna().sum())),
    }
    cov = enriched.groupby("source").agg(**agg).reset_index()
    with pd.ExcelWriter(log_path, engine="openpyxl") as w:
        cov.to_excel(w, sheet_name="覆盖汇总", index=False)
        if not log.empty:
            log.to_excel(w, sheet_name="变更明细", index=False)

    print(f"✓ 已写回: {path} [{sheet}]")
    print(f"✓ 日志: {log_path}")
    print_coverage(enriched)
    return enriched


def main() -> None:
    ap = argparse.ArgumentParser(description="补全论文题目表：query_len / constraint_n / checkpoint_n / L1-L3 等")
    ap.add_argument("--questions", default=str(QUESTIONS_JSONL))
    ap.add_argument("--sheet", default=QUESTIONS_SHEET)
    ap.add_argument("--log", default=str(_ROOT / "output/reports/questions_enrich_log.xlsx"))
    ap.add_argument(
        "--refresh-from-raw",
        action="store_true",
        help="instruction_quality_raw 已更新时：强制重算 difficulty_score/iq_* 并刷新 L1/L2/L3",
    )
    ap.add_argument(
        "--sync-all",
        action="store_true",
        help="(deprecated) 同默认，仅写回 data/questions.jsonl",
    )
    ap.add_argument(
        "--no-sheet1-scores",
        action="store_true",
        help="不用 Sheet1.score 覆盖 difficulty_score（默认会覆盖，供统计分档）",
    )
    ap.add_argument(
        "--apply-sheet1-only",
        action="store_true",
        help="仅将 Sheet1.score 写回 difficulty_score，不做其它 enrich",
    )
    ap.add_argument(
        "--run-stage1-missing",
        action="store_true",
        help="对仍缺 instruction_quality_raw 的题调用 Stage1 API（需 config.py）",
    )
    args = ap.parse_args()

    apply_sheet1 = not args.no_sheet1_scores
    paths = [Path(args.questions)]
    enriched = None
    for i, path in enumerate(paths):
        if not path.is_file():
            print(f"⚠ 跳过不存在: {path}")
            continue
        print(f"\n=== 处理 ({i + 1}/{len(paths)}): {path} ===")
        log_path = Path(args.log) if len(paths) == 1 else Path(args.log).with_name(
            f"{Path(args.log).stem}_{path.parent.name}{Path(args.log).suffix}"
        )
        if args.apply_sheet1_only:
            df = pd.read_excel(path, sheet_name=args.sheet)
            enriched = apply_sheet1_difficulty_scores(path, df)
            write_questions_xlsx(path, args.sheet, enriched)
            print_coverage(enriched)
            continue
        enriched = enrich_one_path(
            path,
            args.sheet,
            log_path,
            refresh_from_raw=args.refresh_from_raw,
            apply_sheet1_scores=apply_sheet1,
        )

    if enriched is None:
        raise SystemExit("未找到可处理的题目表")

    still_missing_raw = enriched["instruction_quality_raw"].isna() | enriched["instruction_quality_raw"].astype(str).str.strip().isin(["", "nan"])
    n_missing_raw = int(still_missing_raw.sum())
    if args.run_stage1_missing and n_missing_raw > 0:
        print(
            f"\n⚠ 仍有 {n_missing_raw} 题缺 instruction_quality_raw / difficulty_score。"
            "\n  请在 DISCBench 目录运行 Stage1（需 config.py API）："
            "\n    bash scripts/run_stage1.sh"
            "\n  完成后再次执行："
            "\n    python3 scripts/enrich_questions_table.py --sync-all"
        )
        print(enriched.loc[still_missing_raw, "source"].value_counts().to_string())


if __name__ == "__main__":
    main()
