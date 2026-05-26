#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按本地 schema.xlsx「任务体系」（任务意图 L1/L2/L3）对齐题目表标签。

题目表中的 L3 多为历史/合成口径（如「分类标记」「总结改写」），
通过别名映射到 schema 标准 L3 后回填 L1、L2、L3。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# instruction_quality_raw 任务意图块：优先在前 head_chars 字符内匹配
TASK_INTENT_TYPE_RE = re.compile(r"^\s*类型\s*[:：]\s*([^\n\r]+)", re.MULTILINE)
DEFAULT_IQ_HEAD_CHARS = 50

# 题目表 L3 → schema「任务体系」标准 L3
L3_ALIASES: Dict[str, str] = {
    "分类标记": "标签分类",
    "总结改写": "总结摘要",
    "逻辑推理": "文字推理",
    "推理消解": "指代消解",
    "翻译": "中外翻译",
    "语篇语义理解": "语篇理解",
    "数学推理": "数理推理",
    "弱智吧/文字游戏": "文化文字游戏",
    "歇后语/诗词": "诗词理解",
    "闲聊对话": "闲聊",
}

TASK_INTENT_SHEET_CANDIDATES = ("任务体系", "任务意图分类体系", "Sheet1")


def default_schema_path(cif_root: Optional[Path] = None) -> Path:
    root = cif_root or Path(__file__).resolve().parents[2]
    return root / "data" / "schema.xlsx"


def load_task_intent_taxonomy(schema_xlsx: Path) -> pd.DataFrame:
    """读取任务意图分类体系（L1/L2/L3 唯一组合）。"""
    schema_xlsx = Path(schema_xlsx)
    if not schema_xlsx.is_file():
        raise FileNotFoundError(f"未找到 schema: {schema_xlsx}")

    xl = pd.ExcelFile(schema_xlsx)
    sheet = next((s for s in TASK_INTENT_SHEET_CANDIDATES if s in xl.sheet_names), xl.sheet_names[0])
    df = pd.read_excel(schema_xlsx, sheet_name=sheet)
    for col in ("L1", "L2", "L3"):
        if col not in df.columns:
            raise ValueError(f"schema 表「{sheet}」缺少列 {col}")
    out = (
        df[["L1", "L2", "L3"]]
        .astype(str)
        .apply(lambda s: s.str.strip())
        .drop_duplicates(subset=["L3"])
    )
    dup = df.groupby("L3").size()
    dup = dup[dup > 1]
    if not dup.empty:
        raise ValueError(f"schema L3 重复定义: {dup.index.tolist()}")
    return out.reset_index(drop=True)


def build_l3_lookup(taxonomy: pd.DataFrame) -> Dict[str, Tuple[str, str, str]]:
    """L3(标准) -> (L1, L2, L3)。"""
    lookup: Dict[str, Tuple[str, str, str]] = {}
    for _, row in taxonomy.iterrows():
        l3 = str(row["L3"]).strip()
        lookup[l3] = (str(row["L1"]).strip(), str(row["L2"]).strip(), l3)
    return lookup


def normalize_iq_raw(raw: object) -> str:
    """统一 instruction_quality_raw 为可解析 YAML 文本（含 JSON/转义兜底）。"""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return ""
    m = re.search(r"```yaml\\n(.*?)```", s, re.DOTALL)
    if m:
        return (
            m.group(1)
            .replace("\\n", "\n")
            .replace("\\'", "'")
            .replace('\\"', '"')
        )
    m = re.search(r"```yaml\n(.*?)```", s, re.DOTALL)
    if m:
        return m.group(1)
    if "\\n" in s and s.count("\n") < 3:
        s = s.replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"')
    return s


def parse_task_intent_type_from_iq_raw(
    raw: object,
    *,
    head_chars: int = DEFAULT_IQ_HEAD_CHARS,
) -> Optional[str]:
    """从 instruction_quality_raw 提取「任务意图 → 类型」（L2/L3 历史名）。"""
    s = normalize_iq_raw(raw)
    if not s:
        return None
    block_m = re.search(r"任务意图\s*:\s*\n(.*?)(?:\n组织形式\s*:|$)", s, re.DOTALL)
    search_texts = []
    if block_m:
        search_texts.append(block_m.group(1))
    if head_chars and head_chars > 0:
        search_texts.append(s[:head_chars])
    search_texts.append(s)
    for text in search_texts:
        m = TASK_INTENT_TYPE_RE.search(text)
        if not m:
            continue
        t = m.group(1).strip()
        if t and len(t) <= 40:
            return t
    return None


def fill_l3_from_instruction_quality_raw(
    questions: pd.DataFrame,
    *,
    raw_col: str = "instruction_quality_raw",
    l3_col: str = "L3",
    source_filter: Optional[str] = None,
    schema_xlsx: Optional[Path] = None,
    head_chars: int = DEFAULT_IQ_HEAD_CHARS,
    overwrite_missing_only: bool = True,
) -> Tuple[pd.DataFrame, int]:
    """
    用 instruction_quality_raw 中的任务类型补全 L3（再经 align 映射到 schema）。

    overwrite_missing_only=True 时仅填补空 L3 或无法 canonicalize 的行。
    返回 (df, filled_count)。
    """
    if raw_col not in questions.columns:
        return questions, 0

    schema_xlsx = schema_xlsx or default_schema_path()
    lookup = build_l3_lookup(load_task_intent_taxonomy(schema_xlsx))

    q = questions.copy()
    filled = 0
    mask = pd.Series(True, index=q.index)
    if source_filter is not None and "source" in q.columns:
        mask = q["source"].astype(str).str.strip() == str(source_filter).strip()

    for idx in q.index[mask]:
        raw_l3 = q.at[idx, l3_col] if l3_col in q.columns else pd.NA
        need = False
        if overwrite_missing_only:
            if pd.isna(raw_l3) or not str(raw_l3).strip() or str(raw_l3).strip().lower() == "nan":
                need = True
            elif canonicalize_l3(str(raw_l3), lookup) is None:
                need = True
        else:
            need = True
        if not need:
            continue
        t = parse_task_intent_type_from_iq_raw(q.at[idx, raw_col], head_chars=head_chars)
        if not t:
            continue
        q.at[idx, l3_col] = t
        filled += 1
    return q, filled


def canonicalize_l3(raw_l3: str, lookup: Dict[str, Tuple[str, str, str]]) -> Optional[str]:
    if raw_l3 is None or (isinstance(raw_l3, float) and pd.isna(raw_l3)):
        return None
    s = str(raw_l3).strip()
    if not s or s.lower() == "nan":
        return None
    canon = L3_ALIASES.get(s, s)
    return canon if canon in lookup else None


def fill_missing_questions_labels(
    questions: pd.DataFrame,
    *,
    schema_xlsx: Optional[Path] = None,
    l3_col: str = "L3",
    raw_col: str = "instruction_quality_raw",
) -> Tuple[pd.DataFrame, int, pd.DataFrame]:
    """
    仅对 L1 为空的行，从 instruction_quality_raw 补 L3 并回填 L1/L2/L3。
    已有标签的行保持不变。
    """
    schema_xlsx = schema_xlsx or default_schema_path()
    lookup = build_l3_lookup(load_task_intent_taxonomy(schema_xlsx))

    q = questions.copy()
    for col in ("L1", "L2", l3_col):
        if col not in q.columns:
            q[col] = pd.NA

    missing = q["L1"].isna() | q["L1"].astype(str).str.strip().isin(["", "nan"])
    filled = 0
    unmapped = []

    for idx in q.index[missing]:
        raw_l3 = q.at[idx, l3_col]
        need_l3 = (
            pd.isna(raw_l3)
            or not str(raw_l3).strip()
            or str(raw_l3).strip().lower() == "nan"
            or canonicalize_l3(str(raw_l3), lookup) is None
        )
        if need_l3 and raw_col in q.columns:
            t = parse_task_intent_type_from_iq_raw(q.at[idx, raw_col])
            if t:
                q.at[idx, l3_col] = t

        canon = canonicalize_l3(q.at[idx, l3_col], lookup)
        if canon is None:
            raw_val = q.at[idx, l3_col]
            if pd.notna(raw_val) and str(raw_val).strip() and str(raw_val).strip().lower() != "nan":
                unmapped.append({"index": idx, "L3_raw": str(raw_val).strip()[:80]})
            continue
        l1, l2, l3 = lookup[canon]
        q.at[idx, "L1"] = l1
        q.at[idx, "L2"] = l2
        q.at[idx, l3_col] = l3
        filled += 1

    report = pd.DataFrame(unmapped)
    if not report.empty:
        report = report.groupby("L3_raw", as_index=False).size().rename(columns={"size": "行数"})
    return q, filled, report


def align_questions_labels(
    questions: pd.DataFrame,
    *,
    schema_xlsx: Optional[Path] = None,
    l3_col: str = "L3",
    keep_legacy: bool = True,
    legacy_col: str = "L3_legacy",
    iq_raw_col: Optional[str] = "instruction_quality_raw",
    iq_source_filter: Optional[str] = None,
    iq_head_chars: int = DEFAULT_IQ_HEAD_CHARS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    为题目表填充/覆盖 L1、L2、L3（schema 标准名）。

    返回 (aligned_df, unmapped_report)。
    """
    schema_xlsx = schema_xlsx or default_schema_path()
    taxonomy = load_task_intent_taxonomy(schema_xlsx)
    lookup = build_l3_lookup(taxonomy)

    q = questions.copy()
    if iq_raw_col and iq_raw_col in q.columns:
        q, _ = fill_l3_from_instruction_quality_raw(
            q,
            raw_col=iq_raw_col,
            l3_col=l3_col,
            source_filter=iq_source_filter,
            schema_xlsx=schema_xlsx,
            head_chars=iq_head_chars,
        )
    if l3_col not in q.columns:
        raise ValueError(f"题目表缺少 {l3_col}")

    if keep_legacy and legacy_col not in q.columns:
        q[legacy_col] = q[l3_col]

    unmapped = []
    l1_list, l2_list, l3_list = [], [], []
    for idx, raw in q[l3_col].items():
        canon = canonicalize_l3(raw, lookup)
        if canon is None:
            l1_list.append(pd.NA)
            l2_list.append(pd.NA)
            l3_list.append(pd.NA)
            if pd.notna(raw) and str(raw).strip() and str(raw).strip().lower() != "nan":
                unmapped.append({"index": idx, "L3_raw": str(raw).strip()})
            continue
        l1, l2, l3 = lookup[canon]
        l1_list.append(l1)
        l2_list.append(l2)
        l3_list.append(l3)

    q["L1"] = l1_list
    q["L2"] = l2_list
    q[l3_col] = l3_list

    report = pd.DataFrame(unmapped)
    if not report.empty:
        report = report.groupby("L3_raw", as_index=False).size().rename(columns={"size": "行数"})
    return q, report
