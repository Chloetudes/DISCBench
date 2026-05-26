#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 DISCBench/output/replies/七个模型/*.jsonl 导入 replies_compared_all.xlsx。

每行 jsonl 含：qid, gen(回复), meta_data(1~4_score/details, model)。
仅处理 Ours 500 题 × 7 模型；ref / reply_slot 不参与。
"""
from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
for p in (str(_ROOT), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lib.cif_stats_common import LEGACY_TO_CANONICAL, to_discbench_model  # noqa: E402
from lib.jsonl_store import load_replies_df, save_replies_jsonl  # noqa: E402

DEFAULT_JSONL_DIR = _ROOT / "output/replies/七个模型"
DEFAULT_REPLIES = _ROOT / "data/replies.jsonl"
# Optional monorepo mirror; not used unless --also-cif is passed explicitly.
CIF_REPLIES = _ROOT.parent / "projects/cif/output/replies/replies_compared_all.xlsx"

FILENAME_TO_MODEL = {
    "doubao-seed-2.0-pro.jsonl": "doubao-seed-2-0-pro",
    "doubao-seed-2-0-pro.jsonl": "doubao-seed-2-0-pro",
}

SCORE_COLS = ["1_score", "2_score", "3_score", "4_score"]
DETAIL_COLS = ["1_details", "2_details", "3_details", "4_details"]
EXCEL_CELL_MAX = 32700


def truncate_cell(val: Any, max_len: int = EXCEL_CELL_MAX) -> Any:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return val
    s = str(val)
    if len(s) <= max_len:
        return val
    return s[: max_len - 40] + "\n...[truncated for Excel cell limit]"


def truncate_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(lambda v: truncate_cell(v))
    return out


def normalize_qid(val) -> str:
    s = str(val).strip()
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
    except (ValueError, TypeError):
        pass
    return s


def canonical_model(name: str) -> str:
    m = str(name or "").strip()
    if m in LEGACY_TO_CANONICAL:
        return LEGACY_TO_CANONICAL[m]
    c = to_discbench_model(m)
    return c or m


def parse_meta(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    s = str(raw).strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return {}


def extract_reply(gen: Any) -> str:
    if gen is None:
        return ""
    if isinstance(gen, list):
        if len(gen) == 1 and isinstance(gen[0], str):
            s = gen[0].strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, list) and parsed:
                        return str(parsed[0]).strip()
                except (SyntaxError, ValueError):
                    pass
            return s
        return "\n".join(str(x).strip() for x in gen if x is not None and str(x).strip())
    return str(gen).strip()


def model_from_filename(path: Path) -> str:
    key = path.name
    if key in FILENAME_TO_MODEL:
        return FILENAME_TO_MODEL[key]
    return canonical_model(path.stem)


def load_jsonl_rows(jsonl_dir: Path) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    stats = {"files": 0, "lines": 0, "skipped_failed": 0, "skipped_empty_reply": 0}
    for fp in sorted(jsonl_dir.glob("*.jsonl")):
        stats["files"] += 1
        default_model = model_from_filename(fp)
        with fp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stats["lines"] += 1
                obj = json.loads(line)
                if obj.get("failed"):
                    stats["skipped_failed"] += 1
                    continue
                qid = normalize_qid(obj.get("qid"))
                meta = parse_meta(obj.get("meta_data"))
                model = canonical_model(meta.get("model") or default_model)
                reply = extract_reply(obj.get("gen"))
                if not reply or reply.startswith("<error"):
                    stats["skipped_empty_reply"] += 1
                    continue
                row = {
                    "qid": qid,
                    "model": model,
                    "reply": reply,
                    "source": str(meta.get("source") or obj.get("source") or "Ours"),
                    "status": "ok",
                    "reply_len": len(reply),
                    "timestamp": meta.get("timestamp") or datetime.now().isoformat(timespec="seconds"),
                }
                if meta.get("provider"):
                    row["provider"] = meta.get("provider")
                for c in SCORE_COLS + DETAIL_COLS:
                    if c in meta and meta[c] is not None and str(meta[c]).strip() != "":
                        row[c] = meta[c]
                rows.append(row)
    return rows, stats


def _row_key(row: pd.Series) -> Tuple[str, str]:
    return normalize_qid(row.get("qid")), canonical_model(row.get("model"))


def upsert_replies(df: pd.DataFrame, incoming: List[dict], *, overwrite: bool) -> Tuple[pd.DataFrame, dict]:
    rep = {"inserted": 0, "updated": 0, "skipped_existing": 0}
    if not incoming:
        return df, rep

    inc_df = pd.DataFrame(incoming)
    inc_df["qid"] = inc_df["qid"].map(normalize_qid)
    inc_df["model"] = inc_df["model"].map(canonical_model)

    if df.empty:
        rep["inserted"] = len(inc_df)
        return inc_df, rep

    df = df.copy()
    df["qid"] = df["qid"].map(normalize_qid)
    df["_key_qid"] = df["qid"]
    df["_key_model"] = df["model"].map(canonical_model)

    existing_idx: Dict[Tuple[str, str], int] = {}
    for i, row in df.iterrows():
        existing_idx[(str(row["_key_qid"]), str(row["_key_model"]))] = i

    new_rows = []
    for _, row in inc_df.iterrows():
        key = (str(row["qid"]), str(row["model"]))
        if key not in existing_idx:
            new_rows.append(row.to_dict())
            rep["inserted"] += 1
            continue
        idx = existing_idx[key]
        if not overwrite:
            rep["skipped_existing"] += 1
            continue
        for col, val in row.items():
            if col in ("qid", "model"):
                continue
            if pd.isna(val) or (isinstance(val, str) and not val.strip()):
                continue
            df.at[idx, col] = val
        rep["updated"] += 1

    if new_rows:
        add = pd.DataFrame(new_rows)
        df = pd.concat([df, add], ignore_index=True)

    df = df.drop(columns=["_key_qid", "_key_model"], errors="ignore")
    return df, rep


def import_to_table(replies_path: Path, incoming: List[dict], *, overwrite: bool, dry_run: bool) -> dict:
    if replies_path.is_file():
        if replies_path.suffix.lower() == ".jsonl":
            df = load_replies_df(replies_path, canonical_only=False)
            backup = replies_path.with_suffix(".jsonl.bak_import7")
        else:
            df = pd.read_excel(replies_path)
            backup = replies_path.with_suffix(".xlsx.bak_import7")
        if not dry_run:
            shutil.copy2(replies_path, backup)
    else:
        df = pd.DataFrame()
    merged, rep = upsert_replies(df, incoming, overwrite=overwrite)
    if not dry_run:
        replies_path.parent.mkdir(parents=True, exist_ok=True)
        if replies_path.suffix.lower() == ".jsonl":
            save_replies_jsonl(replies_path, merged)
        else:
            truncate_for_excel(merged).to_excel(replies_path, index=False)
    rep["path"] = str(replies_path)
    rep["total_rows"] = len(merged)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl-dir", type=Path, default=DEFAULT_JSONL_DIR)
    ap.add_argument("--replies", type=Path, default=DEFAULT_REPLIES)
    ap.add_argument(
        "--also-cif",
        action="store_true",
        help="Also write to sibling projects/cif reply table (monorepo dev only)",
    )
    ap.add_argument("--overwrite", action="store_true", default=True,
                    help="同 (qid,model) 已存在则覆盖 reply 与分数列")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    incoming, load_stats = load_jsonl_rows(args.jsonl_dir)
    models = sorted({r["model"] for r in incoming})
    qids = {r["qid"] for r in incoming}

    print(f"JSONL 目录: {args.jsonl_dir}")
    print(f"  文件: {load_stats['files']}  有效行: {len(incoming)}  "
          f"(跳过 failed={load_stats['skipped_failed']}, 空回复={load_stats['skipped_empty_reply']})")
    print(f"  模型: {', '.join(models)}")
    print(f"  qid 数: {len(qids)}")

    rep1 = import_to_table(args.replies, incoming, overwrite=args.overwrite, dry_run=args.dry_run)
    print(f"\n{'[dry-run] ' if args.dry_run else ''}thesis: {rep1['path']}")
    print(f"  新增 {rep1['inserted']}  更新 {rep1['updated']}  跳过 {rep1['skipped_existing']}  总行 {rep1['total_rows']}")

    if args.also_cif and CIF_REPLIES != args.replies:
        rep2 = import_to_table(CIF_REPLIES, incoming, overwrite=args.overwrite, dry_run=args.dry_run)
        print(f"\n{'[dry-run] ' if args.dry_run else ''}cif: {rep2['path']}")
        print(f"  新增 {rep2['inserted']}  更新 {rep2['updated']}  跳过 {rep2['skipped_existing']}  总行 {rep2['total_rows']}")



if __name__ == "__main__":
    main()
