#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DISCBench unified paths — JSONL-first, staging Excel for evaluation stages."""
from __future__ import annotations

from pathlib import Path

CIF_ROOT = Path(__file__).resolve().parents[2]
DISCBENCH_ROOT = CIF_ROOT  # alias for tooling / README
QUESTIONS_SHEET = "数据对齐"  # staging sheet name (stage bridge)
REPORTS_DIR = CIF_ROOT / "output/reports"
CHARTS_DIR = REPORTS_DIR / "charts"
DATA_DIR = CIF_ROOT / "data"
STAGING_DIR = DATA_DIR / "_staging"

# --- Canonical JSONL (source of truth) ---
QUESTIONS_JSONL = DATA_DIR / "questions.jsonl"
REPLIES_JSONL = DATA_DIR / "replies.jsonl"
DISCBENCH_OPEN_JSONL = DATA_DIR / "DISCbench_data.jsonl"
PUBLIC_OPEN_JSONL = DATA_DIR / "public_benchmark_data.jsonl"

# --- Single consolidated workbook (optional human edit / import) ---
MASTER_XLSX = DATA_DIR / "discbench_master.xlsx"

# --- Legacy paths (import only via consolidate_discbench_data.py) ---
LEGACY_QUESTIONS_XLSX = DATA_DIR / "论文数据_all.xlsx"
LEGACY_REPLIES_XLSX = DATA_DIR / "replies_compared_all.xlsx"

# --- Staging Excel (evaluation stages read/write; synced from JSONL) ---
STAGING_QUESTIONS_XLSX = STAGING_DIR / "questions.xlsx"
STAGING_REPLIES_XLSX = STAGING_DIR / "replies.xlsx"

# Back-compat aliases used by stats CLI defaults
QUESTIONS_XLSX = STAGING_QUESTIONS_XLSX
REPLIES_XLSX = STAGING_REPLIES_XLSX

SCHEMA_XLSX = DATA_DIR / "schema.xlsx"


def resolve_questions_path() -> Path:
    if QUESTIONS_JSONL.is_file():
        return QUESTIONS_JSONL
    if LEGACY_QUESTIONS_XLSX.is_file():
        return LEGACY_QUESTIONS_XLSX
    return QUESTIONS_JSONL


def resolve_replies_path() -> Path:
    if REPLIES_JSONL.is_file():
        return REPLIES_JSONL
    if LEGACY_REPLIES_XLSX.is_file():
        return LEGACY_REPLIES_XLSX
    return REPLIES_JSONL


def ensure_data_layout(verbose: bool = True) -> None:
    """
    Ensure canonical JSONL exists.
    If only legacy split xlsx present, run consolidate (import) automatically.
    """
    if QUESTIONS_JSONL.is_file() and REPLIES_JSONL.is_file():
        if verbose:
            print(f"  ✓ data store: {QUESTIONS_JSONL.name}, {REPLIES_JSONL.name}")
        return

    if LEGACY_QUESTIONS_XLSX.is_file() and LEGACY_REPLIES_XLSX.is_file():
        if verbose:
            print("  → importing legacy xlsx → JSONL (first-time setup)")
        import subprocess
        import sys

        script = CIF_ROOT / "scripts/consolidate_discbench_data.py"
        subprocess.run([sys.executable, str(script)], check=True, cwd=str(CIF_ROOT))
        return

    if DISCBENCH_OPEN_JSONL.is_file():
        if verbose:
            print("  → bootstrapping from DISCbench_data.jsonl (open PK subset only)")
        import subprocess
        import sys

        boot = CIF_ROOT / "scripts/bootstrap_from_open_jsonl.py"
        subprocess.run([sys.executable, str(boot)], check=True, cwd=str(CIF_ROOT))
        return

    if verbose:
        print("  ⚠ missing questions.jsonl / replies.jsonl")
        print("    run: python3 scripts/consolidate_discbench_data.py")
        print("    or:  python3 scripts/bootstrap_from_open_jsonl.py")


def staging_ready() -> bool:
    return STAGING_QUESTIONS_XLSX.is_file() and STAGING_REPLIES_XLSX.is_file()
