#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repair text encoding in open nested JSONL bundles (DISCbench / public).

Fixes:
  1. Latin-1 misread UTF-8 (mojibake)
  2. U+FFFD gaps inferred from other models' replies on the same question
  3. Curated phrase repairs for glm/claude rows where cross-model match fails

Then refreshes data/questions.jsonl + data/replies.jsonl via bootstrap.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
for p in (str(_ROOT), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lib.paths import DISCBENCH_OPEN_JSONL  # noqa: E402

TEXT_KEYS = ("query", "rubrics")

# Longest-first literal repairs (after generic passes)
PHRASE_REPAIRS: list[tuple[str, str]] = [
    ("附\ufffd\ufffd标号缺失", "附件标号缺失"),
    ("附\ufffd标号缺失", "附件标号缺失"),
    ("询\ufffd\ufffd）", "询问）"),
    ("询\ufffd）", "询问）"),
    ("语法\ufffd\ufffd可以", "语法上都可以"),
    ("语法\ufffd可以", "语法上可以"),
    ("建\ufffd\ufffd\ufffd通常", "建议通常"),
    ("主\ufffd\ufffd分析", "主题分析"),
    ("主\ufffd分析", "主题分析"),
    ("总生命\ufffd\ufffd变化", "总生命值变化"),
    ("复\ufffd\ufffd\ufffd期限", "复检期限"),
    ("德\ufffd\ufffd磁悬", "德日磁悬浮"),
    ("销售\ufffd\ufffd5%", "销售额5%"),
    ("的自\ufffd\ufffd变化", "的自然变化"),
    ("\ufffd\ufffd力类型", "暴力类型"),
    ("良好的\ufffd\ufffd展", "良好的发展"),
    ('不能放\ufffd\ufffd"', '不能放松"'),
    ('"一\ufffd\ufffd手"', '"一把手"'),
    ("warriors\ufffd\ufffd\ufffdbarbarian", "warriors、barbarian"),
    ("</情\ufffd\ufffd色彩>", "</情感色彩>"),
    ("目\ufffd\ufffd的VIP", "目前的VIP"),
    ("目\ufffd\ufffd\ufffd的VIP", "目前的VIP"),
    ("目\ufffd的VIP", "目前的VIP"),
    ("意图分\ufffd\ufffd", "意图分析"),
    ("也\ufffd\ufffd礼仪", "也是礼仪"),
    ("情绪趋\ufffd\ufffd平稳", "情绪趋于平稳"),
    ("心理状\ufffd\ufffd", "心理状态"),
    ("\ufffd\ufffd量达标", "质量达标"),
    ("重\ufffd\ufffd性", "重要性"),
    ("艰苦朴\ufffd\ufffd作风", "艰苦朴素作风"),
    ("症\ufffd\ufffd（", "症状（"),
    ("心理状态\ufffd：", "心理状态："),
    ("意图分析\ufffd：", "意图分析："),
    ("不能放\ufffd\ufffd", "不能放松"),
    ("目\ufffd的VIP", "目前的VIP"),
    ("症\ufffd（", "症状（"),
]

_CJK = r"[\u4e00-\u9fff]"


def _cjk_count(s: str) -> int:
    return sum("\u4e00" <= c <= "\u9fff" for c in s)


def fix_mojibake_latin1_utf8(text: str) -> str:
    if not text:
        return text
    try:
        fixed = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    if fixed == text:
        return text
    if _cjk_count(fixed) >= _cjk_count(text):
        return fixed
    return text


def fix_replacement_cross_model(text: str, refs: list[str], *, max_gap: int = 12) -> str:
    if "\ufffd" not in text:
        return text

    def repl(m: re.Match[str]) -> str:
        pre, post = m.group(1), m.group(2)
        broken = m.group(0)
        for ref in refs:
            if ref is text or "\ufffd" in ref:
                continue
            pat = re.escape(pre) + f".{{1,{max_gap}}}" + re.escape(post)
            hit = re.search(pat, ref)
            if not hit:
                continue
            mid = hit.group(0)[len(pre) : len(hit.group(0)) - len(post)]
            if mid and "\ufffd" not in mid:
                return pre + mid + post
        return broken

    return re.sub(rf"({_CJK}{{0,6}})\ufffd+({_CJK}{{0,6}})", repl, text)


def fix_replacement_phrases(text: str) -> str:
    for old, new in PHRASE_REPAIRS:
        text = text.replace(old, new)
    text = re.sub(r"目\ufffd+的VIP", "目前的VIP", text)
    # Drop stray replacement chars before full-width colon after CJK
    text = re.sub(rf"({_CJK})\ufffd+：", r"\1：", text)
    return text


def repair_text(text: str, refs: list[str] | None = None) -> str:
    if not text:
        return text
    out = fix_mojibake_latin1_utf8(text)
    if refs:
        out = fix_replacement_cross_model(out, refs)
    out = fix_replacement_phrases(out)
    return out


def repair_record(rec: dict) -> dict:
    replies = {
        mr.get("model"): (mr.get("reply") or "")
        for mr in rec.get("model_responses") or []
    }
    inst = rec.get("instruction") or {}
    for key in TEXT_KEYS:
        if key in inst and inst[key]:
            inst[key] = repair_text(str(inst[key]))
    rec["instruction"] = inst

    for mr in rec.get("model_responses") or []:
        model = mr.get("model")
        raw = mr.get("reply") or ""
        refs = [replies[m] for m in replies if m != model]
        mr["reply"] = repair_text(raw, refs)
    return rec


def load_records(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_records(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def audit(records: list[dict]) -> dict:
    stats = {"records": len(records), "reply_repl": 0, "inst_repl": 0, "mojibake_hint": 0}
    for rec in records:
        inst = rec.get("instruction") or {}
        for key in TEXT_KEYS:
            if "\ufffd" in str(inst.get(key) or ""):
                stats["inst_repl"] += 1
        for mr in rec.get("model_responses") or []:
            reply = mr.get("reply") or ""
            if "\ufffd" in reply:
                stats["reply_repl"] += 1
            if any(c in reply for c in "Ã©Ã¥Ã¤"):
                stats["mojibake_hint"] += 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DISCBENCH_OPEN_JSONL)
    ap.add_argument("--no-bootstrap", action="store_true", help="Only patch nested JSONL")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = args.input.resolve()
    before = audit(load_records(path))
    records = [repair_record(r) for r in load_records(path)]
    after = audit(records)

    print(f"Input: {path}")
    print(f"Before: {before}")
    print(f"After:  {after}")

    if args.dry_run:
        print("(dry-run, no files written)")
        return

    save_records(path, records)
    print(f"✓ wrote {path}")

    if not args.no_bootstrap:
        from bootstrap_from_open_jsonl import nested_to_flat  # noqa: E402

        nested_to_flat(path)
        print("✓ refreshed questions.jsonl + replies.jsonl")


if __name__ == "__main__":
    main()
