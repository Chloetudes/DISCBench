# -*- coding: utf-8 -*-
"""
DISCBench 评测入口 — 四阶段复现（JSONL 数据 + staging 桥接）。

Stage 1  evaluate_instructions   指令质量评估
Stage 2  generate_replies        多模型生成回复
Stage 3  evaluate_replies        LLM 裁判评估（可前置 test_judge_models）
Stage 4  bash scripts/run_stats.sh  统计与图表（离线）

用法:
  export EVALUATION_PROJECT_CONFIG=data/config/stage1_instruction_quality.json
  python3 -m evaluation.main

或 bash scripts/run_stage1.sh / run_stage2.sh / run_stage3.sh
"""
import os
import sys

_this_file = os.path.abspath(__file__)
_evaluation_pkg_dir = os.path.dirname(_this_file)
_PROJECT_ROOT = os.path.dirname(_evaluation_pkg_dir)

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.config_loader import get_project_output_directory, resolve_config
from evaluation.pipeline import PipelineManager, SUPPORTED_STAGES

# 内置默认值；实际以 data/config/stage*.json 或 EVALUATION_PROJECT_CONFIG 为准
CONFIG = {
    "stages": ["evaluate_instructions"],
    "project_root_dir": ".",
    "output_base_dir": "output",
    "sysprompt_excel": "data/sysprompts.xlsx",
    "provider": "idealab",
    "model": "claude_sonnet4_5",
    "timeout": 300,
    "test_timeout": 60,
    "check_judge_before_use": True,
    "skip_judge_validation": False,
    "max_workers": 3,
    "checkpoint_interval": 10,
    "overwrite_mode": "skip",
}


def main() -> None:
    print(f"\n{'=' * 60}")
    print("DISCBench · 论文评测复现（精简版）")
    print(f"{'=' * 60}\n")

    base_dir = CONFIG.get("output_base_dir", "output")
    env_pr = (os.environ.get("EVALUATION_PROJECT_ROOT") or "").strip()
    if env_pr:
        project_root = os.path.abspath(env_pr)
    else:
        pr = CONFIG.get("project_root_dir")
        if pr and str(pr).strip():
            pr = str(pr).strip()
            project_root = os.path.abspath(pr if os.path.isabs(pr) else os.path.join(_PROJECT_ROOT, pr))
        else:
            project_root = _PROJECT_ROOT

    config = resolve_config(CONFIG, project_root, base_dir)
    pr_merged = (config.get("project_root_dir") or "").strip()
    if pr_merged:
        project_root = os.path.abspath(
            pr_merged if os.path.isabs(pr_merged) else os.path.join(_PROJECT_ROOT, pr_merged)
        )

    out_root = get_project_output_directory(project_root, base_dir, config.get("project_id") or "")
    batch = (config.get("data_batch") or "").strip()
    print(f"  📂 产物目录: {out_root}/" + (f"  批次: {batch}" if batch else "") + "\n")

    stages = config.get("stages") or []
    for s in stages:
        if s not in SUPPORTED_STAGES:
            print(f"❌ 不支持的阶段: {s}")
            print(f"   本精简版仅支持: {', '.join(SUPPORTED_STAGES)}")
            sys.exit(1)
    print(f"  📌 本次阶段: {stages}\n")

    pipeline = PipelineManager(config)

    if "test_judge_models" in stages:
        pipeline._judge_checked_this_run = False
        pipeline.execute_stage("test_judge_models")

    remaining = [s for s in stages if s != "test_judge_models"]
    if remaining:
        ok = pipeline.run(remaining, preserve_judge_selection="test_judge_models" in stages)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
