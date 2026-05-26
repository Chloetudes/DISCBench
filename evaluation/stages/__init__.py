# -*- coding: utf-8 -*-
"""CIF thesis 精简版：仅 Stage 1 指令质量 / Stage 2 回复生成 / Stage 3 回复评估。"""
from .stage1_quality import batch_evaluate_instruction_quality
from .stage3_reply import batch_generate_replies, batch_generate_replies_paired_round_robin
from .stage4_evaluate import batch_evaluate_responses_with_cache, save_results
