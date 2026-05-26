# -*- coding: utf-8 -*-
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

from config import get_provider
from clients.openai_client import OAIClient
from ..core.utils import safe_str, safe_save_excel
from ..core.instruction_quality_scoring import (
    build_instruction_quality_row,
    merge_instruction_quality_into_questions,
)
from ..managers.sysprompt import SyspromptManager
from ..managers.constraint_library import ConstraintLibraryManager
from .stage3_reply import _filter_questions_by_source_include


def _ensure_qid_column(df: pd.DataFrame, qid_col: str = 'qid', prefix: str = 'iq') -> pd.DataFrame:
    df = df.copy()
    if qid_col in df.columns and df[qid_col].notna().any():
        df[qid_col] = df[qid_col].astype(str).str.strip()
        empty = df[qid_col].isin(('', 'nan', 'None')) | df[qid_col].isna()
        if not empty.all():
            if empty.any():
                for idx in df.index[empty]:
                    df.at[idx, qid_col] = f'{prefix}_{idx:05d}'
            return df
    df[qid_col] = [f'{prefix}_{i:05d}' for i in range(len(df))]
    return df


def _load_input_dataframe(
    input_excel: str,
    sheet_name: Any = 0,
    qid_col: str = 'qid',
    qid_prefix: str = 'iq',
) -> pd.DataFrame:
    df = pd.read_excel(input_excel, sheet_name=sheet_name)
    if 'query' not in df.columns:
        raise ValueError(f"缺少必需列 query（文件: {input_excel}）")
    return _ensure_qid_column(df, qid_col=qid_col, prefix=qid_prefix)


_iq_tls = threading.local()


def _iq_pool_initializer(
    provider: str,
    model: str,
    sysprompt_manager: SyspromptManager,
    temperature: float,
    timeout: int,
) -> None:
    """每 worker 线程各建一个 OAIClient，避免多线程共用一个 session。"""
    pc = get_provider(provider)
    _iq_tls.client = OAIClient(
        base_url=pc.base_url, api_key=pc.api_key, protocol=pc.protocol,
        auth_header=pc.auth_header, auth_prefix=pc.auth_prefix,
        extra_headers=pc.extra_headers, timeout=timeout,
    )
    _iq_tls.evaluator = InstructionQualityEvaluator(
        _iq_tls.client, model, sysprompt_manager, temperature,
    )


def _instruction_quality_one_task(
    task: Dict[str, Any],
    *,
    rescoring_only: bool,
    use_model_weights: bool,
    passthrough_cols: List[str],
    qid_col: str,
) -> Tuple[str, Dict[str, Any]]:
    """单题：返回 (qid, result_row)。"""
    if rescoring_only:
        raw_response = task.get('_cached_raw', '')
        error_msg = '' if raw_response else 'missing raw_response'
    else:
        raw_response, error_msg = _iq_tls.evaluator.evaluate(task['query'])

    if error_msg:
        return task[qid_col], {'_failed': True, '_error_row': {
            qid_col: task[qid_col], 'query': task['query'],
            'raw_response': raw_response, 'error': error_msg,
            'timestamp': pd.Timestamp.now(),
        }}

    scored = _apply_scoring_to_row(raw_response, use_model_weights)
    result_row = {
        qid_col: task[qid_col], 'query': task['query'],
        'raw_response': raw_response, 'error': error_msg,
        'timestamp': pd.Timestamp.now(),
    }
    result_row.update(scored)
    for col in passthrough_cols:
        if col in task:
            result_row[col] = task[col]
    return task[qid_col], result_row


def _apply_scoring_to_row(raw_response: str, use_model_weights: bool) -> Dict[str, Any]:
    try:
        scored = build_instruction_quality_row(raw_response, use_model_weights=use_model_weights)
        scored['status'] = 'ok'
        scored['iq_status'] = 'ok'
        scored['iq_error'] = ''
        if not scored.get('iq_parse_ok'):
            scored['iq_status'] = 'parse_warn'
            scored['iq_error'] = 'YAML/约束列表解析为空'
        return scored
    except Exception as e:
        return {
            'instruction_quality_raw': raw_response,
            'status': 'parse_error',
            'iq_status': 'parse_error',
            'iq_error': str(e),
            'iq_parse_ok': False,
        }


class InstructionQualityEvaluator:
    def __init__(self, client: OAIClient, model: str,
                 sysprompt_manager: SyspromptManager, temperature: float = 0.3):
        self.client = client
        self.model = model
        self.sysprompt_manager = sysprompt_manager
        self.temperature = temperature

    def evaluate(self, query: str) -> tuple:
        sys_prompt = self.sysprompt_manager.get('instruction_quality_evaluation', '')
        user_prompt = f"请评估以下指令的质量，并提取所有约束。\n\n【指令】\n{query}"

        try:
            messages = []
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": user_prompt})
            response = self.client.chat(
                model=self.model, messages=messages, temperature=self.temperature,
            )
            return response, ""
        except Exception as e:
            return "", f"<error: {str(e)}>"


def batch_evaluate_instruction_quality(
        input_excel: str,
        output_excel: str,
        provider: str,
        model: str,
        sysprompt_manager: SyspromptManager,
        constraint_library: ConstraintLibraryManager,
        temperature: float = 0.3,
        max_workers: int = 4,
        checkpoint_interval: int = 10,
        timeout: int = 120,
        *,
        input_sheet: Any = 0,
        questions_excel: Optional[str] = None,
        questions_sheet: Any = None,
        merge_back: bool = True,
        qid_col: str = 'qid',
        qid_prefix: str = 'iq',
        use_model_weights: bool = False,
        rescoring_only: bool = False,
        force_rerun: bool = False,
        source_include: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    指令质量评估：调用模型输出 YAML → Python 按固定权重公式计分 → 可选写回题目表。

    rescoring_only=True 时仅根据已有 instruction_quality_raw / raw_response 重算分数，不调 API。
    """
    print(f"\n{'=' * 60}")
    print(f"🚀 Stage 1: 指令质量评估（模型约束识别 + Python 客观计分）")
    print(f"{'=' * 60}\n")

    df = _load_input_dataframe(input_excel, sheet_name=input_sheet, qid_col=qid_col, qid_prefix=qid_prefix)
    rerun_qids: Optional[Set[str]] = None
    if source_include:
        df = _filter_questions_by_source_include(df, source_include)
        rerun_qids = {safe_str(x).strip() for x in df[qid_col].tolist() if safe_str(x).strip()}
    has_reference = 'reference' in df.columns
    print(f"  输入: {input_excel}  sheet={input_sheet!r}  行数: {len(df)}\n")

    evaluator = None
    if not rescoring_only:
        try:
            pc = get_provider(provider)
            client = OAIClient(
                base_url=pc.base_url, api_key=pc.api_key, protocol=pc.protocol,
                auth_header=pc.auth_header, auth_prefix=pc.auth_prefix,
                extra_headers=pc.extra_headers, timeout=timeout,
            )
            evaluator = InstructionQualityEvaluator(client, model, sysprompt_manager, temperature)
            print(f"✅ 评估模型: {provider} / {model}\n")
        except Exception as e:
            raise RuntimeError(f"评估模型初始化失败，流程终止: {e}")

    passthrough_cols = [
        'reference', 'original_id', 'item_num', 'task_type', 'session_id', 'turn_id',
        'history_context', '标题', 'difficulty', 'source', 'sub_source', 'rubrics',
        'category', 'L1', 'L2', 'L3',
    ]

    results: List[Dict[str, Any]] = []
    existing_qids = set()

    raw_col_existing = 'instruction_quality_raw' if 'instruction_quality_raw' in df.columns else 'raw_response'

    if force_rerun:
        if rerun_qids:
            print(
                f"  🔄 force_rerun=True：仅重跑 source_include 子集（{len(rerun_qids)} 题），"
                "其余保留 stage1 缓存\n"
            )
        else:
            print("  🔄 force_rerun=True：忽略已有 stage1 缓存，全部重新调用 API / 重算\n")
    if os.path.exists(output_excel) and (not force_rerun or rerun_qids):
        try:
            df_existing = pd.read_excel(output_excel)
            for _, row in df_existing.iterrows():
                qid = safe_str(row.get(qid_col, row.get('qid', '')))
                if not qid:
                    continue
                if force_rerun and rerun_qids and qid in rerun_qids:
                    continue
                existing_qids.add(qid)
                results.append(row.to_dict())
            print(f"💾 保留已有 stage1 结果: {len(existing_qids)} 条")
        except Exception:
            pass

    tasks = []
    for _, row in df.iterrows():
        qid = safe_str(row[qid_col])
        if qid in existing_qids and not rescoring_only:
            continue
        task = {qid_col: qid, 'query': safe_str(row['query'])}
        if has_reference:
            task['reference'] = safe_str(row.get('reference', ''))
        for col in passthrough_cols:
            if col in df.columns:
                task[col] = safe_str(row[col])
        if rescoring_only:
            prev = safe_str(row.get(raw_col_existing, ''))
            if not prev and os.path.exists(output_excel):
                try:
                    prev_df = pd.read_excel(output_excel)
                    if qid_col in prev_df.columns and raw_col_existing in prev_df.columns:
                        m = prev_df[prev_df[qid_col].astype(str).str.strip() == qid]
                        if not m.empty:
                            prev = safe_str(m.iloc[0][raw_col_existing])
                except Exception:
                    pass
            task['_cached_raw'] = prev
        tasks.append(task)

    print(f"📝 待处理: {len(tasks)} 条  （rescoring_only={rescoring_only}）")
    if tasks and not rescoring_only:
        w = max(1, int(max_workers or 1))
        est = len(tasks) * 22 / 60 / w
        print(
            f"  ⏱  约 {len(tasks)} 次 API 调用；单线程粗估 {len(tasks) * 22 / 60:.0f} 分钟，"
            f"max_workers={w} 时理想约 **{est:.0f} 分钟**（实际受网关限流影响）\n"
        )
    else:
        print()
    sys.stdout.flush()

    if not tasks:
        if results:
            df_result = pd.DataFrame(results)
            if len(df_result) > 0 and qid_col in df_result.columns:
                n0 = len(df_result)
                df_result = df_result.drop_duplicates(subset=[qid_col], keep='last').reset_index(drop=True)
                if len(df_result) < n0:
                    print(
                        f"  ⚠️  stage1 汇总: 按 {qid_col} 去重 {n0} → {len(df_result)}（避免写回 merge 笛卡尔积）"
                    )
            if len(df_result) > 0:
                if 'difficulty_score_delta' in df_result.columns:
                    warn = df_result[df_result['difficulty_score_delta'].fillna(0) > 5]
                    if len(warn) > 0:
                        print(f"\n⚠️  Python 分与模型自报分差 >5 共 {len(warn)} 条（见 difficulty_score_delta）")
                if safe_save_excel(df_result, output_excel):
                    print(
                        f"\n✅ 无新 API 任务：已刷新 stage1 文件 {output_excel}  共 {len(df_result)} 条"
                    )
                if merge_back and questions_excel:
                    q_sheet = questions_sheet if questions_sheet is not None else input_sheet
                    if merge_instruction_quality_into_questions(
                        questions_excel, df_result, sheet_name=q_sheet, qid_col=qid_col,
                    ):
                        print(f"✅ 已写回题目表: {questions_excel}  sheet={q_sheet!r}")
            return df_result
        print("✅ 无新任务（题目与 stage1 均无待处理行）")
        return pd.DataFrame()

    new_results: List[Dict[str, Any]] = []
    failed_count = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3
    workers = max(1, int(max_workers or 1))

    def _sequential_loop() -> None:
        nonlocal failed_count, consecutive_failures
        progress = tqdm(tasks, desc="🔄 指令质量评估", ncols=100, total=len(tasks))
        for i, task in enumerate(progress):
            try:
                if rescoring_only:
                    raw_response = task.get('_cached_raw', '')
                    error_msg = '' if raw_response else 'missing raw_response'
                else:
                    raw_response, error_msg = evaluator.evaluate(task['query'])

                if error_msg:
                    if not rescoring_only:
                        tqdm.write(f"⚠️  {task[qid_col]} API 失败")
                    failed_count += 1
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES and not rescoring_only:
                        raise RuntimeError(f"连续 {MAX_CONSECUTIVE_FAILURES} 次评估失败")
                    continue

                consecutive_failures = 0
                scored = _apply_scoring_to_row(raw_response, use_model_weights)
                result_row = {
                    qid_col: task[qid_col],
                    'query': task['query'],
                    'raw_response': raw_response,
                    'error': error_msg,
                    'timestamp': pd.Timestamp.now(),
                }
                result_row.update(scored)
                for col in passthrough_cols:
                    if col in task:
                        result_row[col] = task[col]
                new_results.append(result_row)

                if not rescoring_only and (i + 1) % checkpoint_interval == 0:
                    df_temp = pd.DataFrame(results + new_results)
                    if safe_save_excel(df_temp, output_excel):
                        tqdm.write(f"💾 检查点: {len(df_temp)} 条")

            except RuntimeError:
                raise
            except Exception as e:
                if not rescoring_only:
                    tqdm.write(f"❌ {task[qid_col]}: {e}")
                failed_count += 1
                consecutive_failures += 1

    def _parallel_pool() -> None:
        nonlocal failed_count, consecutive_failures
        done_lock = threading.Lock()
        completed = 0
        print(f"  ⚙️  并发: max_workers={workers}（每线程独立 HTTP 客户端）\n")
        sys.stdout.flush()

        with ThreadPoolExecutor(
            max_workers=workers,
            initializer=_iq_pool_initializer,
            initargs=(provider, model, sysprompt_manager, temperature, timeout),
        ) as ex:
            future_map = {
                ex.submit(
                    _instruction_quality_one_task,
                    task,
                    rescoring_only=rescoring_only,
                    use_model_weights=use_model_weights,
                    passthrough_cols=passthrough_cols,
                    qid_col=qid_col,
                ): task
                for task in tasks
            }
            progress = tqdm(total=len(tasks), desc="🔄 指令质量评估", ncols=100)
            for fut in as_completed(future_map):
                task = future_map[fut]
                try:
                    _qid, row = fut.result()
                except Exception as e:
                    with done_lock:
                        failed_count += 1
                        consecutive_failures += 1
                        completed += 1
                    if not rescoring_only:
                        tqdm.write(f"❌ {task.get(qid_col)}: {e}")
                    progress.update(1)
                    continue

                with done_lock:
                    if row.get('_failed'):
                        failed_count += 1
                        consecutive_failures += 1
                        if not rescoring_only:
                            tqdm.write(f"⚠️  {task[qid_col]} API 失败")
                    else:
                        consecutive_failures = 0
                        new_results.append(row)
                    completed += 1
                    n_new = len(new_results)
                    if not rescoring_only and n_new > 0 and n_new % checkpoint_interval == 0:
                        df_temp = pd.DataFrame(results + new_results)
                        if safe_save_excel(df_temp, output_excel):
                            tqdm.write(f"💾 检查点: {len(df_temp)} 条")
                progress.update(1)
            progress.close()

    if workers > 1 and not rescoring_only:
        _parallel_pool()
    else:
        if workers > 1 and rescoring_only:
            print("  ℹ️  rescoring_only 模式不使用多线程，按单线程执行。\n")
        _sequential_loop()

    if rescoring_only:
        # 重算模式：以输入表 qid 为准覆盖/合并
        by_qid = {safe_str(r[qid_col]): r for r in new_results}
        merged = []
        seen = set()
        for _, row in df.iterrows():
            qid = safe_str(row[qid_col])
            if qid in by_qid:
                merged.append(by_qid[qid])
                seen.add(qid)
        for qid, r in by_qid.items():
            if qid not in seen:
                merged.append(r)
        results = merged
    else:
        results.extend(new_results)

    df_result = pd.DataFrame(results) if results else pd.DataFrame()

    if len(df_result) > 0 and qid_col in df_result.columns:
        n0 = len(df_result)
        df_result = df_result.drop_duplicates(subset=[qid_col], keep='last').reset_index(drop=True)
        if len(df_result) < n0:
            print(f"  ⚠️  stage1 汇总: 按 {qid_col} 去重 {n0} → {len(df_result)}（避免写回 merge 笛卡尔积）")

    if len(df_result) > 0:
        # 大 delta 告警
        if 'difficulty_score_delta' in df_result.columns:
            warn = df_result[df_result['difficulty_score_delta'].fillna(0) > 5]
            if len(warn) > 0:
                print(f"\n⚠️  Python 分与模型自报分差 >5 共 {len(warn)} 条（见 difficulty_score_delta）")

        if safe_save_excel(df_result, output_excel):
            print(f"\n✅ stage1 结果: {output_excel}  共 {len(df_result)} 条")

        if merge_back and questions_excel:
            q_sheet = questions_sheet if questions_sheet is not None else input_sheet
            if merge_instruction_quality_into_questions(
                questions_excel, df_result, sheet_name=q_sheet, qid_col=qid_col,
            ):
                print(f"✅ 已写回题目表: {questions_excel}  sheet={q_sheet!r}")

    if failed_count > 0:
        print(f"\n⚠️  失败: {failed_count} 条")

    return df_result
