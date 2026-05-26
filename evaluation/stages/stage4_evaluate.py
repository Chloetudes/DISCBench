# -*- coding: utf-8 -*-
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import get_provider
from clients.openai_client import OAIClient
from ..core.utils import safe_str, safe_save_excel
from ..core.cache_messages import build_cached_messages, detect_provider_type
from ..managers.sysprompt import SyspromptManager

# 题目表横表列 reply1..N / 回复1..N 等注入回复表时的最大槽位数（含 ref 之外的模型回复）
WIDE_REPLY_ANCHOR_MAX = 8
_CN_ANCHOR = ['一', '二', '三', '四', '五', '六', '七', '八']


def _wide_reply_text_column_candidates(idx_anchor: int) -> list:
    cands = [f'reply{idx_anchor}', f'回复{idx_anchor}']
    if 1 <= idx_anchor <= len(_CN_ANCHOR):
        cands.append(f'回复{_CN_ANCHOR[idx_anchor - 1]}')
    return cands


def _coalesce_merge_xy_columns(df: pd.DataFrame, col: str, prefer: str = "left") -> None:
    """
    回复表 merge 题目表时，若两侧均有同名列（如 session_id），pandas 会生成 col_x / col_y。
    合并为单一 col：默认 prefer=left（_x 来自回复表，与待评 reply 行一致）。
    """
    if col in df.columns:
        return
    cx, cy = f"{col}_x", f"{col}_y"
    if cx not in df.columns and cy not in df.columns:
        return
    if cx in df.columns and cy in df.columns:
        if prefer == "left":
            df[col] = df[cx].combine_first(df[cy])
        else:
            df[col] = df[cy].combine_first(df[cx])
    elif cx in df.columns:
        df[col] = df[cx]
    else:
        df[col] = df[cy]
    df.drop(columns=[c for c in (cx, cy) if c in df.columns], inplace=True, errors="ignore")


def is_evaluated(value) -> bool:
    """仅 eval_* 分数列有有效数值时视为已评测；raw 列、<error> 占位均不算完成。"""
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return not np.isnan(value)
    value_str = str(value).strip().lower()
    if value_str in ('', 'nan', 'none', 'null', 'na', '<na>'):
        return False
    if value_str.startswith('<error'):
        return False
    try:
        float(value_str)
        return True
    except ValueError:
        return False


def count_eval_numeric_scores(df: pd.DataFrame, eval_col: str) -> int:
    """回复表中 eval_* 分数列的有效数值行数（用于保存前后核对）。"""
    if df is None or df.empty or eval_col not in df.columns:
        return 0
    return int(df[eval_col].apply(is_evaluated).sum())


def _eval_result_has_valid_score(result: dict) -> bool:
    if result.get('status') != 'ok':
        return False
    ts = result.get('total_score')
    if ts is None:
        return False
    if isinstance(ts, float) and np.isnan(ts):
        return False
    return True


def _cell_scalar(val) -> str:
    """写回 Excel 单元格前强制为标量字符串，避免 list/Series 触发 pandas 对齐错误。"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ''
    if isinstance(val, (list, tuple)):
        return safe_str(val[0]) if val else ''
    if isinstance(val, pd.Series):
        return safe_str(val.iloc[0]) if len(val) else ''
    if isinstance(val, np.ndarray):
        return safe_str(val.flat[0]) if val.size else ''
    return safe_str(val)


def _col_iloc_index(df: pd.DataFrame, col: str) -> int:
    loc = df.columns.get_loc(col)
    if isinstance(loc, int):
        return loc
    if isinstance(loc, slice):
        return int(loc.start or 0)
    return int(loc[0])


def _read_reply_cell(df_replies: pd.DataFrame, pos: int, col: str):
    if col not in df_replies.columns:
        return np.nan
    if 0 <= int(pos) < len(df_replies):
        return df_replies.iloc[int(pos), _col_iloc_index(df_replies, col)]
    return np.nan


def _write_reply_cell(df_replies: pd.DataFrame, pos: int, col: str, value) -> None:
    if not col or col not in df_replies.columns:
        return
    if not (0 <= int(pos) < len(df_replies)):
        return
    df_replies.iloc[int(pos), _col_iloc_index(df_replies, col)] = value


def _apply_eval_result_to_df(df_replies: pd.DataFrame, result: dict, lock: threading.Lock) -> str:
    """
    将单条评估结果写回回复表。仅 status=ok 且分数有效时写入分数列；
    失败时不以 NaN 覆盖已有分数。返回: 'scored' | 'api_error' | 'parse_fail' | 'skipped'.
    reply_idx / reply_indices 为行号（iloc），与 merge 后 reset_index 的回复表一致。
    """
    result_eval_col = result.get('eval_column', '')
    result_raw_col = result.get('eval_raw_column', '')
    if not result_eval_col:
        return 'skipped'

    write_indices = result.get('reply_indices') or []
    if not write_indices and result.get('reply_idx') is not None:
        write_indices = [result['reply_idx']]

    raw_s = _cell_scalar(result.get('raw_evaluation', '')).strip()
    is_api_err = raw_s.startswith('<error')
    score_ok = _eval_result_has_valid_score(result)
    score_val = result.get('total_score')
    if score_ok:
        try:
            score_val = float(score_val)
        except (TypeError, ValueError):
            score_ok = False

    with lock:
        if result_eval_col not in df_replies.columns:
            df_replies[result_eval_col] = np.nan
        if result_raw_col and result_raw_col not in df_replies.columns:
            df_replies[result_raw_col] = ''

        outcome = 'skipped'
        for idx in write_indices:
            if idx is None:
                continue
            try:
                pos = int(idx)
            except (TypeError, ValueError):
                continue
            if not (0 <= pos < len(df_replies)):
                continue
            if score_ok:
                _write_reply_cell(df_replies, pos, result_eval_col, score_val)
                if result_raw_col:
                    _write_reply_cell(df_replies, pos, result_raw_col, raw_s)
                outcome = 'scored'
            elif is_api_err:
                if not is_evaluated(_read_reply_cell(df_replies, pos, result_eval_col)):
                    if result_raw_col:
                        existing_raw = _cell_scalar(_read_reply_cell(df_replies, pos, result_raw_col)).strip()
                        if not existing_raw or existing_raw.startswith('<error'):
                            _write_reply_cell(df_replies, pos, result_raw_col, raw_s)
                outcome = 'api_error' if outcome != 'scored' else outcome
            else:
                if not is_evaluated(_read_reply_cell(df_replies, pos, result_eval_col)) and result_raw_col:
                    _write_reply_cell(df_replies, pos, result_raw_col, raw_s)
                outcome = 'parse_fail' if outcome != 'scored' else outcome
        return outcome


def extract_scores_from_evaluation(eval_text: str) -> Dict[str, float]:
    scores = {}
    if not eval_text or not isinstance(eval_text, str):
        return scores

    # 优先解析 JSON 中的 FINAL_SCORE（如 "60% (18/30)"），取百分号前的数字
    try:
        text = eval_text.strip()
        if text.startswith('{') or '{"FINAL_SCORE"' in text or '"FINAL_SCORE"' in text:
            # 提取 FINAL_SCORE 的值
            m = re.search(r'"FINAL_SCORE"\s*:\s*"(\d+)%\s*\(\d+/\d+\)"', text)
            if m:
                scores['total_score'] = float(m.group(1))
                return scores
            # 兼容 "FINAL_SCORE": "80" 或 "80%"
            m = re.search(r'"FINAL_SCORE"\s*:\s*"(\d+)(?:%)?[^"]*"', text)
            if m:
                scores['total_score'] = float(m.group(1))
                return scores
    except Exception:
        pass

    total_patterns = [
        r'总分[:：]\s*(\d+(?:\.\d+)?)\s*(?:/|分)',
        r'总分\s*=\s*(\d+(?:\.\d+)?)',
        r'总体得分[:：]\s*(\d+(?:\.\d+)?)',
        r'最终得分[:：]\s*(\d+(?:\.\d+)?)',
        r'score[:：]\s*(\d+(?:\.\d+)?)',
        r'分数[:：]\s*(\d+(?:\.\d+)?)',
        r'总得分[:：]\s*(\d+(?:\.\d+)?)'
    ]
    for pattern in total_patterns:
        match = re.search(pattern, eval_text, re.IGNORECASE)
        if match:
            try:
                scores['total_score'] = float(match.group(1))
                break
            except Exception:
                pass

    if 'total_score' not in scores:
        # 匹配 "60% (18/30)" 或 "80% (24/27)" 等格式（含 FINAL_SCORE 的 JSON 或纯文本）
        m = re.search(r'(\d+)%\s*\(\d+/\d+\)', eval_text)
        if m:
            scores['total_score'] = float(m.group(1))
        else:
            for pattern in [r'(\d+(?:\.\d+)?)\s*/\s*\d+(?:\.\d+)?\s*分',
                             r'(\d+(?:\.\d+)?)\s*分\s*/\s*\d+(?:\.\d+)?',
                             r'得分\s*(\d+(?:\.\d+)?)\s*分']:
                match = re.search(pattern, eval_text)
                if match:
                    try:
                        scores['total_score'] = float(match.group(1))
                        break
                    except Exception:
                        pass

    for pattern in [r'(\w+)[:：]\s*(\d+(?:\.\d+)?)\s*(?:/|分)',
                    r'(\w+)\s*=\s*(\d+(?:\.\d+)?)',
                    r'(\w+)\s*得分[:：]\s*(\d+(?:\.\d+)?)']:
        for key, value in re.findall(pattern, eval_text, re.IGNORECASE):
            try:
                scores[key.lower().replace(' ', '_')] = float(value)
            except Exception:
                pass

    return scores


def evaluate_single_reply_with_cache(
        client: OAIClient,
        eval_model: str,
        provider_type: str,
        sys_prompt: str,
        query: str,
        evaluation_criteria: str,
        reference: str,
        reply: str,
        model_name: str,
        reference_type: str = 'model',
        temperature: float = 0.3,
        retries: int = 3,
        history_context: str = '',
        expert_opinions: str = '',
        conversation_sysprompt: str = '',
        ground_truth: str = '',
) -> Tuple[str, str]:
    messages = build_cached_messages(
        provider_type, sys_prompt, query, evaluation_criteria,
        reference, reply, model_name, reference_type, history_context, expert_opinions,
        conversation_sysprompt=conversation_sysprompt,
        ground_truth=ground_truth,
    )

    last_err = None
    for attempt in range(retries):
        try:
            kwargs = {"model": eval_model, "messages": messages, "temperature": temperature}

            if hasattr(client, "chat_with_meta"):
                try:
                    resp = client.chat_with_meta(**kwargs)
                    if isinstance(resp, dict):
                        from ..analysis.rubric_dimension_analysis import normalize_eval_raw_text
                        eval_text = normalize_eval_raw_text(resp)
                    elif isinstance(resp, (tuple, list)) and len(resp) >= 1:
                        eval_text = resp[0]
                    else:
                        eval_text = str(resp)
                except Exception:
                    eval_text = client.chat(**kwargs)
            else:
                eval_text = client.chat(**kwargs)

            if eval_text and not str(eval_text).strip().startswith("<error"):
                from ..analysis.rubric_dimension_analysis import normalize_eval_raw_text
                eval_text = normalize_eval_raw_text(eval_text)
                return eval_text, ""
            if eval_text and str(eval_text).strip().startswith("<error"):
                last_err = eval_text

        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                err_str = str(e).strip()
                err_low = err_str.lower()
                # 限流 / 503 / 网关过载：延长等待再重试（aimux/routify 批量评测常见）
                if (
                    "资源限制" in err_str
                    or "限流" in err_str
                    or "rate limit" in err_low
                    or "quota" in err_low
                    or "503" in err_str
                    or "service temporarily unavailable" in err_low
                    or "temporarily unavailable" in err_low
                ):
                    wait = 30 + 20 * attempt  # 30s, 50s, 70s...
                    time.sleep(wait)
                else:
                    time.sleep(1.0 + (2 * attempt))

    if last_err is not None:
        err_s = str(last_err).strip()
        if err_s.startswith("<error"):
            return "", err_s
        return "", f"<error: {last_err}>"
    return "", "<error: judge API failed after retries>"


def _sort_replies_by_qid_model(df: pd.DataFrame) -> pd.DataFrame:
    """按题目 qid、再按 model 排序，便于同一题集中对比。qid 按数值排（1,2,...,10 而非 1,10,2）。"""
    if df.empty or 'qid' not in df.columns or 'model' not in df.columns:
        return df
    out = df.copy()
    out['_qid_num'] = pd.to_numeric(out['qid'], errors='coerce')
    out = out.sort_values(by=['_qid_num', 'model']).drop(columns=['_qid_num'])
    return out


def _load_preserved_sheets(output_excel: str) -> dict:
    """读取需保留的 sheet（非 Sheet1/replies、batch_log），评估时只更新主表与 batch_log。"""
    if not os.path.exists(output_excel):
        return {}
    try:
        xls = pd.ExcelFile(output_excel)
        preserved = {}
        for name in xls.sheet_names:
            if name in ('Sheet1', 'replies', 'batch_log', 'Batch_log'):
                continue
            preserved[name] = pd.read_excel(output_excel, sheet_name=name)
        return preserved
    except Exception:
        return {}


def _count_eval_scores_on_disk(output_excel: str, eval_col: str, main_sheet_name: str = 'Sheet1') -> Optional[int]:
    if not output_excel or not os.path.exists(output_excel) or not eval_col:
        return None
    try:
        df_disk = pd.read_excel(output_excel, sheet_name=main_sheet_name)
        if eval_col not in df_disk.columns:
            return 0
        return count_eval_numeric_scores(df_disk, eval_col)
    except Exception:
        return None


def save_results(df_replies: pd.DataFrame, df_batch_log: pd.DataFrame,
                 results: List[dict], batch_id: str, output_excel: str,
                 main_sheet_name: str = 'Sheet1', *, log_score_count: bool = True,
                 eval_column_for_verify: Optional[str] = None,
                 include_expert_stats: Optional[bool] = None) -> Tuple[bool, str]:
    """
    原子写回回复表；返回 (是否成功, 说明)。
    写后读盘统计 eval 列有效分数，避免「进度条在走、表里没分」。
    """
    path_abs = os.path.abspath(output_excel)
    eval_col = eval_column_for_verify or f"eval_{batch_id}"
    n_scored_mem = count_eval_numeric_scores(df_replies, eval_col) if eval_col in df_replies.columns else 0
    n_disk_before = _count_eval_scores_on_disk(output_excel, eval_col, main_sheet_name)

    if '出题人' not in df_replies.columns:
        df_replies = df_replies.copy()
        df_replies['出题人'] = ''

    def _write_workbook(target_path: str) -> bool:
        from ..analysis.data_loader import sanitize_replies_score_columns

        df_to_write = _sort_replies_by_qid_model(sanitize_replies_score_columns(df_replies))
        temp_path = target_path.replace('.xlsx', f'_temp_{int(time.time())}.xlsx')
        preserved = _load_preserved_sheets(output_excel) if target_path == output_excel else {}
        with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
            df_to_write.to_excel(writer, sheet_name=main_sheet_name, index=False)
            try:
                from ..analysis.inter_batch_consistency import write_batch_log_sheet_with_eval_stats
                write_batch_log_sheet_with_eval_stats(
                    writer, df_to_write, df_batch_log,
                    include_expert_stats=include_expert_stats,
                )
            except Exception as stats_err:
                print(f"⚠️  batch_log 统计块写入跳过: {stats_err}")
                if not df_batch_log.empty:
                    df_batch_log.to_excel(writer, sheet_name="batch_log", index=False)
            for name, frame in preserved.items():
                frame.to_excel(writer, sheet_name=name[:31], index=False)
        test_df = pd.read_excel(temp_path, sheet_name=main_sheet_name, nrows=5)
        if len(test_df) <= 0:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
        if os.path.exists(target_path):
            os.replace(temp_path, target_path)
        else:
            os.rename(temp_path, target_path)
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return True

    try:
        if not _write_workbook(output_excel):
            return False, f"保存验证失败（临时文件无数据）: {path_abs}"
    except Exception as e:
        print(f"⚠️  保存结果失败: {e}")
        try:
            if not _write_workbook(output_excel):
                return False, f"保存失败: {e}；兜底写入也失败: {path_abs}"
        except Exception as e2:
            return False, f"保存失败: {e}；兜底: {e2}"

    n_disk_after = _count_eval_scores_on_disk(output_excel, eval_col, main_sheet_name)
    if n_disk_after is None:
        return False, f"已写入但读盘校验失败，请关闭 Excel 后查看: {path_abs}"

    detail = (
        f"磁盘有效分数 {n_disk_before or 0}→{n_disk_after}（内存 {n_scored_mem}）| "
        f"{os.path.basename(path_abs)}"
    )
    if log_score_count:
        print(f"  💾 {detail}")

    # 主表保存成功后再写备份（便于 Excel 占用主文件时仍有一份）
    backup_path = output_excel.replace('.xlsx', f'_backup_eval_{batch_id}.xlsx')
    try:
        _write_workbook(backup_path)
    except Exception:
        pass

    if n_disk_before is not None and n_disk_after < n_disk_before:
        return False, f"磁盘分数变少（{detail}）"
    return True, detail


def _canonical_reply_model(
    model_name: str,
    existing_model_aliases: Optional[Dict[str, str]] = None,
) -> str:
    """与 stage3 补跑 skip 一致：历史导入名映射到 8 个逻辑模型名。"""
    from .stage3_reply import _canonical_logical_model_for_skip

    return _canonical_logical_model_for_skip(model_name, existing_model_aliases)


def _reply_indices_for_canonical(
    df_replies: pd.DataFrame,
    qid: str,
    canonical_model: str,
    existing_model_aliases: Optional[Dict[str, str]] = None,
) -> List[int]:
    """同一 qid 下所有映射到该逻辑模型的行（glm-5 与 glm-5.1 等）。"""
    out: List[int] = []
    for idx, row in df_replies.iterrows():
        if str(row.get("qid", "")).strip() != qid:
            continue
        if _canonical_reply_model(row.get("model", ""), existing_model_aliases) == canonical_model:
            out.append(idx)
    return out


def batch_evaluate_responses_with_cache(
        questions_excel: str,
        replies_excel: str,
        output_excel: str,
        provider: str,
        model: str,
        sysprompt_manager: SyspromptManager,
        batch_id: str = None,
        data_filters: dict = None,
        temperature: float = 0.3,
        max_workers: int = 5,
        checkpoint_interval: int = 10,
        timeout: int = 120,
        overwrite_mode: str = 'skip',
        inject_expert_opinions_in_evaluation: bool = True,
        skip_ref_in_evaluation: bool = False,
        trim_replies_to_question_qids: bool = True,
        questions_sheet: Optional[object] = None,
        existing_model_aliases: Optional[Dict[str, str]] = None,
        eval_deduplicate_by_logical_model: bool = True,
        eval_preflight_probe: bool = False,
        eval_abort_if_no_score_after: int = 10,
):
    print(f"\n{'=' * 60}")
    print(f"🚀 模块4: 批量评估回复（并发评估版）")
    print(f"{'=' * 60}\n")

    pc = get_provider(provider)
    client = OAIClient(
        base_url=pc.base_url, api_key=pc.api_key, protocol=pc.protocol,
        auth_header=pc.auth_header, auth_prefix=pc.auth_prefix,
        extra_headers=pc.extra_headers, timeout=timeout
    )
    print(f"✅ 客户端初始化成功\n")

    if skip_ref_in_evaluation:
        print("  📌 已启用 skip_ref_in_evaluation：不调用裁判评估 model=ref（仅评 reply1/reply2 等模型行）\n")

    provider_type = detect_provider_type(provider, model)
    print(f"📋 Provider类型: {provider_type}  并发数: {max_workers}  覆盖策略: {overwrite_mode}")
    if provider_type in ('claude', 'openai', 'gemini'):
        print(f"✅ 支持Prompt Caching，将启用缓存优化")
    print()

    sys_prompt = sysprompt_manager.get('reply_evaluation', '')
    if not sys_prompt:
        print("⚠️  未配置 reply_evaluation sysprompt")

    read_kw_q: Dict = {}
    if questions_sheet is not None and str(questions_sheet).strip() != "":
        read_kw_q["sheet_name"] = (
            int(questions_sheet)
            if str(questions_sheet).strip().isdigit()
            else questions_sheet
        )
    df_questions = pd.read_excel(questions_excel, **read_kw_q)
    if "evaluation_criteria" not in df_questions.columns and "rubrics" in df_questions.columns:
        df_questions["evaluation_criteria"] = df_questions["rubrics"]
    required_cols_q = ['qid', 'query', 'evaluation_criteria']
    missing_cols = [col for col in required_cols_q if col not in df_questions.columns]
    if missing_cols:
        raise ValueError(f"题目表缺少必需列: {', '.join(missing_cols)}")

    has_reference = 'reference' in df_questions.columns
    has_reference_type = 'reference_type' in df_questions.columns
    has_ground_truth = 'ground_truth' in df_questions.columns
    has_history_context = 'history_context' in df_questions.columns
    print(
        f"  题目数量: {len(df_questions)}  包含参考答案: {'是' if has_reference else '否'}  "
        f"ground_truth: {'是' if has_ground_truth else '否'}  多轮对话: {'是' if has_history_context else '否'}\n"
    )

    _empty_log = pd.DataFrame(columns=['batch_id', 'timestamp', 'total_tasks', 'completed',
                                       'failed', 'eval_model', 'temperature', 'max_workers'])
    # 主回复 sheet 名：新建文件或与读入文件一致（避免「文件不存在」分支未赋值）
    target_sheet = 'Sheet1'
    if not os.path.exists(replies_excel):
        print(f"  📌 回复表不存在，将新建空表并从题目表横表列（reply1..、参考等）注入后再评估: {replies_excel}\n")
        _rd = os.path.dirname(os.path.abspath(replies_excel))
        if _rd:
            os.makedirs(_rd, exist_ok=True)
        df_replies = pd.DataFrame(columns=['qid', 'model', 'reply'])
        df_batch_log = _empty_log
        sheet_names = ['Sheet1']
    else:
        try:
            xls = pd.ExcelFile(replies_excel)
            sheet_names = xls.sheet_names
            print(f"  发现sheet: {sheet_names}")

            target_sheet = next((s for s in ('Sheet1', 'replies') if s in sheet_names), sheet_names[0])
            df_replies = pd.read_excel(replies_excel, sheet_name=target_sheet)
            print(f"  从{target_sheet}读取回复表: {len(df_replies)} 行")

            batch_log_sheet = next((s for s in ('batch_log', 'batch_logs') if s in sheet_names), None)
            df_batch_log = (
                pd.read_excel(replies_excel, sheet_name=batch_log_sheet)
                if batch_log_sheet
                else _empty_log.copy()
            )
        except Exception as e:
            print(f"⚠️  读取Excel文件失败，尝试直接读取: {e}")
            df_replies = pd.read_excel(replies_excel)
            df_batch_log = _empty_log.copy()
            target_sheet = 'Sheet1'

    required_cols_r = ['qid', 'model', 'reply']
    missing_cols = [col for col in required_cols_r if col not in df_replies.columns]
    if missing_cols:
        raise ValueError(f"回复表缺少必需列: {', '.join(missing_cols)}")

    def _normalize_qid_for_merge(val):
        """统一 qid 格式便于关联：Excel 读成 400.0 的转为 \"400\"，与回复表的 \"1\" \"2\" 一致。"""
        s = str(val).strip()
        try:
            f = float(s)
            if not np.isnan(f) and f == int(f):
                return str(int(f))
        except (ValueError, TypeError):
            pass
        return s

    def _inject_anchor_replies_from_questions(df_replies: pd.DataFrame, df_questions: pd.DataFrame) -> pd.DataFrame:
        """
        从题目表横表列注入回复表：
          - reference → model='ref'（ground_truth 不注入为待评回复，仅随题目表进入裁判辅助信息）
          - reply1..replyN（或 回复1..N、回复一二…）→ model 为 replyN_model / 回复N模型；若未填模型名则用 reply_slot_N
          - reply*_score / 专家分 等 → 专家打分、专家理由
        已存在的 (qid, model) 行会更新；不存在则新增。
        """
        if 'qid' not in df_questions.columns:
            return df_replies

        df_replies = df_replies.copy()
        df_questions = df_questions.copy()
        df_replies['qid'] = df_replies['qid'].astype(str).str.strip().map(_normalize_qid_for_merge)
        df_replies['model'] = df_replies['model'].astype(str).str.strip()
        df_questions['qid'] = df_questions['qid'].astype(str).str.strip().map(_normalize_qid_for_merge)

        has_expert_col = '专家' in df_questions.columns

        def _coerce_score(val):
            """题目表分数转为 float；支持纯数字或「57 是否指出...」「62% (8/13)」等混排，取首个有效数字。"""
            if val is None or pd.isna(val):
                return None
            s = str(val).strip()
            if not s or s.lower() in ('nan', 'none', 'null', '-', '—'):
                return None
            try:
                f = float(s)
                return f if not np.isnan(f) else None
            except (ValueError, TypeError):
                pass
            # 混排时取首个数字（如 "57 是否指出..." -> 57；"62% (8/13)" -> 62）
            m = re.search(r'(\d+)\s*%?', s)
            if m:
                try:
                    return float(m.group(1))
                except (ValueError, TypeError):
                    pass
            return None

        def _upsert_row(qid, model_name, reply_text, score_val, reason_val, expert_name):
            nonlocal df_replies
            if model_name is None and reply_text is None:
                return
            if model_name is None or not str(model_name).strip():
                return
            model_name_str = str(model_name).strip()
            qid_str = str(qid).strip()
            # 题目表 reply 为空/NaN/占位字符串时，不向回复表生成行
            if reply_text is None:
                return
            if isinstance(reply_text, float) and np.isnan(reply_text):
                return
            reply_text_str = str(reply_text).strip()
            if not reply_text_str or reply_text_str.lower() in ('nan', 'none'):
                return
            score_val = _coerce_score(score_val)

            mask = (df_replies['qid'] == qid_str) & (df_replies['model'] == model_name_str)
            if mask.any():
                idx = df_replies.index[mask][0]
                if 'reply' in df_replies.columns:
                    df_replies.at[idx, 'reply'] = reply_text_str
                if score_val is not None and not (isinstance(score_val, float) and np.isnan(score_val)):
                    df_replies.at[idx, '专家打分'] = score_val
                if reason_val is not None and str(reason_val).strip():
                    df_replies.at[idx, '专家理由'] = str(reason_val)
                if expert_name and has_expert_col:
                    df_replies.at[idx, '专家'] = str(expert_name)
            else:
                row = {
                    'qid': qid_str,
                    'model': model_name_str,
                    'reply': reply_text_str,
                }
                if score_val is not None and not (isinstance(score_val, float) and np.isnan(score_val)):
                    row['专家打分'] = score_val
                if reason_val is not None and str(reason_val).strip():
                    row['专家理由'] = str(reason_val)
                if expert_name and has_expert_col:
                    row['专家'] = str(expert_name)
                df_replies = pd.concat([df_replies, pd.DataFrame([row])], ignore_index=True)

        for _, q in df_questions.iterrows():
            qid = q.get('qid')
            expert_name = q.get('专家') if has_expert_col else None

            # reference 视作 model='ref'；无题目表分数时默认专家认为参考回复 100 分；支持列名 reference/参考/参考答案
            ref_text_col = next((c for c in ['reference', 'reference_answer', '参考', '参考答案', '参考回复'] if c in df_questions.columns), None)
            if ref_text_col:
                ref_text = q.get(ref_text_col, None)
                if pd.notna(ref_text) and str(ref_text).strip():
                    ref_score = _coerce_score(
                        next((q.get(c) for c in ['ref_score', 'reference_score', '参考得分', 'ref分数', '参考分数'] if c in df_questions.columns and pd.notna(q.get(c))), None)
                    )
                    if ref_score is None:
                        ref_score = 100.0
                    ref_reason = next((q.get(c) for c in ['ref_reason', 'reference_reason', '参考理由'] if c in df_questions.columns and pd.notna(q.get(c)) and str(q.get(c)).strip()), None)
                    ref_reason = str(ref_reason).strip() if ref_reason is not None else None
                    _upsert_row(qid, 'ref', ref_text, ref_score, ref_reason, expert_name)

            # reply1..N 横表锚点（默认最多 WIDE_REPLY_ANCHOR_MAX 列）；无模型列时用 reply_slot_N 占位
            for idx_anchor in range(1, WIDE_REPLY_ANCHOR_MAX + 1):
                text_candidates = _wide_reply_text_column_candidates(idx_anchor)
                r_col = next((c for c in text_candidates if c in df_questions.columns), None)
                if r_col is None:
                    continue
                reply_text = q.get(r_col, None)
                if reply_text is None or (isinstance(reply_text, float) and np.isnan(reply_text)):
                    continue
                if not str(reply_text).strip() or str(reply_text).strip().lower() in ('nan', 'none'):
                    continue
                model_candidates = [f'reply{idx_anchor}_model', f'回复{idx_anchor}_model', f'回复{idx_anchor}模型']
                m_col = next((c for c in model_candidates if c in df_questions.columns), None)
                model_name = q.get(m_col, None) if m_col else None
                if model_name is None or (isinstance(model_name, float) and np.isnan(model_name)) or not str(model_name).strip():
                    model_name = f'reply_slot_{idx_anchor}'
                score_candidates = [f'reply{idx_anchor}_score', f'回复{idx_anchor}得分', f'评估{idx_anchor}', f'得分{idx_anchor}', f'评估结果{idx_anchor}']
                score_val = next((q.get(c) for c in score_candidates if c in df_questions.columns and pd.notna(q.get(c))), None)
                score_val = _coerce_score(score_val) if score_val is not None else None
                reason_col = next((c for c in [f'reply{idx_anchor}_reason', f'回复{idx_anchor}_reason'] if c in df_questions.columns), None)
                reason_val = q.get(reason_col, None) if reason_col else None
                _upsert_row(qid, model_name, reply_text, score_val, reason_val, expert_name)

        return df_replies

    df_replies['qid'] = df_replies['qid'].astype(str).str.strip().map(_normalize_qid_for_merge)

    # 从题目表把 reference / reply1 / reply2 及分数正确传到回复表（含专家打分与理由）；ref 默认满分 100
    df_replies = _inject_anchor_replies_from_questions(df_replies, df_questions)
    if len(df_replies) == 0:
        raise ValueError(
            "【evaluate_replies】从题目表注入后回复表仍为空。请至少满足其一："
            "① 在题目表中填写「参考/参考答案/reference」或「reply1、reply2…」等横表列（非空文本）；"
            "② 在 config 的 stages 中加入 generate_replies，先由模型生成回复并写入回复表。"
            "（仅含 qid/query/evaluation_criteria 而无任何待评回复时，不会生成 output/replies/*.xlsx。）"
        )
    ref_count = ((df_replies['model'].astype(str).str.strip().str.lower() == 'ref').sum() if 'model' in df_replies.columns else 0)
    if ref_count > 0:
        print(f"  📌 题目表 ref + 横表 reply1..{WIDE_REPLY_ANCHOR_MAX} 已注入回复表: ref 共 {ref_count} 条（默认专家打分=100），将参与裁判评估")

    # 出题人 = 题目表【专家】或【出题人】列，按 qid 写入回复表；统计阶段仅基于回复表即可
    df_questions['qid'] = df_questions['qid'].astype(str).str.strip().map(_normalize_qid_for_merge)
    expert_source_col = next((c for c in ['专家', '出题人'] if c in df_questions.columns), None)
    if expert_source_col:
        qid_to_expert = df_questions.drop_duplicates('qid').set_index('qid')[expert_source_col].astype(str)
        df_replies['出题人'] = df_replies['qid'].map(qid_to_expert)
        n_with = (df_replies['出题人'].notna() & (df_replies['出题人'].astype(str).str.strip() != '')).sum()
        if n_with > 0:
            print(f"  📌 已按 qid 将题目表【{expert_source_col}】写入回复表「出题人」列，共 {n_with} 条")
    else:
        df_replies['出题人'] = ''
        print(f"  ⚠️  题目表无「专家」或「出题人」列，已建空列，请补题目表后重跑以填充")

    # 传递核对：题目表 vs 回复表 (qid, 专家打分) 对应情况
    if '专家打分' in df_replies.columns:
        n_reply_score = df_replies['专家打分'].notna().sum()
        qids_with_score = df_replies.loc[df_replies['专家打分'].notna(), 'qid'].nunique()
        print(f"  📌 核对: 回复表中 专家打分 非空 {n_reply_score} 条，涉及 {qids_with_score} 题（统计将仅基于回复表 qid/reply/专家打分/出题人）")

    # 若第一个 sheet 无专家列，尝试从第二个 sheet 加载专家评估并合并（供裁判 prompt 使用）
    if '专家打分' not in df_replies.columns and '专家理由' not in df_replies.columns:
        for sh_name in sheet_names:
            if sh_name in ('Sheet1', 'replies', 'batch_log', 'Batch_log'):
                continue
            ex_df = pd.read_excel(replies_excel, sheet_name=sh_name)
            if 'qid' in ex_df.columns and 'model' in ex_df.columns and ('专家打分' in ex_df.columns or '专家理由' in ex_df.columns):
                ex_df = ex_df.copy()
                ex_df['qid'] = ex_df['qid'].astype(str).str.strip().map(_normalize_qid_for_merge)
                ex_df['model'] = ex_df['model'].astype(str).str.strip()
                ex_cols = [c for c in ['专家打分', '专家理由', '专家意见'] if c in ex_df.columns]
                ex_sub = ex_df[['qid', 'model'] + ex_cols].drop_duplicates(subset=['qid', 'model'], keep='first')
                df_replies = df_replies.merge(ex_sub, on=['qid', 'model'], how='left')
                print(f"  📌 从 sheet「{sh_name}」加载专家评估 {len(ex_sub)} 条，已合并")
                break

    df_replies['model'] = df_replies['model'].astype(str).str.strip()
    df_questions['qid'] = df_questions['qid'].astype(str).str.strip().map(_normalize_qid_for_merge)

    if 'sysprompt' in df_questions.columns:
        n_sp = df_questions['sysprompt'].apply(lambda v: bool(safe_str(v).strip())).sum()
        print(
            f"  📌 题目表 sysprompt：{n_sp}/{len(df_questions)} 行非空"
            f"（裁判 prompt 将含题目级设定 + query + rubrics；空行等同仅两参数）"
        )

    # 专家洞察来源优先级：题目表「专家洞察」列（LLM 归纳）> 回复表专家 sheet 原始汇总
    # 题目表专家洞察：由 summarize_expert_assessments 阶段生成，精炼后注入更有效
    expert_insight_from_questions: Dict[str, str] = {}
    if '专家洞察' in df_questions.columns:
        for _, r in df_questions.iterrows():
            qid = str(r['qid']).strip()
            insight = r.get('专家洞察', '')
            if pd.notna(insight) and str(insight).strip() and str(insight).lower() not in ('nan', 'none', ''):
                expert_insight_from_questions[qid] = str(insight).strip()[:2000]

    # 按题目汇总原始专家评估（当题目表无专家洞察时使用）
    raw_expert_per_qid: Dict[str, str] = {}
    has_expert_score = '专家打分' in df_replies.columns
    has_expert_reason = '专家理由' in df_replies.columns
    has_expert_opinion = '专家意见' in df_replies.columns
    if has_expert_score or has_expert_reason or has_expert_opinion:
        for qid, grp in df_replies.groupby('qid'):
            lines = []
            for _, r in grp.iterrows():
                score_val = r.get('专家打分') if has_expert_score else None
                reason_val = r.get('专家理由') if has_expert_reason else (r.get('专家意见') if has_expert_opinion else None)
                if pd.isna(score_val) and (pd.isna(reason_val) or not str(reason_val).strip()):
                    continue
                model_name_val = safe_str(r.get('model', ''))
                part = f"模型 {model_name_val}:"
                if score_val is not None and not (isinstance(score_val, float) and np.isnan(score_val)):
                    part += f" 得分 {score_val}"
                if reason_val is not None and str(reason_val).strip() and str(reason_val).lower() not in ('nan', 'none', ''):
                    part += f"; 专家意见: {safe_str(reason_val)[:500]}"
                lines.append(part)
            if lines:
                raw_expert_per_qid[str(qid).strip()] = "\n".join(lines)

    expert_opinions_per_qid: Dict[str, str] = {}
    for qid in set(expert_insight_from_questions) | set(raw_expert_per_qid):
        if qid in expert_insight_from_questions:
            expert_opinions_per_qid[qid] = expert_insight_from_questions[qid]
        else:
            expert_opinions_per_qid[qid] = raw_expert_per_qid.get(qid, '')
    # 以少博大：无本题专家时，用同 L2 类型的专家洞察作为参考（迁移视角）
    l2_to_insight: Dict[str, str] = {}
    if 'L2' in df_questions.columns and expert_opinions_per_qid:
        qid_to_l2 = dict(zip(df_questions['qid'].astype(str), df_questions['L2'].astype(str)))
        for qid, insight in expert_opinions_per_qid.items():
            l2 = qid_to_l2.get(qid, '').strip()
            if l2 and l2 not in l2_to_insight:
                l2_to_insight[l2] = f"【同类型（L2={l2}）题目专家洞察参考】\n{insight[:1500]}"
    if expert_opinions_per_qid:
        insight_count = sum(1 for q in expert_opinions_per_qid if q in expert_insight_from_questions)
        if insight_count:
            print(f"  📌 已加载 {len(expert_opinions_per_qid)} 题的专家参考（其中 {insight_count} 题为 LLM 归纳的专家洞察）")
        else:
            print(f"  📌 已加载 {len(expert_opinions_per_qid)} 题的专家评估示范，将作为裁判参考锚点")

    eval_columns = [col for col in df_replies.columns if isinstance(col, str) and col.startswith('eval_')]
    print(f"\n📊 回复表列名: {list(df_replies.columns)}")
    print(f"  发现评估列: {eval_columns}")

    # 统一 qid 格式；可选「以题目表为真源裁剪回复表」：
    # - trim_replies_to_question_qids=True（默认）：剔除题目表中不存在的 qid，写回文件时也不保留（适合题目表=全集）
    # - False：保留回复表里题目表未覆盖的 qid（适合题目表仅为子集/克隆子表，需与历史专家题共存）
    df_replies = df_replies.copy()
    df_replies['qid'] = df_replies['qid'].astype(str).str.strip().map(_normalize_qid_for_merge)
    df_questions_for_merge = df_questions.copy()
    df_questions_for_merge['qid'] = df_questions_for_merge['qid'].astype(str).str.strip().map(_normalize_qid_for_merge)
    valid_qids = set(df_questions_for_merge['qid'].astype(str).tolist())
    before_sync = len(df_replies)
    orphan_mask = ~df_replies['qid'].astype(str).isin(valid_qids)
    n_orphan = int(orphan_mask.sum())
    if trim_replies_to_question_qids:
        df_replies = df_replies[~orphan_mask].copy()
        removed_by_sync = before_sync - len(df_replies)
        if removed_by_sync > 0:
            raw_df = pd.read_excel(replies_excel, sheet_name=target_sheet)
            raw_df['qid'] = raw_df['qid'].astype(str).str.strip().map(_normalize_qid_for_merge)
            removed_df = raw_df[~raw_df['qid'].astype(str).isin(valid_qids)]
            qid_counts = removed_df.groupby('qid').size()
            qid_dict = {str(k): int(v) for k, v in qid_counts.items()}
            print(f"  🔄 已按题目表同步回复表: 移除 {removed_by_sync} 条无对应 qid 的历史回复")
            if qid_dict:
                print(f"  📌 被移除 qid 及条数: {qid_dict}")
    else:
        if n_orphan > 0:
            print(
                f"  📌 trim_replies_to_question_qids=false：保留题目表未覆盖的回复行 {n_orphan} 条（不删除、不参评）；"
                f"仅对题目表与回复表 qid 交集调用裁判。\n"
            )

    df_replies = df_replies.reset_index(drop=True)
    df_merged = df_replies.merge(df_questions_for_merge, on='qid', how='left')
    df_merged['_reply_pos'] = df_merged.index.astype(np.int64)
    for _mt_col in ('session_id', 'turn_id', 'history_context'):
        _coalesce_merge_xy_columns(df_merged, _mt_col, prefer='left')
    # 合并后若列名冲突会变成 query_x/query_y，优先用题目表的 query（右侧）作为题干
    if 'query' not in df_merged.columns and 'query_y' in df_merged.columns:
        df_merged['query'] = df_merged['query_y']
    elif 'query' not in df_merged.columns and 'query_x' in df_merged.columns:
        df_merged['query'] = df_merged['query_x']
    if 'sysprompt' not in df_merged.columns and 'sysprompt_y' in df_merged.columns:
        df_merged['sysprompt'] = df_merged['sysprompt_y']
    elif 'sysprompt' not in df_merged.columns and 'sysprompt_x' in df_merged.columns:
        df_merged['sysprompt'] = df_merged['sysprompt_x']
    if 'ground_truth' not in df_merged.columns and 'ground_truth_y' in df_merged.columns:
        df_merged['ground_truth'] = df_merged['ground_truth_y']
    elif 'ground_truth' not in df_merged.columns and 'ground_truth_x' in df_merged.columns:
        df_merged['ground_truth'] = df_merged['ground_truth_x']
    missing_questions = df_merged['query'].isna().sum() if 'query' in df_merged.columns else 0
    if missing_questions > 0:
        dropped = df_merged[df_merged['query'].isna()].copy()
        if 'qid' in dropped.columns and 'model' in dropped.columns:
            qid_counts = dropped.groupby('qid').size()
            qid_dict = {str(k): int(v) for k, v in qid_counts.items()}
            print(f"⚠️  警告: {missing_questions} 条回复找不到对应题目，将不参与评估（无法补齐这些行的分数）")
            print(f"  📌 涉及 qid 及条数: {qid_dict}")
            if trim_replies_to_question_qids:
                print("  💡 trim_replies_to_question_qids=true：这些行已在评估前从内存中的回复表剔除，写回文件时不含题目表外 qid。")
            else:
                print("  💡 trim_replies_to_question_qids=false：题目表外 qid 仍保留在回复表文件中，仅不进入本次裁判任务。")
        else:
            print(f"⚠️  警告: {missing_questions} 条回复找不到对应题目，将不参与评估")
        if missing_questions == len(df_merged) and len(df_merged) > 0:
            q_reply = df_replies['qid'].astype(str).drop_duplicates().head(5).tolist()
            q_quest = df_questions['qid'].astype(str).drop_duplicates().head(5).tolist()
            print(f"  📌 题目表 qid 样例（共 {len(df_questions)} 题）: {q_quest}")
            print(f"  📌 回复表 qid 样例（共 {df_replies['qid'].nunique()} 个不同 qid）: {q_reply}")
            print(f"  📌 请检查两表是否同一批题目、qid 格式是否一致（如 1 vs \"1\"、避免 Excel 把数字读成 1.0）")
        df_merged = df_merged[df_merged['query'].notna()]
    # generate_criteria 若因增量跳过未写入 evaluation_criteria，但题目表有 human_rubrics 时，评估仍按人工 rubrics
    if 'human_rubrics' in df_merged.columns:
        hr = df_merged['human_rubrics'].apply(safe_str)
        if 'evaluation_criteria' not in df_merged.columns:
            df_merged['evaluation_criteria'] = hr
        else:
            ec = df_merged['evaluation_criteria'].apply(safe_str)
            df_merged['evaluation_criteria'] = ec.where(ec.str.strip().ne(''), hr)
    if 'rubrics' in df_merged.columns:
        rb = df_merged['rubrics'].apply(safe_str)
        if 'evaluation_criteria' not in df_merged.columns:
            df_merged['evaluation_criteria'] = rb
        else:
            ec = df_merged['evaluation_criteria'].apply(safe_str)
            df_merged['evaluation_criteria'] = ec.where(ec.str.strip().ne(''), rb)
    print(f"  关联后参与评估的数据量: {len(df_merged)} 条\n")

    # 评测顺序：Ours 题对应的回复行排在前面（仍评全表，仅调整并发任务先后）
    _src_col = next((c for c in ('source_y', 'source', 'source_x') if c in df_merged.columns), None)
    if _src_col:
        df_merged = (
            df_merged.assign(_eval_ours_first=df_merged[_src_col].astype(str).str.strip().str.lower().ne('ours'))
            .sort_values('_eval_ours_first', kind='stable')
            .drop(columns='_eval_ours_first')
            .reset_index(drop=True)
        )

    if has_history_context and 'session_id' in df_merged.columns and 'turn_id' in df_merged.columns:
        print(f"  🔄 检测到多轮对话数据，按 session_id + turn_id 排序")
        def _multiturn_sort_key(s: pd.Series) -> pd.Series:
            if s.name == 'session_id':
                return s.fillna('').astype(str)
            return pd.to_numeric(s, errors='coerce').fillna(0)

        if '_reply_pos' not in df_merged.columns:
            df_merged['_reply_pos'] = df_merged.index.astype(np.int64)
        df_merged = df_merged.sort_values(
            by=['session_id', 'turn_id'],
            key=_multiturn_sort_key,
        ).reset_index(drop=True)

    if data_filters:
        original_count = len(df_merged)
        src_inc = data_filters.get('source_include')
        if src_inc:
            allow_src = {str(x).strip().lower() for x in src_inc if str(x).strip()}
            _src_f = next((c for c in ('source_y', 'source', 'source_x') if c in df_merged.columns), None)
            if _src_f:
                df_merged = df_merged[
                    df_merged[_src_f].astype(str).str.strip().str.lower().isin(allow_src)
                ]
                print(
                    f"  📌 按 source 筛选 {sorted(allow_src)}: {original_count} → {len(df_merged)} 条"
                )
                original_count = len(df_merged)
            else:
                print("  ⚠️  data_filters.source_include 已配置但关联表无 source 列，未筛选")
        # 若启用 expert_qids_only，仅重评有专家评估的题目（充分利用专家洞察，可换裁判模型再测一遍）
        if data_filters.get('expert_qids_only') and expert_opinions_per_qid:
            expert_qids = [str(q).strip() for q in expert_opinions_per_qid.keys()]
            df_merged = df_merged[df_merged['qid'].isin(expert_qids)]
            print(f"  📌 仅重评有专家评估的题目: {len(expert_qids)} 题 | 数据量 {original_count} → {len(df_merged)} 条")
            original_count = len(df_merged)
        if data_filters.get('qid_list'):
            qid_list = [str(q).strip() for q in data_filters['qid_list']]
            df_merged = df_merged[df_merged['qid'].isin(qid_list)]
            print(f"  📌 按QID筛选: {original_count} → {len(df_merged)} 条")
            original_count = len(df_merged)
        if data_filters.get('model_list'):
            model_list = [str(m).strip() for m in data_filters['model_list']]
            df_merged = df_merged[df_merged['model'].isin(model_list)]
            print(f"  📌 按模型筛选: {original_count} → {len(df_merged)} 条")
            original_count = len(df_merged)
        if data_filters.get('reference_type') and has_reference_type:
            df_merged = df_merged[df_merged['reference_type'].isin(data_filters['reference_type'])]
            print(f"  📌 按参考类型筛选: {original_count} → {len(df_merged)} 条")
            original_count = len(df_merged)
        if data_filters.get('batch_size') and len(df_merged) > data_filters['batch_size']:
            df_merged = df_merged.head(data_filters['batch_size'])
            print(f"  📌 限制批次大小: {original_count} → {len(df_merged)} 条")
            print(f"  ✅ 筛选后数据量: {len(df_merged)} 条\n")

    # 独立裁判：不向裁判 prompt 注入【专家参考】；回复表里 专家打分/理由 仍保留，供 analyze 人机对比
    if not inject_expert_opinions_in_evaluation:
        if expert_opinions_per_qid or l2_to_insight:
            print(
                "  📌 独立裁判模式：不向裁判 API 注入专家打分/理由/洞察（仅依赖题干 + rubrics ± ground_truth ± 参考答案）"
            )
        expert_opinions_per_qid = {}
        l2_to_insight = {}

    if len(df_merged) == 0:
        print(
            "⚠️  筛选后无数据需要处理（关联题目表后 0 行）。"
            "常见原因：题目表与回复表 qid 不一致；或 evaluation_criteria/human_rubrics 全空导致无法建评估任务。"
        )
        print("  💡 不会写入回复表文件；请修正题目/回复表后重跑 evaluate_replies。")
        return

    if not batch_id:
        batch_id = f"eval_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"📋 评估批次ID: {batch_id}")

    eval_column = f"eval_{batch_id}"
    eval_raw_column = f"{eval_column}_raw"

    # 补齐模式：若指定列不存在，尝试与已有评估列名对齐（如 batch_1 ↔ eval_batch1）
    if eval_column not in df_replies.columns and overwrite_mode == 'skip':
        eval_cols = [c for c in df_replies.columns if isinstance(c, str) and c.startswith('eval_') and not c.endswith('_raw')]
        alt = batch_id.replace('_', '')  # batch_1 -> batch1
        alt2 = batch_id.replace('batch', 'batch_') if 'batch' in batch_id else batch_id  # batch1 -> batch_1
        for cand in [f"eval_{alt}", f"eval_{alt2}"]:
            if cand != eval_column and cand in df_replies.columns:
                eval_column = cand
                eval_raw_column = f"{eval_column}_raw"
                print(f"  📌 补齐模式：将写入已有列 {eval_column}（与 batch_id 对应）")
                break

    if eval_column in df_replies.columns:
        existing_mask = df_replies[eval_column].apply(is_evaluated)
        existing_evaluated = int(existing_mask.sum())
        n_raw_err = 0
        if eval_raw_column in df_replies.columns:
            n_raw_err = int(
                df_replies[eval_raw_column].fillna('').astype(str).str.startswith('<error').sum()
            )
        print(
            f"📊 发现已有评估列: {eval_column}  已有分数: {existing_evaluated} 条"
            f"（raw 中 API 失败占位 {n_raw_err} 条，不算已评测，会重试）"
        )

        if existing_evaluated > 0:
            if overwrite_mode == 'overwrite':
                print(f"  ⚠️  overwrite_mode=overwrite，将覆盖所有已有评估结果")
                df_replies[eval_column] = np.nan
                if eval_raw_column in df_replies.columns:
                    df_replies[eval_raw_column] = ''
            elif overwrite_mode == 'new_batch':
                batch_id = f"eval_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                eval_column = f"eval_{batch_id}"
                eval_raw_column = f"{eval_column}_raw"
                print(f"  🆕 overwrite_mode=new_batch，新批次ID: {batch_id}")
            else:
                print(f"  ✅ overwrite_mode=skip，跳过已有评估，只评估空白数据")

    if eval_column not in df_replies.columns:
        df_replies[eval_column] = np.nan
        print(f"  📌 列 {eval_column} 不存在，已新建（将评估全部合并数据）")
    if eval_raw_column not in df_replies.columns:
        df_replies[eval_raw_column] = ''

    def _rescore_rows_with_raw_but_no_score() -> int:
        """已有 raw、无分数：用 ✅/❌ 本地重算，避免重复调 API。"""
        from ..analysis.rubric_dimension_analysis import score_eval_from_raw_text

        if eval_raw_column not in df_replies.columns or eval_column not in df_replies.columns:
            return 0
        fixed = 0
        for pos in range(len(df_replies)):
            if is_evaluated(_read_reply_cell(df_replies, pos, eval_column)):
                continue
            raw = _cell_scalar(_read_reply_cell(df_replies, pos, eval_raw_column)).strip()
            if not raw or raw.startswith("<error"):
                continue
            scored = score_eval_from_raw_text(raw)
            ts = scored.get("total_score")
            if ts is None or (isinstance(ts, float) and np.isnan(ts)):
                continue
            norm = scored.get("normalized_raw") or raw
            _write_reply_cell(df_replies, pos, eval_column, float(ts))
            _write_reply_cell(df_replies, pos, eval_raw_column, norm)
            fixed += 1
        return fixed

    n_rescored = _rescore_rows_with_raw_but_no_score()
    if n_rescored:
        print(f"  🔁 从已有 raw 本地补分: {n_rescored} 条（✅/❌ 通过率，无需重调 API）")

    df_merged = df_merged.copy()
    if '_reply_pos' not in df_merged.columns:
        df_merged['_reply_pos'] = df_merged.index.astype(np.int64)
    df_merged_reset = df_merged.reset_index(drop=True)
    n_replies = len(df_replies)
    already_evaluated = 0
    tasks = []

    def _row_has_ok_reply(pos: int) -> bool:
        if pos < 0 or pos >= n_replies:
            return False
        rep = safe_str(df_replies.iloc[pos]['reply']).strip()
        if not rep or rep.startswith('<error'):
            return False
        st = safe_str(df_replies.iloc[pos].get('status', '')).strip().lower()
        return not st or st == 'ok'

    def _eval_score_at(pos: int):
        if pos < 0 or pos >= n_replies:
            return np.nan
        return df_replies.iloc[pos][eval_column]

    def _criteria_from_row(row) -> str:
        criteria_str = safe_str(row.get('evaluation_criteria', ''))
        if not (criteria_str and criteria_str.strip()):
            criteria_str = safe_str(row.get('human_rubrics', ''))
        if not (criteria_str and criteria_str.strip()):
            criteria_str = safe_str(row.get('rubrics', ''))
        return criteria_str if (criteria_str and criteria_str.strip()) else ''

    def _append_task(i: int, row, reply_idx: int, reply_indices: List[int], model_name: str, reply_text: str, criteria_str: str):
        qid = str(row['qid']).strip()
        history_context_value = ''
        if has_history_context and 'history_context' in row.index:
            raw_hc = row['history_context']
            if not pd.isna(raw_hc):
                history_context_value = str(raw_hc)
        expert_opinions_text = expert_opinions_per_qid.get(qid, '')
        if not expert_opinions_text and l2_to_insight and 'L2' in row.index:
            l2_val = str(row.get('L2', '')).strip()
            expert_opinions_text = l2_to_insight.get(l2_val, '')
        conv_sp = safe_str(row.get('sysprompt', '')) if 'sysprompt' in row.index else ''
        tasks.append({
            'merged_index': i, 'qid': qid, 'model': model_name,
            'reply_indices': reply_indices,
            'query': safe_str(row['query']),
            'evaluation_criteria': criteria_str,
            'reference': safe_str(row['reference']) if has_reference else '',
            'reference_type': safe_str(row['reference_type']) if has_reference_type else 'model',
            'reply': reply_text,
            'history_context': history_context_value,
            'expert_opinions': expert_opinions_text,
            'conversation_sysprompt': conv_sp,
            'ground_truth': safe_str(row.get('ground_truth', '')) if 'ground_truth' in row.index else '',
            'batch_id': batch_id, 'eval_column': eval_column,
            'eval_raw_column': eval_raw_column, 'reply_idx': reply_idx,
        })

    if eval_deduplicate_by_logical_model:
        seen_qm = set()
        if existing_model_aliases:
            print(
                f"  ℹ️  评测按逻辑模型去重（历史别名 {len(existing_model_aliases)} 条）\n"
            )
        for i, row in df_merged_reset.iterrows():
            qid = str(row['qid']).strip()
            model_name = str(row['model']).strip()
            logical_model = _canonical_reply_model(model_name, existing_model_aliases)
            if skip_ref_in_evaluation and model_name.lower() == 'ref':
                continue
            if (qid, logical_model) in seen_qm:
                continue
            reply_indices = _reply_indices_for_canonical(
                df_replies, qid, logical_model, existing_model_aliases
            )
            if not reply_indices:
                continue
            if any(is_evaluated(_eval_score_at(ri)) for ri in reply_indices):
                already_evaluated += 1
                seen_qm.add((qid, logical_model))
                continue
            eval_indices = [ri for ri in reply_indices if _row_has_ok_reply(ri)]
            if not eval_indices:
                continue
            reply_idx = eval_indices[0]
            if len(eval_indices) > 1:
                reply_idx = max(
                    eval_indices,
                    key=lambda ri: (
                        1 if safe_str(df_replies.iloc[ri]['model']).strip() == logical_model else 0,
                        len(safe_str(df_replies.iloc[ri]['reply'])),
                    ),
                )
            criteria_str = _criteria_from_row(row)
            if not criteria_str:
                continue
            seen_qm.add((qid, logical_model))
            _append_task(
                i, row, reply_idx, reply_indices, model_name,
                safe_str(df_replies.iloc[reply_idx]['reply']).strip(), criteria_str,
            )
        total_merged = len(seen_qm)
        print(
            f"📊 合并去重后: {total_merged} 条；将写入/补齐列: {eval_column}；"
            f"已有分数: {already_evaluated} 条，本次将评估: {len(tasks)} 条"
        )
    else:
        print("  📌 按回复表逐行评测（不做逻辑模型去重）\n")
        eligible = 0
        for i, row in df_merged_reset.iterrows():
            reply_pos = int(row['_reply_pos'])
            model_name = str(row['model']).strip()
            if skip_ref_in_evaluation and model_name.lower() == 'ref':
                continue
            if not _row_has_ok_reply(reply_pos):
                continue
            eligible += 1
            if is_evaluated(_eval_score_at(reply_pos)):
                already_evaluated += 1
                continue
            criteria_str = _criteria_from_row(row)
            if not criteria_str:
                continue
            _append_task(
                i, row, reply_pos, [reply_pos], model_name,
                safe_str(df_replies.iloc[reply_pos]['reply']).strip(), criteria_str,
            )
        print(
            f"📊 回复表有效行: {eligible} 条；列 {eval_column} 已有分数: {already_evaluated} 条；"
            f"本次将评估: {len(tasks)} 条"
        )
        total_merged = eligible
    if not tasks and total_merged > 0:
        if already_evaluated >= total_merged:
            print(f"  💡 所有 {total_merged} 条在列「{eval_column}」中已有分数，故本次 0 条待评估。若你刚补充了新回复行：请确认该列对应行是否被误填（如 0、空格），可清空该两行的分数后重跑或设 overwrite_mode='overwrite' 重评。")
        else:
            print(f"  💡 若预期有空白需补齐：请确认 CONFIG 中 batch_id 与回复表列名一致（列名为 eval_{{batch_id}}，如 batch_1 → eval_batch_1）；或检查上述「因无对应题目而丢弃」的回复是否包含待补齐行。")

    if not tasks:
        print("✅ 所有任务已完成（无空白需评估）")
        if '出题人' in df_replies.columns:
            print(f"  📌 保存回复表（含出题人列）至 {output_excel} 的「{target_sheet}」sheet")
        save_results(
            df_replies, df_batch_log, [], batch_id, output_excel,
            main_sheet_name=target_sheet, eval_column_for_verify=eval_column,
            include_expert_stats=(
                None if inject_expert_opinions_in_evaluation else False
            ),
        )
        return

    def evaluate_task(task: dict) -> dict:
        try:
            row = df_merged_reset.iloc[task['merged_index']]
            # 评估 model='ref' 时 deliberately 不提供 reference，避免裁判偷懒直接对比给满分
            # 裁判须仅按 rubrics 评估，reference 满分为 rubrics 自洽性的真实检验
            is_ref_model = str(task.get('model', '')).strip().lower() == 'ref'
            ref_for_prompt = '' if is_ref_model else (safe_str(row['reference']) if has_reference else '')
            gt_for_prompt = task.get('ground_truth', '') or (
                safe_str(row.get('ground_truth', '')) if 'ground_truth' in row.index else ''
            )
            raw_eval, error_msg = evaluate_single_reply_with_cache(
                client, model, provider_type, sys_prompt,
                safe_str(row['query']), safe_str(row['evaluation_criteria']),
                ref_for_prompt,
                safe_str(row['reply']), task['model'],
                safe_str(row['reference_type']) if has_reference_type else 'model',
                temperature, retries=5,
                history_context=task.get('history_context', ''),
                expert_opinions=task.get('expert_opinions', ''),
                conversation_sysprompt=task.get('conversation_sysprompt', ''),
                ground_truth=gt_for_prompt,
            )

            if error_msg or (raw_eval and str(raw_eval).strip().startswith('<error')):
                err = error_msg or str(raw_eval).strip()
                return {'qid': task['qid'], 'model': task['model'], 'batch_id': task['batch_id'],
                        'raw_evaluation': err, 'scores': {}, 'total_score': np.nan,
                        'error': err, 'status': 'error', 'reply_idx': task['reply_idx'],
                        'reply_indices': task.get('reply_indices'),
                        'eval_column': task['eval_column'], 'eval_raw_column': task['eval_raw_column']}

            # 总分 = 通过考点数/总考点数（✅/❌）；先 unwrap API dict，再结构化/符号块解析
            from ..analysis.rubric_dimension_analysis import score_eval_from_raw_text

            scored = score_eval_from_raw_text(raw_eval)
            raw_eval = scored.get("normalized_raw") or raw_eval
            total_score = scored.get("total_score", np.nan)
            scores = extract_scores_from_evaluation(raw_eval)
            if total_score is not None and not (isinstance(total_score, float) and np.isnan(total_score)):
                scores["total_score"] = float(total_score)
            st = 'ok'
            err_out = ''
            if total_score is None or (isinstance(total_score, float) and np.isnan(total_score)):
                st = 'error'
                err_out = '裁判输出已返回但无法解析考点通过率（请检查 reply_evaluation 编号+✅/❌ 格式）'
            return {'qid': task['qid'], 'model': task['model'], 'batch_id': task['batch_id'],
                    'raw_evaluation': raw_eval, 'scores': scores,
                    'total_score': total_score,
                    'error': err_out, 'status': st, 'reply_idx': task['reply_idx'],
                    'reply_indices': task.get('reply_indices'),
                    'eval_column': task['eval_column'], 'eval_raw_column': task['eval_raw_column']}

        except Exception as e:
            return {'qid': task['qid'], 'model': task['model'], 'batch_id': task['batch_id'],
                    'raw_evaluation': f"<error: {str(e)}>", 'scores': {}, 'total_score': np.nan,
                    'error': str(e), 'status': 'error', 'reply_idx': task['reply_idx'],
                    'reply_indices': task.get('reply_indices'),
                    'eval_column': task['eval_column'], 'eval_raw_column': task['eval_raw_column']}

    eval_results = []
    failed_count = 0
    scored_count = 0
    api_error_count = 0
    parse_fail_count = 0
    completed_count = 0
    batch_tasks = list(tasks)
    total_tasks = len(tasks)
    df_write_lock = threading.Lock()
    scored_before = count_eval_numeric_scores(df_replies, eval_column)

    abort_after = max(1, int(eval_abort_if_no_score_after or 10))
    if eval_preflight_probe and tasks:
        print(f"\n🔍 评测前探活（真实 rubrics 请求，裁判 {provider}/{model}）...")
        probe_result = evaluate_task(tasks[0])
        eval_results.append(probe_result)
        probe_outcome = _apply_eval_result_to_df(df_replies, probe_result, df_write_lock)
        if probe_outcome != 'scored':
            err = probe_result.get('error') or probe_result.get('raw_evaluation') or 'unknown'
            err_s = str(err).strip()[:500]
            raise RuntimeError(
                f"裁判探活失败，已中止批量评测（避免空跑 {len(tasks)} 条）。\n"
                f"  渠道: {provider} / {model}\n"
                f"  错误: {err_s}\n"
                f"  💡 请重新运行并在 test_judge_models 阶段选择 idealab 或 routify_* 渠道"
                f"（列表靠前）；aimux 若 503 请勿选用。"
            )
        scored_count = 1
        completed_count = 1
        batch_tasks = tasks[1:]
        print(f"  ✅ 探活成功（qid={tasks[0].get('qid')}），开始批量评测（剩余 {len(batch_tasks)} 条）\n")

    print(f"\n⚡ 开始并发评估 ({max_workers} workers)...")
    print(f"   裁判: {provider}/{model}")
    print(
        f"   结果写入: {output_excel}  每 {checkpoint_interval} 条保存一次；"
        f"请以列「{eval_column}」中的数字为准（不是 raw 列、也不是进度条条数）。\n"
    )
    print(f"   表中已有有效分数: {scored_before} 行（列 {eval_column}）")
    print(f"   若连续 {abort_after} 条仍无分数将自动中止\n")

    aborted_early = False
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(evaluate_task, task): task for task in batch_tasks
        }

        for future in tqdm(
            as_completed(future_to_task),
            total=len(batch_tasks),
            desc="🔄 评估进度",
            ncols=100,
        ):
            task = future_to_task[future]
            try:
                result = future.result()
                eval_results.append(result)

                outcome = _apply_eval_result_to_df(df_replies, result, df_write_lock)
                if outcome == 'scored':
                    scored_count += 1
                elif outcome == 'api_error':
                    api_error_count += 1
                    failed_count += 1
                elif outcome == 'parse_fail':
                    parse_fail_count += 1
                    failed_count += 1
                elif result.get('status') == 'error':
                    failed_count += 1

                completed_count += 1

                if completed_count % checkpoint_interval == 0:
                    saved, save_msg = save_results(
                        df_replies, df_batch_log, eval_results, batch_id, output_excel,
                        main_sheet_name=target_sheet, log_score_count=False,
                        eval_column_for_verify=eval_column,
                        include_expert_stats=(
                            None if inject_expert_opinions_in_evaluation else False
                        ),
                    )
                    if saved:
                        tqdm.write(
                            f"💾 检查点 {completed_count}/{total_tasks} | "
                            f"本次成功打分 {scored_count} | API失败 {api_error_count} | "
                            f"解析失败 {parse_fail_count} | {save_msg}"
                        )
                    else:
                        tqdm.write(
                            f"❌ 检查点保存失败: {save_msg}（请关闭 Excel 后重试；"
                            f"已处理 {completed_count} 条，本次成功打分 {scored_count}）"
                        )
                    if (
                        scored_count == 0
                        and completed_count >= abort_after
                        and api_error_count >= abort_after
                    ):
                        aborted_early = True
                        tqdm.write(
                            f"\n❌ 已连续处理 {completed_count} 条仍无有效分数（API失败 {api_error_count}），"
                            f"中止剩余 {total_tasks - completed_count} 条，避免空跑。\n"
                            f"   请换 idealab/routify 裁判或检查 {provider}/{model} 是否 503。\n"
                        )
                        for pending in future_to_task:
                            pending.cancel()
                        break

            except Exception as e:
                tqdm.write(f"❌ 任务处理异常 [{task['qid']}][{task['model']}]: {e}")
                failed_count += 1

    if aborted_early:
        raise RuntimeError(
            f"批量评测已中止：前 {completed_count} 条均无有效分数（裁判 {provider}/{model}）。"
            f"请重新 test_judge_models 并选择非 aimux 或可用的 routify/idealab 渠道。"
        )

    scored_after = count_eval_numeric_scores(df_replies, eval_column)
    batch_log_entry = {
        'batch_id': batch_id, 'timestamp': pd.Timestamp.now(),
        'total_tasks': total_tasks, 'completed': scored_count,
        'failed': failed_count, 'api_errors': api_error_count, 'parse_failures': parse_fail_count,
        'scored_in_table': scored_after, 'eval_model': f"{provider}/{model}",
        'temperature': temperature, 'max_workers': max_workers,
        'success_rate': scored_count / total_tasks if total_tasks > 0 else 0
    }
    df_batch_log = pd.concat([df_batch_log, pd.DataFrame([batch_log_entry])], ignore_index=True)

    try:
        ok, save_msg = save_results(
            df_replies, df_batch_log, eval_results, batch_id, output_excel,
            main_sheet_name=target_sheet, eval_column_for_verify=eval_column,
            include_expert_stats=(
                None if inject_expert_opinions_in_evaluation else False
            ),
        )
        if ok:
            print(f"  📌 回复表已写入「{target_sheet}」sheet（含出题人列），统计阶段将仅基于回复表")
            print(f"📊 批次日志已更新: {os.path.abspath(output_excel)}")
            print(f"  ✅ {save_msg}")
        else:
            print(f"❌ 最终保存未通过校验: {save_msg}")
    except Exception as e:
        print(f"⚠️  最终保存失败: {e}")
        backup_path = output_excel.replace('.xlsx', f'_backup_{batch_id}.xlsx')
        try:
            ok_b, msg_b = save_results(
                df_replies, df_batch_log, eval_results, batch_id, backup_path,
                main_sheet_name=target_sheet, eval_column_for_verify=eval_column,
                include_expert_stats=(
                    None if inject_expert_opinions_in_evaluation else False
                ),
            )
            if ok_b:
                print(f"💾 已保存到备份文件: {os.path.abspath(backup_path)}（{msg_b}）")
            else:
                print(f"❌ 备份保存也失败: {msg_b}")
        except Exception as e2:
            print(f"❌ 备份保存也失败: {e2}")

    print(f"\n{'=' * 60}")
    print(f"✅ 评估完成!  总任务: {total_tasks}")
    print(f"   本次成功写入分数: {scored_count}  |  API/网络失败: {api_error_count}  |  解析失败: {parse_fail_count}")
    print(f"   列「{eval_column}」有效分数: {scored_before} → {scored_after} 行（以保存后表内为准）")
    if scored_count == 0 and total_tasks > 0:
        print(f"   ⚠️  本次无任何有效分数写入磁盘。进度条「已处理」≠「已打分」；请检查裁判 API（provider/model）或 raw 列中的 <error>。")

    successful_results = [r for r in eval_results if _eval_result_has_valid_score(r)]
    if successful_results:
        scores = [r['total_score'] for r in successful_results]
        print(f"  平均分: {np.mean(scores):.2f}  最高: {np.max(scores):.2f}  最低: {np.min(scores):.2f}")

    print(f"  评估结果列: {eval_column}")
    print(f"{'=' * 60}\n")
