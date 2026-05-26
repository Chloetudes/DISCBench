# -*- coding: utf-8 -*-
import json
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import get_provider, get_provider_for_model, aimux_provider_for_origin
from clients.openai_client import OAIClient
from ..core.cache_messages import merge_conversation_sysprompt
from ..core.utils import safe_str, safe_save_excel, reply_vendor_family, eval_benchmark_eight_family
from ..core.blacklist import (
    MODEL_BLACKLIST,
    is_permission_error,
    is_connection_error,
    is_transient_gateway_error,
)
from ..models_from_excel import SOURCE_MODELS_EXCEL_KEY, mark_single_model_unavailable
from ..reply_model_routes import (
    logical_name_from_config,
    resolve_provider_config_for_route,
)


def _resolve_models_excel_for_mark(
    task_or_result: dict,
    models_excel_path: Optional[str],
    models_excel_paths: Optional[List[str]],
) -> Optional[str]:
    """失败写回表格时：优先用任务/结果上的来源表；否则单表路径回退。"""
    src = task_or_result.get(SOURCE_MODELS_EXCEL_KEY)
    if src is not None and str(src).strip():
        p = str(src).strip()
        if os.path.exists(p):
            return p
    paths = [str(p).strip() for p in (models_excel_paths or []) if p and str(p).strip()]
    existing = [p for p in paths if os.path.exists(p)]
    if len(existing) == 1:
        return existing[0]
    if models_excel_path and str(models_excel_path).strip():
        p = str(models_excel_path).strip()
        if os.path.exists(p):
            return p
    return None


_GLOBAL_COOLDOWN_LOCK = Lock()
_GLOBAL_COOLDOWN_UNTIL_TS = 0.0


def _all_models_in_taskset_blacklisted(model_keys: set) -> bool:
    """本轮任务里出现过的 (provider, model) 是否已全部进入本运行黑名单。"""
    if not model_keys:
        return False
    return all(MODEL_BLACKLIST.is_blacklisted(p, m) for p, m in model_keys)


def _executor_shutdown(executor: ThreadPoolExecutor, futures_map: dict, cancel_pending: bool) -> None:
    """退出线程池。cancel_pending=True 时取消尚未执行的任务（避免单模型拉黑后空跑上千次进度）。"""
    if sys.version_info >= (3, 9):
        executor.shutdown(wait=True, cancel_futures=cancel_pending)
    else:
        if cancel_pending:
            for f in futures_map:
                f.cancel()
        executor.shutdown(wait=True)


def _filter_questions_by_source_include(
    df: pd.DataFrame,
    source_include: Optional[List[str]],
) -> pd.DataFrame:
    """仅保留 source（忽略大小写）在 source_include 中的题目。"""
    if df is None or df.empty or not source_include:
        return df
    if "source" not in df.columns:
        print("  ⚠️  reply_source_include 已配置但题目表无 source 列，未筛选")
        return df
    allow = {str(x).strip().lower() for x in source_include if str(x).strip()}
    if not allow:
        return df
    out = df.copy()
    src_norm = out["source"].astype(str).str.strip().str.lower()
    mask = src_norm.isin(allow)
    filtered = out.loc[mask].reset_index(drop=True)
    print(
        f"  题目筛选 source∈{sorted(allow)}：{len(filtered)} / {len(out)} 题"
    )
    return filtered


def _build_qid_source_map(df_questions: pd.DataFrame) -> Dict[str, str]:
    """题目 qid → source（bench 标记），用于写入/回填回复表末尾 source 列。"""
    if df_questions is None or df_questions.empty or "source" not in df_questions.columns:
        return {}
    out: Dict[str, str] = {}
    for _, row in df_questions.iterrows():
        qid = safe_str(row.get("qid", "")).strip()
        src = safe_str(row.get("source", "")).strip()
        if qid and src:
            out[qid] = src
    return out


def _question_source_from_row(row) -> str:
    if row is None:
        return ""
    return safe_str(row.get("source", "")).strip() if hasattr(row, "get") else ""


def _prepare_replies_df_for_save(
    df: pd.DataFrame,
    qid_source_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """回填缺失的 source，并将 source 列置于表末。"""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "source" not in out.columns:
        out["source"] = ""
    if qid_source_map:
        for idx, row in out.iterrows():
            cur = safe_str(row.get("source", "")).strip()
            if cur:
                continue
            qid = safe_str(row.get("qid", "")).strip()
            if qid and qid in qid_source_map:
                out.at[idx, "source"] = qid_source_map[qid]
    cols = [c for c in out.columns if c != "source"]
    if "source" in out.columns:
        cols.append("source")
    return out[cols]


def _eval_cell_has_score(value) -> bool:
    if pd.isna(value):
        return False
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        s = str(value).strip().lower()
        return s not in ("", "nan", "none") and not s.startswith("<error")


def _pick_preserved_eval_value(rows: List[pd.Series], col: str):
    """从同 (qid, 逻辑模型) 的多行旧记录里择优保留评测单元格。"""
    if not rows:
        return np.nan
    is_raw = col.endswith("_raw")
    if is_raw:
        for row in rows:
            v = row.get(col)
            if pd.notna(v) and str(v).strip() and not str(v).strip().startswith("<error"):
                return v
        for row in rows:
            v = row.get(col)
            if pd.notna(v) and str(v).strip():
                return v
        return np.nan
    for row in rows:
        v = row.get(col)
        if _eval_cell_has_score(v):
            return v
    return np.nan


def _reply_upsert_key(
    row_like,
    existing_model_aliases: Optional[Dict[str, str]] = None,
) -> Tuple[str, str, str]:
    qid = safe_str(row_like.get("qid", "")).strip()
    model = _canonical_logical_model_for_skip(
        row_like.get("model", ""), existing_model_aliases,
    )
    source = safe_str(row_like.get("source", "")).strip().lower()
    return qid, model, source


def _reply_row_merge_rank(row_like) -> int:
    """写回合并时优先级：成功 > 有正文非 error > 其它。"""
    if _reply_row_is_success(row_like):
        return 3
    reply = safe_str(row_like.get("reply", "")).strip()
    if reply and not reply.startswith("<error"):
        return 2
    return 1


def _upsert_replies_with_disk(
    df_memory: pd.DataFrame,
    output_excel: str,
    existing_model_aliases: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    写回前与磁盘已有表按 (qid, 逻辑模型, source) 合并，避免：
    - 启动时读表失败导致 results=[] 时 checkpoint 覆盖清空全表；
    - 同题 duplicate 行（大小写不同的 model 名）。
    后写入且回复质量更高者覆盖旧行。
    """
    if df_memory is None:
        df_memory = pd.DataFrame()
    merged: Dict[Tuple[str, str, str], dict] = {}

    def _ingest(frame: pd.DataFrame) -> None:
        if frame is None or frame.empty:
            return
        for _, row in frame.iterrows():
            key = _reply_upsert_key(row, existing_model_aliases)
            if not key[0] or not key[1]:
                continue
            cur = row.to_dict()
            prev = merged.get(key)
            if prev is None or _reply_row_merge_rank(cur) >= _reply_row_merge_rank(prev):
                merged[key] = cur

    if output_excel and os.path.exists(output_excel):
        try:
            on_disk = pd.read_excel(output_excel)
            n_before = len(on_disk)
            _ingest(on_disk)
            n_disk_keys = len(merged)
        except Exception as e:
            print(f"  ⚠️  写回前读取已有回复表失败，仅用内存数据保存: {e}")
            n_before = 0
            n_disk_keys = 0
    else:
        n_before = 0
        n_disk_keys = 0

    _ingest(df_memory)
    if not merged:
        return df_memory
    out = pd.DataFrame(list(merged.values()))
    if n_before and len(out) < n_before * 0.5:
        print(
            f"  ⚠️  合并后行数 {len(out)} 远少于磁盘 {n_before}，请确认未误覆盖；"
            f"若 Excel 正打开该文件请先关闭后重试"
        )
    elif n_disk_keys and len(out) < n_disk_keys:
        print(f"  ℹ️  写回合并：磁盘 {n_before} 行 → 去重合并后 {len(out)} 行")
    return out


def _merge_preserve_eval_columns(
    df: pd.DataFrame,
    output_excel: str,
    existing_model_aliases: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """写回回复表时保留已有 eval_* 列（按 qid + 逻辑模型名，含 glm-5→glm-5.1 等别名）。"""
    if df is None or df.empty or not output_excel or not os.path.exists(output_excel):
        return df
    try:
        old = pd.read_excel(output_excel)
    except Exception:
        return df
    eval_cols = [c for c in old.columns if isinstance(c, str) and c.startswith("eval_")]
    if not eval_cols or "qid" not in df.columns or "model" not in df.columns:
        return df
    if "qid" not in old.columns or "model" not in old.columns:
        return df

    def _logic_key(qid: str, model: str) -> Tuple[str, str]:
        q = safe_str(qid).strip()
        m = _canonical_logical_model_for_skip(model, existing_model_aliases)
        return q, m

    old = old.copy()
    old["_logic_key"] = old.apply(lambda r: _logic_key(r["qid"], r["model"]), axis=1)
    old_by_key: Dict[Tuple[str, str], List[pd.Series]] = {}
    for _, row in old.iterrows():
        old_by_key.setdefault(row["_logic_key"], []).append(row)

    out = df.copy()
    out["qid"] = out["qid"].astype(str).str.strip()
    out["model"] = out["model"].astype(str).str.strip()

    for c in eval_cols:
        if c not in out.columns:
            out[c] = np.nan
        is_raw = c.endswith("_raw")
        for idx, row in out.iterrows():
            lk = _logic_key(row["qid"], row["model"])
            old_rows = old_by_key.get(lk, [])
            if not old_rows:
                continue
            cur = row.get(c)
            cur_empty = pd.isna(cur) or str(cur).strip() in ("", "nan", "None")
            if is_raw:
                need = cur_empty or str(cur).strip().startswith("<error")
            else:
                need = not _eval_cell_has_score(cur)
            if need:
                picked = _pick_preserved_eval_value(old_rows, c)
                if is_raw:
                    if pd.notna(picked) and str(picked).strip():
                        out.at[idx, c] = picked
                elif _eval_cell_has_score(picked):
                    out.at[idx, c] = picked
    return out


def _count_excel_rows(path: str) -> Optional[int]:
    if not path or not os.path.exists(path):
        return None
    try:
        return len(pd.read_excel(path))
    except Exception:
        return None


def _save_replies_df(
    df: pd.DataFrame,
    output_excel: str,
    qid_source_map: Optional[Dict[str, str]] = None,
    existing_model_aliases: Optional[Dict[str, str]] = None,
) -> Tuple[bool, str]:
    """
    写回 Excel；返回 (是否成功, 说明)。
    写后会读盘校验行数，避免「日志显示已保存但 Excel/磁盘未变」。
    """
    n_disk_before = _count_excel_rows(output_excel)
    prepared = _prepare_replies_df_for_save(df, qid_source_map)
    n_mem = len(prepared) if prepared is not None else 0
    prepared = _upsert_replies_with_disk(prepared, output_excel, existing_model_aliases)
    n_merged = len(prepared) if prepared is not None else 0
    prepared = _merge_preserve_eval_columns(prepared, output_excel, existing_model_aliases)
    ok = safe_save_excel(prepared, output_excel)
    n_disk_after = _count_excel_rows(output_excel)
    path_abs = os.path.abspath(output_excel)
    if not ok:
        return False, (
            f"保存失败（请关闭 Excel 中的文件后重试）: {path_abs}"
        )
    if n_disk_after is None:
        return False, f"保存后无法读回校验: {path_abs}"
    detail = f"合并后 {n_merged} 行 → 磁盘 {n_disk_before or 0}→{n_disk_after} 行"
    if n_disk_before is not None and n_disk_after < n_disk_before:
        return False, f"磁盘行数变少（{detail}），已中止信任本次写入"
    if n_merged > 0 and n_disk_after == 0:
        return False, f"磁盘为空（{detail}）"
    return True, detail


def _seed_reply_output_from_excel(
    output_excel: str,
    seed_path: Optional[str],
    existing_model_aliases: Optional[Dict[str, str]] = None,
) -> None:
    """
    保证 output 表以种子大表为底：
    - 不存在 → 整表复制；
    - 已存在但行数远少于种子（例如误跑成只有几条）→ 种子 + 现有小表 upsert 合并，不丢已写入的新行。
    """
    if not seed_path or not str(seed_path).strip():
        return
    seed = os.path.abspath(str(seed_path).strip())
    out = os.path.abspath(output_excel)
    if out == seed:
        return
    if not os.path.exists(seed):
        print(f"  ⚠️  reply_seed_from_excel 不存在: {seed}")
        return
    n_seed = _count_excel_rows(seed) or 0
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if not os.path.exists(out):
        shutil.copy2(seed, out)
        print(f"  📋 已从旧表复制起点 → {out}（{n_seed} 行），本 run 只写此文件")
        return
    n_out = _count_excel_rows(out) or 0
    # 已有 output 且规模正常：不动
    if n_seed > 0 and n_out >= max(500, int(n_seed * 0.85)):
        print(f"  📋 续写已有 {out}（{n_out} 行，与种子表 {n_seed} 行规模一致）")
        return
    # output 过小：用种子垫底合并，避免「磁盘 0→1→2」把万行历史弄没
    print(
        f"  ⚠️  {os.path.basename(out)} 仅 {n_out} 行，远少于种子表 {n_seed} 行；"
        f"正在把种子表与现有行合并（不丢弃已写入的成功回复）…"
    )
    try:
        df_out = pd.read_excel(out)
        shutil.copy2(seed, out)
        merged = _upsert_replies_with_disk(df_out, out, existing_model_aliases)
        if safe_save_excel(merged, out):
            n_after = _count_excel_rows(out) or 0
            print(f"  ✅ 合并完成 → {out} 现为 {n_after} 行（种子 {n_seed} + 原 {n_out}）")
        else:
            print(f"  ❌ 合并保存失败，请关闭 Excel 后删除或改名 {out} 再重跑")
    except Exception as e:
        print(f"  ❌ 合并种子表失败: {e}；可手动复制 {seed} → {out} 后重跑")


def _sync_reply_from_auxiliary_excel(
    output_excel: str,
    aux_path: Optional[str],
    existing_model_aliases: Optional[Dict[str, str]] = None,
) -> None:
    """将辅助表（如 replies_compared.xlsx）里较新的行合并进主输出表，避免写错文件导致「看不见记录」。"""
    if not aux_path or not str(aux_path).strip():
        return
    aux = os.path.abspath(str(aux_path).strip())
    out = os.path.abspath(output_excel)
    if aux == out or not os.path.exists(aux):
        return
    n_aux = _count_excel_rows(aux) or 0
    if n_aux < 10:
        return
    n_out = _count_excel_rows(out) or 0
    print(
        f"  🔄 合并辅助表 {os.path.basename(aux)}（{n_aux} 行）→ 主表 {os.path.basename(out)}（{n_out} 行）…"
    )
    try:
        df_aux = pd.read_excel(aux)
        if os.path.exists(out):
            df_out = pd.read_excel(out)
            combined = pd.concat([df_out, df_aux], ignore_index=True)
        else:
            combined = df_aux
        merged = _upsert_replies_with_disk(combined, out, existing_model_aliases)
        if safe_save_excel(merged, out):
            print(f"  ✅ 主表现为 {_count_excel_rows(out)} 行：{out}")
        else:
            print(f"  ❌ 合并辅助表保存失败，请关闭 Excel 后重试")
    except Exception as e:
        print(f"  ⚠️  合并辅助表跳过: {e}")


def _apply_reply_max_questions(df: pd.DataFrame, max_questions: Optional[int]) -> pd.DataFrame:
    """仅用于 generate_replies：正整数则只取当前表前 N 行（应在按 source 等重排之后调用）；None/0/无效则全量。"""
    if df is None or df.empty:
        return df
    if max_questions is None:
        return df
    try:
        n = int(max_questions)
    except (TypeError, ValueError):
        return df
    if n <= 0:
        return df
    orig = len(df)
    out = df.head(n)
    if len(out) < orig:
        print(f"  reply_max_questions={n}：仅用前 {len(out)} 题（表中共 {orig} 题）\n")
    return out


def _resolve_reply_source_deprioritize(
    source_deprioritize: Optional[object],
) -> Optional[List[str]]:
    """
    False / 空列表 → 不重排；None → 默认 cello、ours 置后；非空 list → 小写去空后的 token 集合。
    """
    if source_deprioritize is False:
        return None
    if source_deprioritize is None:
        return ["cello", "ours"]
    if isinstance(source_deprioritize, list):
        xs = [str(x).strip().lower() for x in source_deprioritize if str(x).strip()]
        return xs if xs else None
    return ["cello", "ours"]


def _order_questions_by_source_deprioritize(
    df: pd.DataFrame,
    deprioritize_sources: Optional[List[str]],
) -> pd.DataFrame:
    """
    存在 source 列且 deprioritize_sources 非空时：source（忽略大小写）命中集合的行排在后面，其余优先。
    同组内保持原表行序。
    """
    if df is None or df.empty or not deprioritize_sources:
        return df
    if "source" not in df.columns:
        return df
    low = {str(x).strip().lower() for x in deprioritize_sources}
    out = df.copy().reset_index(drop=True)
    out["_reply_pri_idx"] = range(len(out))

    def tier(val) -> int:
        t = safe_str(val).strip().lower()
        if not t:
            return 0
        return 1 if t in low else 0

    out["_reply_src_tier"] = out["source"].map(tier)
    n_last = int(out["_reply_src_tier"].sum())
    n_first = len(out) - n_last
    out = (
        out.sort_values(by=["_reply_src_tier", "_reply_pri_idx"], kind="stable")
        .drop(columns=["_reply_src_tier", "_reply_pri_idx"])
        .reset_index(drop=True)
    )
    joined = "/".join(sorted(low))
    print(
        f"  题目顺序: 按 source 优先非 [{joined}]（{n_first} 条），"
        f"命中 [{joined}] 的置后（{n_last} 条）；reply_max_questions 截取前 N 行时先经此顺序\n"
    )
    return out


def _reply_source_scope_set(source_include: Optional[List[str]]) -> Optional[Set[str]]:
    """reply_source_include 非空时，跳过判定仅看该 source 下已有回复（避免 infobench 误挡 Ours 补跑）。"""
    if not source_include:
        return None
    out = {safe_str(s).strip().lower() for s in source_include if safe_str(s).strip()}
    return out or None


def _row_in_reply_source_scope(
    row_like,
    scope: Optional[Set[str]],
    ours_qids: Optional[Set[str]] = None,
) -> bool:
    if scope is None:
        return True
    src = safe_str(row_like.get("source", "")).strip().lower()
    if src in scope:
        return True
    # Ours 补跑：历史行 source 为空但 qid 属于 Ours 题，仍计入 skip（避免重复调 API）
    if ours_qids is not None and scope == {"ours"} and not src:
        qid = safe_str(row_like.get("qid", "")).strip()
        if qid in ours_qids:
            return True
    return False


def _vendor_family_for_logical_model(
    logical_model: str,
    existing_model_aliases: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """八大家厂商族，用于「按厂商判齐」跳过逻辑。"""
    canonical = _canonical_logical_model_for_skip(logical_model, existing_model_aliases)
    return eval_benchmark_eight_family("", canonical, None)


def _qid_vendor_already_has_reply(
    qid: str,
    logical_model: str,
    existing_keys: set,
    existing_vendor_keys: set,
    existing_model_aliases: Optional[Dict[str, str]] = None,
    *,
    skip_by_vendor_family: bool = True,
) -> bool:
    qid = safe_str(qid).strip()
    logical = _canonical_logical_model_for_skip(logical_model, existing_model_aliases)
    if (qid, logical) in existing_keys:
        return True
    if not skip_by_vendor_family:
        return False
    fam = _vendor_family_for_logical_model(logical, existing_model_aliases)
    return bool(fam and (qid, fam) in existing_vendor_keys)


def _canonical_logical_model_for_skip(
    model_name: str,
    existing_model_aliases: Optional[Dict[str, str]] = None,
) -> str:
    """将回复表里历史模型名映射到配置 logical_model，便于 skip 已导入的跨渠道回复。"""
    m = safe_str(model_name).strip()
    if not m or not existing_model_aliases:
        return m
    if m in existing_model_aliases:
        return safe_str(existing_model_aliases[m]).strip() or m
    low_map = {
        safe_str(k).strip().lower(): safe_str(v).strip()
        for k, v in existing_model_aliases.items()
        if safe_str(k).strip()
    }
    return low_map.get(m.lower(), m)


def _reply_row_is_success(row_like) -> bool:
    """兼容旧回复表：优先看 status=ok；无 status 时用 reply 是否为非 error 文本兜底。"""
    status = safe_str(row_like.get('status', '')).strip().lower()
    if status:
        return status == 'ok'
    reply = safe_str(row_like.get('reply', '')).strip()
    return bool(reply and not reply.startswith('<error'))


def _cooldown_sleep_if_needed() -> None:
    """当任意线程遇到 429 限流时触发全局冷却，减少并发雪崩。"""
    global _GLOBAL_COOLDOWN_UNTIL_TS
    with _GLOBAL_COOLDOWN_LOCK:
        until_ts = _GLOBAL_COOLDOWN_UNTIL_TS
    now_ts = time.time()
    if now_ts < until_ts:
        time.sleep(until_ts - now_ts)


def _set_global_cooldown(seconds: float) -> None:
    """设置全局冷却窗口（多线程共享）。"""
    global _GLOBAL_COOLDOWN_UNTIL_TS
    secs = max(0.0, float(seconds))
    if secs <= 0:
        return
    with _GLOBAL_COOLDOWN_LOCK:
        _GLOBAL_COOLDOWN_UNTIL_TS = max(_GLOBAL_COOLDOWN_UNTIL_TS, time.time() + secs)


def _build_messages_for_reply(query: str, history_context: str = '') -> List[Dict]:
    if not history_context or history_context.strip() in ('', '[]', 'nan', 'none', 'null'):
        return [{"role": "user", "content": query}]

    try:
        history = json.loads(history_context)
        if not isinstance(history, list) or len(history) == 0:
            return [{"role": "user", "content": query}]

        messages = []
        for turn in history:
            user_text = turn.get('user', '')
            assistant_text = turn.get('assistant', '')
            if user_text:
                messages.append({"role": "user", "content": user_text})
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})

        messages.append({"role": "user", "content": query})
        return messages
    except (json.JSONDecodeError, TypeError):
        return [{"role": "user", "content": query}]


def generate_reply(
        client: OAIClient,
        model: str,
        query: str,
        temperature: float = 0.7,
        enable_thinking: bool = False,
        retries: int = 5,
        history_context: str = '',
        system_prompt: str = '',
) -> Tuple[str, Optional[str], Optional[str]]:
    if isinstance(query, list):
        query = '\n'.join(str(item) for item in query if item is not None)
    elif not isinstance(query, str):
        query = str(query)

    messages = _build_messages_for_reply(query, history_context)
    sys_p = (system_prompt or '').strip()
    if sys_p:
        messages = [{"role": "system", "content": sys_p}] + messages
    last_err = None
    time.sleep(random.uniform(0.3, 0.8))

    for attempt in range(retries):
        _cooldown_sleep_if_needed()
        try:
            kwargs = {"model": model, "messages": messages, "temperature": temperature}
            if enable_thinking:
                kwargs["enable_thinking"] = True

            if hasattr(client, "chat_with_meta"):
                text, finish_reason, reasoning = client.chat_with_meta(**kwargs)
            else:
                text = client.chat(**kwargs)
                finish_reason = None
                reasoning = None

            if isinstance(text, list):
                text = '\n'.join(str(t) for t in text if t is not None) if text else ''
            elif not isinstance(text, str):
                text = str(text) if text is not None else ''

            if isinstance(text, str):
                is_html_error = (
                    '<a id="a-link"' in text or
                    'bixi.alicdn.com/punish' in text or
                    text.strip().startswith('<!DOCTYPE') or
                    text.strip().startswith('<html')
                )
                if is_html_error:
                    if attempt < retries - 1:
                        wait_time = 3.0 + (2 * attempt)
                        print(f"⚠️  模型 {model} 触发风控拦截，等待 {wait_time:.1f} 秒后重试 ({attempt + 1}/{retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        return "<error: 触发风控拦截>", None, None

            return text, finish_reason, reasoning

        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                err_str = str(e)
                err_l = err_str.lower()
                is_429 = ("429" in err_str) or ("Too Many Requests" in err_str) or ("rate limit" in err_l)
                is_gateway = is_transient_gateway_error(err_str)
                if is_429:
                    # 429 通常是限流窗口触发，使用更长 backoff，且写入全局冷却避免并发继续撞上窗口
                    wait_time = 12.0 * (attempt + 1) + random.uniform(0.0, 3.0)
                    _set_global_cooldown(wait_time)
                    print(
                        f"⚠️  模型 {model} 触发 429 限流，等待 {wait_time:.1f} 秒后重试 ({attempt + 1}/{retries}): {e}"
                    )
                elif is_gateway:
                    # 502/503/504 等多为网关或上游瞬时过载，短退避易连续撞超时，拉长间隔
                    wait_time = 18.0 + (12 * attempt) + random.uniform(0.0, 5.0)
                    print(
                        f"⚠️  模型 {model} 网关/上游超时 (502/503/504 等)，{wait_time:.1f} 秒后重试 ({attempt + 1}/{retries}): {e}"
                    )
                else:
                    wait_time = 2.0 + (2 * attempt) + random.uniform(0.0, 1.0)
                    print(f"⚠️  模型 {model} 调用失败，{wait_time:.1f}秒后重试 ({attempt + 1}/{retries}): {e}")
                time.sleep(wait_time)

    return f"<error: {repr(last_err)}>", None, None


def _ensure_oai_client(
    provider_config,
    clients: Dict[str, OAIClient],
    client_lock: Lock,
    timeout: int,
) -> str:
    client_key = f"{provider_config.name}::{provider_config.base_url}"
    if client_key not in clients:
        with client_lock:
            if client_key not in clients:
                clients[client_key] = OAIClient(
                    base_url=provider_config.base_url,
                    api_key=provider_config.api_key,
                    protocol=provider_config.protocol,
                    auth_header=provider_config.auth_header,
                    auth_prefix=provider_config.auth_prefix,
                    extra_headers=provider_config.extra_headers,
                    timeout=timeout,
                )
    return client_key


def _build_enriched_configs_multi(
    model_configs: List[Dict],
    *,
    timeout: int,
    clients: Dict[str, OAIClient],
    client_lock: Lock,
) -> List[Dict]:
    """支持 routes 多来源：每个 logical_model 一条 enriched，内含 routes_enriched。"""
    enriched: List[Dict] = []
    for cfg in model_configs:
        logical = logical_name_from_config(cfg) or str(cfg.get("model", "")).strip()
        raw_routes = cfg.get("routes")
        if raw_routes:
            routes = list(raw_routes)
        elif cfg.get("provider") and cfg.get("model"):
            routes = [dict(cfg)]
        else:
            print(f"  ⚠️  模型配置无效（无 routes / provider+model）: {cfg}")
            continue
        routes_enriched: List[Dict] = []
        for route in routes:
            mid = str(route.get("model", "")).strip()
            if not mid:
                continue
            try:
                pc = resolve_provider_config_for_route(route)
                ck = _ensure_oai_client(pc, clients, client_lock, timeout)
                re = {
                    "provider": pc.name,
                    "model": mid,
                    "client_key": ck,
                    "enable_thinking": bool(route.get("enable_thinking", False)),
                    "aimux_origin_code": (
                        str(route["aimux_origin_code"]).strip()
                        if route.get("aimux_origin_code") and str(route.get("aimux_origin_code")).strip()
                        else None
                    ),
                    "route_label": route.get("route_label") or route.get("table_source") or "",
                }
                sx = route.get(SOURCE_MODELS_EXCEL_KEY)
                if sx is not None and str(sx).strip():
                    re[SOURCE_MODELS_EXCEL_KEY] = str(sx).strip()
                routes_enriched.append(re)
                if len(clients) <= 20:
                    label = re["route_label"] or f"{pc.name}/{mid}"
                    print(
                        f"  ✅ [{logical}] {label} → {pc.name} / {mid}"
                        f" ({pc.protocol})"
                    )
            except (ValueError, Exception) as e:
                print(f"  ❌ [{logical}] 路由 {mid} 初始化失败: {e}")
        if not routes_enriched:
            print(f"  ❌ 逻辑模型 {logical} 无可用路由客户端，已跳过")
            continue
        enriched.append(
            {
                "logical_model": logical,
                "model": logical,
                "routes_enriched": routes_enriched,
            }
        )
    return enriched


def _call_reply_with_routes(
    task: dict,
    *,
    clients: Dict[str, OAIClient],
    client_lock: Lock,
    temperature: float,
    sticky_routes: Dict[str, Dict],
    sticky_lock: Lock,
    marked_failed_for_excel: set,
    models_excel_path: Optional[str],
    models_excel_paths: Optional[List[str]],
) -> Optional[dict]:
    """按 routes 顺序调用，鉴权失败则换下一来源；成功则记住 sticky 路由。"""
    logical = task.get("logical_model") or task.get("model_name") or task.get("model")
    routes = list(task.get("routes") or [])
    if not routes:
        return None
    with sticky_lock:
        pick = sticky_routes.get(logical)
    if pick:
        routes = [pick] + [r for r in routes if r is not pick]

    n_routes = len(routes)
    # 多路由时不在单条 504 上长退避，尽快切下一条（routify / 降级版本等）
    call_retries = 1 if n_routes > 1 else 4

    last_err_reply = ""
    for route in routes:
        prov = route["provider"]
        mid = route["model"]
        if MODEL_BLACKLIST.is_blacklisted(prov, mid):
            continue
        with client_lock:
            client = clients[route["client_key"]]
        reply, finish_reason, reasoning = generate_reply(
            client,
            mid,
            task["query"],
            temperature,
            enable_thinking=route.get("enable_thinking", False),
            retries=call_retries,
            history_context=task.get("history_context", ""),
            system_prompt=task.get("system_prompt", ""),
        )
        reply_str = reply if isinstance(reply, str) else (str(reply) if reply is not None else "")
        if reply_str.startswith("<error"):
            last_err_reply = reply_str
            transient_gw = is_transient_gateway_error(reply_str)
            if transient_gw:
                with sticky_lock:
                    if sticky_routes.get(logical) is route:
                        sticky_routes.pop(logical, None)
                tqdm.write(
                    f"⚠️  [{logical}] {prov}/{mid} 网关超时/504，换下一来源（不重试同网关）…"
                )
                continue
            if not MODEL_BLACKLIST.is_blacklisted(prov, mid):
                MODEL_BLACKLIST.add(prov, mid, reply_str)
                if is_permission_error(reply_str):
                    tqdm.write(
                        f"⚠️  [{logical}] {prov}/{mid} 鉴权失败，换下一来源…"
                    )
                excel_p = _resolve_models_excel_for_mark(
                    {**route, "provider": prov, "model": mid},
                    models_excel_path,
                    models_excel_paths,
                )
                if excel_p:
                    key = (excel_p, prov, mid, route.get("aimux_origin_code") or "")
                    if key not in marked_failed_for_excel:
                        if mark_single_model_unavailable(
                            excel_p, prov, mid, aimux_origin_code=route.get("aimux_origin_code"),
                        ):
                            marked_failed_for_excel.add(key)
            continue
        MODEL_BLACKLIST.mark_first_task_tested(prov, mid)
        with sticky_lock:
            sticky_routes[logical] = route
        reasoning_str = reasoning if isinstance(reasoning, str) else (str(reasoning) if reasoning else "") or ""
        out = {
            "qid": task["qid"],
            "model": logical,
            "provider": prov,
            "api_model": mid,
            "route_label": route.get("route_label", ""),
            "reply": reply_str,
            "reasoning": reasoning_str,
            "reply_len": len(reply_str),
            "reasoning_len": len(reasoning_str),
            "finish_reason": finish_reason,
            "enable_thinking": route.get("enable_thinking", False),
            "status": "ok",
            "timestamp": pd.Timestamp.now(),
        }
        if route.get("aimux_origin_code"):
            out["aimux_origin_code"] = route["aimux_origin_code"]
        if task.get("source"):
            out["source"] = task["source"]
        if "session_id" in task:
            out["session_id"] = task.get("session_id", "")
            out["turn_id"] = task.get("turn_id", "")
            out["history_context"] = task.get("history_context", "")
        return out
    if last_err_reply:
        tried = [f"{r.get('provider')}/{r.get('model')}" for r in routes]
        tqdm.write(
            f"⚠️  [{logical}] {len(tried)} 条路由均失败"
            f"（{' → '.join(tried[:4])}{'…' if len(tried) > 4 else ''}）: {last_err_reply[:60]}"
        )
    return {
        "qid": task["qid"],
        "model": logical,
        "provider": routes[-1]["provider"] if routes else "",
        "api_model": routes[-1]["model"] if routes else "",
        "reply": last_err_reply or "<error: all routes failed>",
        "reasoning": "",
        "reply_len": 0,
        "reasoning_len": 0,
        "finish_reason": "",
        "enable_thinking": False,
        "status": "error",
        "timestamp": pd.Timestamp.now(),
        **({"source": task["source"]} if task.get("source") else {}),
    }


def _task_has_usable_route(task: dict) -> bool:
    for route in task.get("routes") or []:
        if not MODEL_BLACKLIST.is_blacklisted(route.get("provider", ""), route.get("model", "")):
            return True
    return False


def _enriched_route_for_chosen(
    chosen_raw: Optional[dict],
    routes_enriched: List[dict],
) -> Optional[dict]:
    if not routes_enriched:
        return None
    if not chosen_raw:
        return routes_enriched[0]
    cm = str(chosen_raw.get("model", "")).strip()
    cp = str(chosen_raw.get("provider", "")).strip()
    for re in routes_enriched:
        if str(re.get("model", "")).strip() != cm:
            continue
        if not cp or str(re.get("provider", "")).strip() == cp:
            return re
    return routes_enriched[0]


def batch_generate_replies(
        questions_excel: str,
        model_configs: List[Dict[str, str]],
        output_excel: str,
        temperature: float = 0.7,
        max_workers: int = 4,
        checkpoint_interval: int = 10,
        timeout: int = 120,
        models_excel_path: Optional[str] = None,
        models_excel_paths: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        max_questions: Optional[int] = None,
        target_total_successes: Optional[int] = None,
        target_successes_per_qid: Optional[int] = None,
        auto_ordered_model_fallback: bool = False,
        source_deprioritize: Optional[object] = None,
        source_include: Optional[List[str]] = None,
        questions_sheet: Optional[object] = None,
        existing_model_aliases: Optional[Dict[str, str]] = None,
        reply_skip_by_vendor_family: bool = True,
        reply_seed_from_excel: Optional[str] = None,
        reply_sync_from_excel: Optional[str] = None,
) -> pd.DataFrame:
    print(f"\n{'=' * 60}")
    print(f"🚀 模块3: 批量生成回复")
    print(f"{'=' * 60}\n")

    _seed_reply_output_from_excel(
        output_excel, reply_seed_from_excel, existing_model_aliases,
    )
    _sync_reply_from_auxiliary_excel(
        output_excel, reply_sync_from_excel, existing_model_aliases,
    )
    out_abs = os.path.abspath(output_excel)
    n0 = _count_excel_rows(output_excel)
    print(f"  📁 本 run 写入: {out_abs}")
    if n0 is not None:
        print(f"  📊 启动时磁盘已有 {n0} 行（以此为准做 skip；勿用未刷新的 Excel 窗口行数）")
        if reply_seed_from_excel:
            n_seed = _count_excel_rows(
                os.path.abspath(str(reply_seed_from_excel).strip())
            )
            if n_seed and n0 < max(500, int(n_seed * 0.85)):
                raise RuntimeError(
                    f"输出表仅 {n0} 行，种子表 {n_seed} 行；"
                    f"请关闭 Excel 后重跑，或删除 {out_abs} 再启动以自动从种子表恢复。"
                )
    print()

    read_kw: Dict = {}
    if questions_sheet is not None and str(questions_sheet).strip() != "":
        read_kw["sheet_name"] = (
            int(questions_sheet)
            if str(questions_sheet).strip().isdigit()
            else questions_sheet
        )
    try:
        df_questions = pd.read_excel(questions_excel, **read_kw)
    except Exception as e:
        raise FileNotFoundError(f"无法读取题目文件: {e}")

    required_cols = ['qid', 'query']
    missing_cols = [col for col in required_cols if col not in df_questions.columns]
    if missing_cols:
        raise ValueError(f"题目表缺少必需列: {', '.join(missing_cols)}")

    qid_source_map = _build_qid_source_map(df_questions)
    df_questions = _filter_questions_by_source_include(df_questions, source_include)
    if df_questions.empty:
        raise ValueError("筛选后无题目（请检查 reply_source_include / 题目表 source 列）")

    dep = _resolve_reply_source_deprioritize(source_deprioritize)
    df_questions = _order_questions_by_source_deprioritize(df_questions, dep)
    df_questions = _apply_reply_max_questions(df_questions, max_questions)

    has_history_context = 'history_context' in df_questions.columns
    has_session_turn = 'session_id' in df_questions.columns and 'turn_id' in df_questions.columns
    has_sysprompt = 'sysprompt' in df_questions.columns
    n_sp = 0
    if has_sysprompt:
        n_sp = df_questions['sysprompt'].apply(lambda v: bool(safe_str(v).strip())).sum()
    if auto_ordered_model_fallback:
        print(
            f"  题目数量: {len(df_questions)}  模型池(顺序换模): {len(model_configs)} 个  "
            f"多轮对话: {'是' if has_history_context else '否'}"
        )
    else:
        print(f"  题目数量: {len(df_questions)}  模型数量: {len(model_configs)}  多轮对话: {'是' if has_history_context else '否'}")
    if has_sysprompt:
        print(f"  题目级 sysprompt: 列已存在，{n_sp} 行非空（非空行生成回复时将合并进 system）")
    if has_session_turn:
        print(f"  多轮字段: session_id, turn_id 将写入回复表，便于多轮分析与定位失败轮次")

    results = []
    existing_keys = set()
    existing_vendor_keys: Set[Tuple[str, str]] = set()
    existing_success_total = 0
    existing_success_by_qid: Dict[str, int] = {}
    reply_source_scope = _reply_source_scope_set(source_include)
    ours_qids_for_scope: Optional[Set[str]] = None
    if reply_source_scope == {"ours"}:
        ours_qids_for_scope = {
            safe_str(r.get("qid", "")).strip()
            for _, r in df_questions.iterrows()
            if safe_str(r.get("qid", "")).strip()
        }

    if os.path.exists(output_excel):
        try:
            df_existing = pd.read_excel(output_excel)
            if qid_source_map and "source" in df_existing.columns:
                for idx, row in df_existing.iterrows():
                    if safe_str(row.get("source", "")).strip():
                        continue
                    qid = safe_str(row.get("qid", "")).strip()
                    if qid and qid in qid_source_map:
                        df_existing.at[idx, "source"] = qid_source_map[qid]
            scoped_ok_rows = 0
            for _, row in df_existing.iterrows():
                qid_existing = safe_str(row.get('qid', ''))
                model_existing = _canonical_logical_model_for_skip(
                    row.get('model', ''), existing_model_aliases,
                )
                results.append(row.to_dict())
                if not (_reply_row_is_success(row) and qid_existing and model_existing):
                    continue
                if not _row_in_reply_source_scope(
                    row, reply_source_scope, ours_qids_for_scope,
                ):
                    continue
                scoped_ok_rows += 1
                existing_keys.add((qid_existing, model_existing))
                if reply_skip_by_vendor_family:
                    fam = eval_benchmark_eight_family(
                        row.get('provider'), row.get('model'), row.get('aimux_origin_code'),
                    ) or _vendor_family_for_logical_model(model_existing, existing_model_aliases)
                    if fam:
                        existing_vendor_keys.add((qid_existing, fam))
                existing_success_total += 1
                existing_success_by_qid[qid_existing] = existing_success_by_qid.get(qid_existing, 0) + 1
            scope_note = (
                f"仅统计 source∈{list(source_include)} 的成功行"
                if reply_source_scope
                else "全表成功行"
            )
            skip_note = (
                f"{scope_note}: {scoped_ok_rows} 行 → "
                f"{len(existing_keys)} 个 (qid,逻辑模型)"
                + (
                    f"、{len(existing_vendor_keys)} 个 (qid,厂商族) 已齐"
                    if reply_skip_by_vendor_family
                    else ""
                )
            )
            print(
                f"💾 发现已有结果: {len(df_existing)} 行；{skip_note}；"
                f"其余将尝试调用 API 补齐\n"
            )
        except Exception as e:
            print(
                f"⚠️  读取已有结果失败: {e}\n"
                f"  ⚠️  若 Excel 正打开该文件，请先关闭后重跑。"
                f"写回已与磁盘按 (qid,模型,source) 合并，避免 checkpoint 清空全表。\n"
            )

    logical_models = [
        logical_name_from_config(c) or safe_str(c.get("model", "")).strip()
        for c in model_configs
    ]
    logical_models = [m for m in logical_models if m]
    full_matrix = len(df_questions) * len(logical_models)
    pending_estimate = sum(
        1
        for _, row in df_questions.iterrows()
        for lm in logical_models
        if not _qid_vendor_already_has_reply(
            safe_str(row["qid"]), lm, existing_keys, existing_vendor_keys,
            existing_model_aliases, skip_by_vendor_family=reply_skip_by_vendor_family,
        )
    )
    if not auto_ordered_model_fallback:
        scope_lbl = f"source={source_include}" if reply_source_scope else "全表"
        print(
            f"  全量矩阵: {full_matrix} 条（{len(df_questions)} 题 × {len(logical_models)} 模型）"
            f"  →  待调用 API 约 {pending_estimate} 条（{scope_lbl} 下未齐）\n"
        )
    else:
        print(f"  并行粒度: 每题一线程顺序扫模型池（max_workers 题并发）\n")

    reply_system = (system_prompt or '').strip()

    clients: Dict[str, OAIClient] = {}
    client_lock = Lock()
    print("  初始化回复客户端（支持多来源 routes）…\n")
    enriched_configs = _build_enriched_configs_multi(
        model_configs, timeout=timeout, clients=clients, client_lock=client_lock,
    )
    sticky_routes: Dict[str, Dict] = {}
    sticky_lock = Lock()
    print()

    eff_target_total = None
    if target_total_successes is not None and str(target_total_successes).strip() != '':
        try:
            eff_target_total = int(target_total_successes)
            if eff_target_total <= 0:
                eff_target_total = None
        except (TypeError, ValueError):
            eff_target_total = None

    eff_target_per_qid = None
    if target_successes_per_qid is not None and str(target_successes_per_qid).strip() != '':
        try:
            eff_target_per_qid = int(target_successes_per_qid)
            if eff_target_per_qid <= 0:
                eff_target_per_qid = None
        except (TypeError, ValueError):
            eff_target_per_qid = None

    if auto_ordered_model_fallback and eff_target_per_qid is None:
        eff_target_per_qid = 2

    if eff_target_total is not None or eff_target_per_qid is not None:
        msg = [f"  已有成功回复: {existing_success_total}"]
        if eff_target_total is not None:
            msg.append(f"总目标: {eff_target_total}")
        if eff_target_per_qid is not None:
            msg.append(f"单题成功上限: {eff_target_per_qid}")
        print(f"  回复补齐策略: {' | '.join(msg)}\n")
        if auto_ordered_model_fallback:
            print(
                "  自动多模型补全: 每题按模型池顺序依次尝试，失败则换下一个模型；"
                "成功条数按「不同 model」计，直至达到单题目标或总目标。\n"
            )
        if eff_target_total is not None and existing_success_total >= eff_target_total:
            print(f"✅ 已达到目标成功回复数 {existing_success_total}/{eff_target_total}，无需继续补跑")
            MODEL_BLACKLIST.print_summary()
            df = pd.read_excel(output_excel) if os.path.exists(output_excel) else pd.DataFrame(results)
            return df, []

    tasks = []
    rows_by_qid = {}
    for _, row in df_questions.iterrows():
        qid = safe_str(row['qid'])
        query_raw = row['query']
        query = '\n'.join(str(i) for i in query_raw if i is not None) if isinstance(query_raw, list) else safe_str(query_raw)
        history_context_value = ''
        if has_history_context:
            raw_hc = row['history_context']
            if not pd.isna(raw_hc):
                history_context_value = str(raw_hc)
        session_id_val = safe_str(row.get('session_id', '')) if has_session_turn else ''
        turn_id_val = row.get('turn_id', '')
        if turn_id_val is not None and not pd.isna(turn_id_val):
            try:
                turn_id_val = int(float(turn_id_val))
            except (TypeError, ValueError):
                turn_id_val = ''
        else:
            turn_id_val = ''
        row_sp = safe_str(row['sysprompt']).strip() if has_sysprompt else ''
        rows_by_qid[qid] = {
            'query': query, 'history_context': history_context_value,
            'session_id': session_id_val, 'turn_id': turn_id_val,
            'sysprompt': row_sp,
            'source': _question_source_from_row(row) or qid_source_map.get(qid, ''),
        }

    qid_order = list(rows_by_qid.keys())

    if auto_ordered_model_fallback:
        if not enriched_configs:
            print("❌ 模型池为空（无成功初始化的客户端），跳过批量回复")
            MODEL_BLACKLIST.print_summary()
            df = pd.read_excel(output_excel) if os.path.exists(output_excel) else pd.DataFrame(results)
            return df, []

        success_models_by_qid: Dict[str, Set[str]] = {}
        for r in results:
            if not _reply_row_is_success(r):
                continue
            qk = safe_str(r.get('qid', ''))
            mk = safe_str(r.get('model', ''))
            if qk and mk:
                success_models_by_qid.setdefault(qk, set()).add(mk)

        work_qids: List[str] = []
        for qid in qid_order:
            if eff_target_per_qid is not None and len(success_models_by_qid.get(qid, set())) >= eff_target_per_qid:
                continue
            work_qids.append(qid)

        print(
            f"📝 自动多模型补全: {len(work_qids)} 道题待跑，模型池 {len(enriched_configs)} 个"
            f"（优先跨厂商/系族凑满每题成功数，同族仅在池内无法满足时再放宽；失败则按配置顺序换模）\n"
        )
        if not work_qids:
            print("✅ 所有题目已达到单题成功模型数目标")
            MODEL_BLACKLIST.print_summary()
            df = pd.read_excel(output_excel) if os.path.exists(output_excel) else pd.DataFrame(results)
            return df, []

        marked_failed_for_excel: set = set()

        def generate_task(task: dict):
            query = task['query']
            if not isinstance(query, str):
                query = '\n'.join(str(i) for i in query if i is not None) if isinstance(query, list) else str(query)
                task['query'] = query

            if MODEL_BLACKLIST.is_blacklisted(task['provider'], task['model']):
                return None

            with client_lock:
                client = clients[task['client_key']]

            reply, finish_reason, reasoning = generate_reply(
                client, task['model'], task['query'], temperature,
                enable_thinking=task.get('enable_thinking', False), retries=5,
                history_context=task.get('history_context', ''),
                system_prompt=task.get('system_prompt', ''),
            )

            reply_str = reply if isinstance(reply, str) else (str(reply) if reply is not None else '')
            if reply_str.startswith('<error'):
                transient_gw = is_transient_gateway_error(reply_str)
                if transient_gw:
                    tqdm.write(
                        f"⚠️  模型 {task['model']} 本题因网关超时/502/503/504 等失败，**未**加入运行黑名单、**未**标表格不可用；"
                        f"下一题仍会调用该模型。若每题都失败请降低 max_workers 或增大 config timeout。"
                    )
                elif not MODEL_BLACKLIST.is_blacklisted(task['provider'], task['model']):
                    MODEL_BLACKLIST.add(task['provider'], task['model'], reply_str)
                    if is_permission_error(reply_str) or is_connection_error(reply_str):
                        kind = "连接/服务端中断" if is_connection_error(reply_str) else "权限/限流"
                    else:
                        kind = "请求/响应异常"
                    tqdm.write(
                        f"⚠️  模型调用失败（{kind}），已加入本运行黑名单，后续题目将跳过: "
                        f"{task['provider']} / {task['model']}"
                    )
                    excel_p = _resolve_models_excel_for_mark(task, models_excel_path, models_excel_paths)
                    if excel_p:
                        key = (excel_p, task['provider'], task['model'], task.get('aimux_origin_code') or '')
                        if key not in marked_failed_for_excel:
                            if mark_single_model_unavailable(
                                excel_p,
                                task['provider'],
                                task['model'],
                                aimux_origin_code=task.get('aimux_origin_code'),
                            ):
                                marked_failed_for_excel.add(key)
                                tqdm.write(f"  📝 模型 {task['model']} 已标记为不可用（下次运行将不再出现）")
                return None
            MODEL_BLACKLIST.mark_first_task_tested(task['provider'], task['model'])

            reasoning_str = reasoning if isinstance(reasoning, str) else (str(reasoning) if reasoning else '') or ''
            out = {
                'qid': task['qid'], 'model': task['model_name'], 'provider': task['provider'],
                'reply': reply_str, 'reasoning': reasoning_str,
                'reply_len': len(reply_str),
                'reasoning_len': len(reasoning_str),
                'finish_reason': finish_reason,
                'enable_thinking': task.get('enable_thinking', False),
                'status': 'ok' if not reply_str.startswith('<error') else 'error',
                'timestamp': pd.Timestamp.now()
            }
            if task.get('aimux_origin_code'):
                out['aimux_origin_code'] = task['aimux_origin_code']
            if task.get('source'):
                out['source'] = task['source']
            if 'session_id' in task:
                out['session_id'] = task.get('session_id', '')
                out['turn_id'] = task.get('turn_id', '')
                out['history_context'] = task.get('history_context', '')
            return out

        state_lock = Lock()
        new_ok_ctr = [0]
        new_results: List[dict] = []
        failed_models_for_table: List[Dict] = []
        task_model_keys = {(c['provider'], c['model']) for c in enriched_configs}

        def fill_one_qid(qid: str) -> None:
            row = rows_by_qid[qid]
            got = set(success_models_by_qid.get(qid, set()))
            got_families: Set[str] = set()
            for mk in got:
                fam_m = reply_vendor_family("", mk, None)
                for cfg in enriched_configs:
                    if cfg['model'] == mk:
                        fam_m = reply_vendor_family(
                            cfg['provider'], mk, cfg.get('aimux_origin_code'),
                        )
                        break
                got_families.add(fam_m)
            while eff_target_per_qid is not None and len(got) < eff_target_per_qid:
                with state_lock:
                    if eff_target_total is not None and existing_success_total + new_ok_ctr[0] >= eff_target_total:
                        break
                got_success = False
                for strict_cv in (True, False):
                    for cfg in enriched_configs:
                        model_name = cfg['model']
                        provider_name = cfg['provider']
                        if MODEL_BLACKLIST.is_blacklisted(provider_name, model_name):
                            continue
                        if model_name in got:
                            continue
                        if (qid, model_name) in existing_keys:
                            continue
                        fam = reply_vendor_family(
                            provider_name, model_name, cfg.get('aimux_origin_code'),
                        )
                        if strict_cv and len(got) >= 1 and fam in got_families:
                            continue
                        t = {
                            'qid': qid, 'query': row['query'],
                            'provider': provider_name, 'model': model_name,
                            'model_name': model_name,
                            'client_key': cfg['client_key'],
                            'enable_thinking': cfg.get('enable_thinking', False),
                            'history_context': row['history_context'],
                        }
                        if cfg.get('aimux_origin_code'):
                            t['aimux_origin_code'] = cfg['aimux_origin_code']
                        if cfg.get(SOURCE_MODELS_EXCEL_KEY):
                            t[SOURCE_MODELS_EXCEL_KEY] = cfg[SOURCE_MODELS_EXCEL_KEY]
                        if row.get('source'):
                            t['source'] = row['source']
                        row_sp = (row.get('sysprompt') or '').strip()
                        eff_sys = merge_conversation_sysprompt(reply_system, row_sp) if row_sp else reply_system
                        if eff_sys:
                            t['system_prompt'] = eff_sys
                        if has_session_turn:
                            t['session_id'] = row['session_id']
                            t['turn_id'] = row['turn_id']
                        res = generate_task(t)
                        if res and res.get('status') == 'ok':
                            with state_lock:
                                new_results.append(res)
                                new_ok_ctr[0] += 1
                                existing_keys.add((qid, model_name))
                            got.add(model_name)
                            got_families.add(fam)
                            got_success = True
                            break
                    if got_success:
                        break
                if not got_success:
                    break

        failed_count = 0
        skipped_count = 0
        completed = 0
        cancel_pending_futures = False
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures_map = {executor.submit(fill_one_qid, qid): qid for qid in work_qids}
        try:
            for future in tqdm(
                as_completed(futures_map), total=len(futures_map), desc="🔄 生成进度(按题)", ncols=100
            ):
                try:
                    future.result()
                except Exception as e:
                    tqdm.write(f"❌ 题目任务失败: {e}")
                    failed_count += 1
                completed += 1
                with state_lock:
                    snapshot = results + new_results
                if completed % checkpoint_interval == 0:
                    df_temp = pd.DataFrame(snapshot)
                    saved, save_msg = _save_replies_df(
                        df_temp, output_excel, qid_source_map, existing_model_aliases
                    )
                    if saved:
                        tqdm.write(f"💾 检查点: {save_msg}")
                    else:
                        tqdm.write(f"❌ 检查点保存失败: {save_msg}")
                if _all_models_in_taskset_blacklisted(task_model_keys):
                    pending = sum(1 for f in futures_map if not f.done())
                    if pending:
                        tqdm.write(
                            f"⚠️  模型池已全部进入黑名单，取消剩余 {pending} 个待执行题目并结束（避免进度条空跑）。"
                        )
                    cancel_pending_futures = True
                    break
        finally:
            _executor_shutdown(executor, futures_map, cancel_pending_futures)

        results.extend(new_results)

        if not results:
            return pd.DataFrame(), failed_models_for_table

        df_final = _prepare_replies_df_for_save(pd.DataFrame(results), qid_source_map)
        saved, save_msg = _save_replies_df(df_final, output_excel, qid_source_map, existing_model_aliases)
        if saved:
            print(f"\n✅ 保存成功: {os.path.abspath(output_excel)}（{save_msg}）")
        else:
            print(f"\n❌ 保存失败: {save_msg}")

        print(f"\n{'=' * 60}")
        print(f"✅ 回复生成完成!")
        print(f"{'=' * 60}")
        print(f"  总结果数: {len(results)}  新增: {len(new_results)}")
        print(f"  成功: {sum(1 for r in results if r.get('status') == 'ok')}  失败: {failed_count}  跳过: {skipped_count}")
        if eff_target_total is not None:
            final_success_total = existing_success_total + len(new_results)
            print(f"  成功回复目标: {final_success_total}/{eff_target_total}")

        thinking_rows = df_final[df_final['enable_thinking'] == True] if saved else pd.DataFrame()
        if len(thinking_rows) > 0:
            print(f"  开启 thinking: {len(thinking_rows)} 条  平均推理长度: {thinking_rows['reasoning_len'].mean():.0f} 字符")

        print(f"{'=' * 60}\n")
        MODEL_BLACKLIST.print_summary()
        return df_final, failed_models_for_table

    # 按厂商分批：探针主路由定好后，先集中补齐该厂商全部 Ours 缺口，再换下一厂商。
    tasks_by_logical: Dict[str, List[dict]] = {}
    cfg_by_logical: Dict[str, dict] = {}
    planned_success_total = 0
    planned_success_by_qid: Dict[str, int] = {}
    hit_total_target = False
    for cfg in enriched_configs:
        logical = cfg.get("logical_model") or cfg.get("model")
        routes = cfg.get("routes_enriched") or []
        if not logical or not routes:
            continue
        cfg_by_logical[logical] = cfg
        for qid in qid_order:
            row = rows_by_qid[qid]
            if eff_target_per_qid is not None and existing_success_by_qid.get(qid, 0) >= eff_target_per_qid:
                continue
            if eff_target_total is not None and existing_success_total + planned_success_total >= eff_target_total:
                hit_total_target = True
                break
            if _qid_vendor_already_has_reply(
                qid, logical, existing_keys, existing_vendor_keys,
                existing_model_aliases, skip_by_vendor_family=reply_skip_by_vendor_family,
            ):
                continue
            if (
                eff_target_per_qid is not None
                and existing_success_by_qid.get(qid, 0) + planned_success_by_qid.get(qid, 0) >= eff_target_per_qid
            ):
                break
            t = {
                'qid': qid, 'query': row['query'],
                'logical_model': logical,
                'model_name': logical,
                'routes': routes,
                'history_context': row['history_context'],
            }
            row_sp = (row.get('sysprompt') or '').strip()
            eff_sys = merge_conversation_sysprompt(reply_system, row_sp) if row_sp else reply_system
            if eff_sys:
                t['system_prompt'] = eff_sys
            if has_session_turn:
                t['session_id'] = row['session_id']
                t['turn_id'] = row['turn_id']
            if row.get('source'):
                t['source'] = row['source']
            tasks_by_logical.setdefault(logical, []).append(t)
            planned_success_total += 1
            planned_success_by_qid[qid] = planned_success_by_qid.get(qid, 0) + 1
        if hit_total_target:
            break

    total_api = sum(
        len([t for t in ts if _task_has_usable_route(t)])
        for ts in tasks_by_logical.values()
    )
    print(
        f"📝 待调用 API: {total_api} 条，按 {len(tasks_by_logical)} 个厂商依次补齐"
        f"（每厂商多路由依次切换，504 不重试同网关）\n"
    )

    if total_api == 0:
        print(
            "✅ 当前范围内无需调用 API（均已齐或路由不可用）。"
            "若不应全部跳过，请检查 reply_source_include 与表中 source 列是否一致。"
        )
        MODEL_BLACKLIST.print_summary()
        df = pd.read_excel(output_excel) if os.path.exists(output_excel) else pd.DataFrame(results)
        return df, []

    marked_failed_for_excel: set = set()

    def generate_task(task: dict):
        query = task['query']
        if not isinstance(query, str):
            query = '\n'.join(str(i) for i in query if i is not None) if isinstance(query, list) else str(query)
            task['query'] = query
        return _call_reply_with_routes(
            task,
            clients=clients,
            client_lock=client_lock,
            temperature=temperature,
            sticky_routes=sticky_routes,
            sticky_lock=sticky_lock,
            marked_failed_for_excel=marked_failed_for_excel,
            models_excel_path=models_excel_path,
            models_excel_paths=models_excel_paths,
        )

    new_results = []
    failed_models_for_table = []
    failed_count = 0
    skipped_count = 0
    ok_count = 0
    last_ckpt_ok = 0

    def _process_batch_result(result, task_done: dict):
        nonlocal ok_count, failed_count, skipped_count, last_ckpt_ok
        if result is None:
            skipped_count += 1
            return
        if result.get('status') == 'ok':
            new_results.append(result)
            ok_count += 1
            qk = safe_str(task_done.get('qid', ''))
            lk = _canonical_logical_model_for_skip(
                task_done.get('logical_model') or task_done.get('model_name', ''),
                existing_model_aliases,
            )
            existing_keys.add((qk, lk))
            if reply_skip_by_vendor_family:
                fam = _vendor_family_for_logical_model(lk, existing_model_aliases)
                if fam and qk:
                    existing_vendor_keys.add((qk, fam))
            return
        failed_count += 1
        row_fail = {'provider': result.get('provider'), 'model': result.get('model')}
        if result.get('aimux_origin_code'):
            row_fail['aimux_origin_code'] = result['aimux_origin_code']
        failed_models_for_table.append(row_fail)

    logical_run_order = [
        (c.get("logical_model") or c.get("model"))
        for c in enriched_configs
        if (c.get("logical_model") or c.get("model"))
    ]

    for logical in logical_run_order:
        batch = [t for t in tasks_by_logical.get(logical, []) if _task_has_usable_route(t)]
        if not batch:
            continue
        cfg = cfg_by_logical.get(logical, {})
        routes_en = cfg.get("routes_enriched") or []
        chosen_en = _enriched_route_for_chosen(cfg.get("chosen_route"), routes_en)
        if chosen_en:
            with sticky_lock:
                sticky_routes[logical] = chosen_en
        main_api = chosen_en.get("model", "?") if chosen_en else "?"
        print(f"\n{'─' * 48}\n▶ [{logical}] 主路由 {main_api} · 待补 {len(batch)} 题\n{'─' * 48}")

        model_ok_start = ok_count
        pbar = tqdm(total=len(batch), desc=f"  {logical}", ncols=100, leave=True)
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures_map = {executor.submit(generate_task, t): t for t in batch}
        try:
            for future in as_completed(futures_map):
                try:
                    _process_batch_result(future.result(), futures_map[future])
                    pbar.update(1)
                    pbar.set_postfix(
                        本批成功=ok_count - model_ok_start,
                        累计成功=ok_count,
                        refresh=False,
                    )
                    if ok_count > last_ckpt_ok:
                        last_ckpt_ok = ok_count
                        df_temp = pd.DataFrame(results + new_results)
                        saved, save_msg = _save_replies_df(
                            df_temp, output_excel, qid_source_map, existing_model_aliases
                        )
                        if saved:
                            tqdm.write(
                                f"  💾 检查点：本次运行新增成功 {ok_count} 条；{save_msg}"
                            )
                        else:
                            tqdm.write(f"  ❌ 检查点保存失败：{save_msg}")
                except Exception as e:
                    tqdm.write(f"  ❌ [{logical}] 任务异常: {e}")
                    failed_count += 1
                    pbar.update(1)
        finally:
            pbar.close()
            _executor_shutdown(executor, futures_map, False)
        print(f"  ✓ [{logical}] 本批新增成功 {ok_count - model_ok_start} 条（本次运行累计 {ok_count}）")
        if new_results:
            results.extend(new_results)
            new_results.clear()

    print(
        f"📊 本批完成: 新增成功={ok_count}  API失败={failed_count}  无结果={skipped_count}"
    )
    if ok_count == 0 and total_api > 0:
        print(
            "  ⚠️  本批无任何新增成功回复。请查看上方鉴权/503 日志；"
            "若不应全部失败，确认 Ours 题在表中 source 列是否为「Ours」/「ours」。"
        )
    if new_results:
        results.extend(new_results)
        new_results.clear()

    if not results:
        return pd.DataFrame(), failed_models_for_table

    df_final = _prepare_replies_df_for_save(pd.DataFrame(results), qid_source_map)
    saved, save_msg = _save_replies_df(df_final, output_excel, qid_source_map, existing_model_aliases)
    if saved:
        print(f"\n✅ 保存成功: {os.path.abspath(output_excel)}（{save_msg}）")
    else:
        print(f"\n❌ 保存失败: {save_msg}")

    print(f"\n{'=' * 60}")
    print(f"✅ 回复生成完成!")
    print(f"{'=' * 60}")
    print(f"  总结果数: {len(results)}  新增: {len(new_results)}")
    print(f"  成功: {sum(1 for r in results if r['status'] == 'ok')}  失败: {failed_count}  跳过: {skipped_count}")
    if eff_target_total is not None:
        final_success_total = existing_success_total + len(new_results)
        print(f"  成功回复目标: {final_success_total}/{eff_target_total}")

    thinking_rows = df_final[df_final['enable_thinking'] == True]
    if len(thinking_rows) > 0:
        print(f"  开启 thinking: {len(thinking_rows)} 条  平均推理长度: {thinking_rows['reasoning_len'].mean():.0f} 字符")

    print(f"{'=' * 60}\n")
    MODEL_BLACKLIST.print_summary()
    return df_final, failed_models_for_table


def _is_wide_paired_replies_df(df: Optional[pd.DataFrame]) -> bool:
    if df is None or df.empty:
        return False
    cols = set(df.columns.astype(str))
    return {'model_a', 'model_b', 'reply_a', 'reply_b'}.issubset(cols)


def _side_row_pack(sr: Optional[pd.Series], suffix: str) -> Dict:
    if sr is None:
        return {
            f'provider_{suffix}': '', f'model_{suffix}': '', f'aimux_origin_code_{suffix}': '',
            f'reply_{suffix}': '', f'status_{suffix}': '',
        }

    def g(key: str, default=''):
        if key not in sr.index:
            return default
        v = sr[key]
        return default if pd.isna(v) else v

    if 'reply' not in sr.index:
        reply_val = ''
    else:
        rv = sr['reply']
        reply_val = '' if pd.isna(rv) else (rv if isinstance(rv, str) else str(rv))

    return {
        f'provider_{suffix}': safe_str(g('provider', '')),
        f'model_{suffix}': safe_str(g('model', '')),
        f'aimux_origin_code_{suffix}': safe_str(g('aimux_origin_code', '')),
        f'reply_{suffix}': reply_val,
        f'status_{suffix}': safe_str(g('status', '')),
    }


def _long_side_records_to_wide_df(records: List[dict]) -> pd.DataFrame:
    """将按侧的回复记录（含 pair_id/side）汇总为每场对战一行：model_a/reply_a + model_b/reply_b。"""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    need = {'qid', 'pair_id', 'side', 'model'}
    if not need.issubset(set(df.columns.astype(str))):
        return pd.DataFrame()
    df = df.copy()
    df['pair_id'] = df['pair_id'].astype(str).str.strip()
    df['side'] = df['side'].astype(str).str.strip()
    rows_out: List[dict] = []
    for (qid, pair_id), g in df.groupby(['qid', 'pair_id']):
        if not pair_id or str(pair_id).lower() in ('nan', 'none', 'null'):
            continue
        ga = g[g['side'] == 'A']
        gb = g[g['side'] == 'B']
        ra = ga.iloc[-1] if not ga.empty else None
        rb = gb.iloc[-1] if not gb.empty else None
        query = ''
        if 'query' in g.columns and g['query'].notna().any():
            query = safe_str(g['query'].dropna().iloc[0])
        row = {
            'qid': qid,
            'pair_id': pair_id,
            'query': query,
            **_side_row_pack(ra, 'a'),
            **_side_row_pack(rb, 'b'),
            'timestamp': pd.Timestamp.now(),
        }
        if 'session_id' in g.columns and g['session_id'].notna().any():
            row['session_id'] = safe_str(g['session_id'].dropna().iloc[0])
        if 'turn_id' in g.columns and g['turn_id'].notna().any():
            row['turn_id'] = g['turn_id'].dropna().iloc[0]
        rows_out.append(row)
    if not rows_out:
        return pd.DataFrame()
    out = pd.DataFrame(rows_out)
    pref = [
        'qid', 'pair_id', 'query',
        'provider_a', 'model_a', 'aimux_origin_code_a', 'reply_a', 'status_a',
        'provider_b', 'model_b', 'aimux_origin_code_b', 'reply_b', 'status_b',
    ]
    tail = [c for c in ['session_id', 'turn_id', 'timestamp'] if c in out.columns]
    ordered = [c for c in pref if c in out.columns] + [c for c in out.columns if c not in pref + tail] + tail
    return out[[c for c in ordered if c in out.columns]]


def _wide_paired_replies_to_long_records(df: pd.DataFrame) -> List[dict]:
    """宽表（每场一行）→ 按侧记录列表，用于断点续跑与内存合并。"""
    out: List[dict] = []
    for _, row in df.iterrows():
        qid = safe_str(row['qid'])
        pair_id = safe_str(row['pair_id'])
        query = safe_str(row.get('query', ''))
        for suffix, side in [('a', 'A'), ('b', 'B')]:
            model_col = f'model_{suffix}'
            if model_col not in row.index:
                continue
            model_val = row[model_col]
            if pd.isna(model_val) or str(model_val).strip() == '':
                continue
            prov_col = f'provider_{suffix}'
            oc_col = f'aimux_origin_code_{suffix}'
            rep_col = f'reply_{suffix}'
            st_col = f'status_{suffix}'
            if st_col in row.index:
                st_raw = row[st_col]
                if pd.isna(st_raw) or str(st_raw).strip() == '':
                    status_v = 'ok'
                else:
                    status_v = safe_str(st_raw)
            else:
                status_v = 'ok'
            out.append({
                'qid': qid,
                'pair_id': pair_id,
                'side': side,
                'model': str(model_val).strip(),
                'provider': safe_str(row.get(prov_col, '')),
                'aimux_origin_code': row.get(oc_col, '') if oc_col in row.index else '',
                'reply': row.get(rep_col, '') if rep_col in row.index else '',
                'status': status_v,
                'query': query,
                'session_id': safe_str(row.get('session_id', '')),
                'turn_id': row.get('turn_id', ''),
            })
    return out


def _inject_query_from_questions(records: Optional[List[dict]], qid_to_query: Optional[Dict[str, str]]) -> None:
    """用题目表 qid→query 写入每条按侧记录，便于宽表 matches/replies 含题干供标注。"""
    if not records or not qid_to_query:
        return
    for r in records:
        qid = str(r.get('qid', '')).strip()
        if not qid:
            continue
        qt = qid_to_query.get(qid)
        if qt is None:
            continue
        qs = qt if isinstance(qt, str) else str(qt)
        if qs.strip():
            r['query'] = qs


def _wide_df_apply_query_map(wide: pd.DataFrame, qid_to_query: Optional[Dict[str, str]]) -> pd.DataFrame:
    """宽表按 qid 补全/覆盖 query 列（题目表为准）。"""
    if wide.empty or not qid_to_query or 'qid' not in wide.columns:
        return wide
    w = wide.copy()
    if 'query' not in w.columns:
        w['query'] = ''

    def lookup(q: str) -> Optional[str]:
        v = qid_to_query.get(str(q).strip())
        if v is None:
            return None
        s = v if isinstance(v, str) else str(v)
        return s if str(s).strip() else None

    keys_series = w['qid'].astype(str).str.strip()
    mapped = keys_series.map(lambda q: lookup(q))
    has = mapped.notna()
    if has.any():
        w.loc[has, 'query'] = mapped[has].values
    return w


def _load_qid_query_map_from_questions_excel(questions_excel: str) -> Dict[str, str]:
    """从题目表读取 qid→query（与组队任务构造规则一致）。"""
    out: Dict[str, str] = {}
    if not questions_excel or not os.path.isfile(questions_excel):
        return out
    try:
        dq = pd.read_excel(questions_excel)
    except Exception:
        return out
    if 'qid' not in dq.columns or 'query' not in dq.columns:
        return out
    for _, row in dq.iterrows():
        qid = safe_str(row['qid'])
        if not qid:
            continue
        query_raw = row['query']
        query = (
            '\n'.join(str(i) for i in query_raw if i is not None)
            if isinstance(query_raw, list)
            else safe_str(query_raw)
        )
        if query.strip():
            out[qid] = query
    return out


def _long_side_records_to_existing_keys(records: List[dict]) -> set:
    """已成功生成回复的 (qid, model)，用于跳过 API（与 _reply_row_is_success 一致）。"""
    keys = set()
    for r in records:
        if not _reply_row_is_success(r):
            continue
        qk = safe_str(r.get('qid', ''))
        mk = safe_str(r.get('model', ''))
        if qk and mk:
            keys.add((qk, mk))
    return keys


def _save_paired_wide_outputs(
    records: List[dict],
    output_excel: str,
    matches_excel: str,
    qid_to_query: Optional[Dict[str, str]] = None,
) -> bool:
    if qid_to_query:
        _inject_query_from_questions(records, qid_to_query)
    wide = _long_side_records_to_wide_df(records)
    if wide.empty:
        return False
    if qid_to_query:
        wide = _wide_df_apply_query_map(wide, qid_to_query)
    ok = safe_save_excel(wide, output_excel)
    if matches_excel and str(matches_excel).strip():
        mp = str(matches_excel).strip()
        if os.path.abspath(mp) != os.path.abspath(output_excel):
            safe_save_excel(wide, mp)
    return ok


def _vendor_key_for_pairing(c: Dict) -> str:
    """对战分组用「厂商」键：Aimux 用供应商(原厂)；其它用 provider 名。"""
    cp = str(c.get('provider', '')).strip().lower()
    if cp == 'aimux':
        oc = str(c.get('aimux_origin_code', '') or '').strip().lower()
        return oc if oc else 'aimux'
    return cp or 'unknown'


def _enriched_is_judge(
    c: Dict,
    judge_provider: Optional[str],
    judge_model: Optional[str],
    judge_aimux_origin: Optional[str] = None,
) -> bool:
    """是否与当前裁判配置为同一条模型（对战池应排除，仅生成回复）。"""
    if not judge_model or not str(judge_model).strip():
        return False
    if str(c.get('model', '')).strip() != str(judge_model).strip():
        return False
    jp = str(judge_provider or '').strip().lower()
    cp = str(c.get('provider', '')).strip().lower()
    if jp and cp != jp:
        return False
    if cp == 'aimux':
        jo = str(judge_aimux_origin or '').strip().lower()
        co = str(c.get('aimux_origin_code') or '').strip().lower()
        if jo and co != jo:
            return False
    return True


def _pair_picked_cross_vendor(picked: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """
    将本轮选出的模型两两组队，贪心优先「不同厂商」配对，减少同厂商内战。
    picked 顺序仍由上游轮询决定；仅在组队时重排配对关系。
    """
    remaining = list(picked)
    pairs: List[Tuple[Dict, Dict]] = []
    while len(remaining) >= 2:
        a = remaining.pop(0)
        vk_a = _vendor_key_for_pairing(a)
        b_i = -1
        for i, b in enumerate(remaining):
            if _vendor_key_for_pairing(b) != vk_a:
                b_i = i
                break
        if b_i < 0:
            b_i = 0
        b = remaining.pop(b_i)
        pairs.append((a, b))
    return pairs


def _round_robin_pick(enriched_pool: List[Dict], start: int, k: int) -> List[Dict]:
    """从 enriched_pool 中按 start 轮询取 k 个（不重复）。"""
    if not enriched_pool or k <= 0:
        return []
    n = len(enriched_pool)
    k = min(int(k), n)
    out: List[Dict] = []
    idx = int(start) % n
    seen = set()
    while len(out) < k and len(seen) < n:
        m = enriched_pool[idx]
        key = (
            str(m.get('provider', '')).strip(),
            str(m.get('model', '')).strip(),
            str(m.get('aimux_origin_code', '')).strip(),
        )
        if key not in seen:
            out.append(m)
            seen.add(key)
        idx = (idx + 1) % n
    return out


def _write_matches_from_replies(
    replies_df: pd.DataFrame,
    matches_excel: str,
    questions_excel: Optional[str] = None,
) -> None:
    """从按侧长表（pair_id/side）生成每场一行的宽表并写入 matches；可选 questions_excel 注入 query。"""
    if replies_df is None or replies_df.empty or not matches_excel:
        return
    if 'pair_id' not in replies_df.columns or 'side' not in replies_df.columns:
        return
    qmap = _load_qid_query_map_from_questions_excel(questions_excel) if questions_excel else {}
    recs = replies_df.to_dict('records')
    _inject_query_from_questions(recs, qmap or None)
    wide = _long_side_records_to_wide_df(recs)
    if qmap:
        wide = _wide_df_apply_query_map(wide, qmap)
    if not wide.empty:
        safe_save_excel(wide, matches_excel)


def batch_generate_replies_paired_round_robin(
        questions_excel: str,
        model_configs: List[Dict[str, str]],
        output_excel: str,
        matches_excel: str,
        per_question_max_models: int = 8,
        temperature: float = 0.7,
        max_workers: int = 4,
        checkpoint_interval: int = 10,
        timeout: int = 120,
        models_excel_path: Optional[str] = None,
        models_excel_paths: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        max_questions: Optional[int] = None,
        exclude_judge_from_paired_pool: bool = True,
        judge_provider: Optional[str] = None,
        judge_model: Optional[str] = None,
        judge_aimux_origin_code: Optional[str] = None,
        prefer_cross_vendor_pairing: bool = True,
        source_deprioritize: Optional[object] = None,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    B 模式：按题轮询分配模型，最多 per_question_max_models（默认 8），并两两组队生成对局。
    默认排除裁判模型出对战池；默认贪心优先跨厂商配对（同厂商尽量少打）。
    输出（均为「每场对战一行」，含 model_a/reply_a 与 model_b/reply_b）：
    - replies.xlsx（output_excel）：宽表主产物
    - matches.xlsx：与 replies 列结构一致；若路径与 replies 相同则只写一份
    断点续跑：支持宽表或旧版按侧长表；按已成功 (qid, model) 跳过 API。
    """
    print(f"\n{'=' * 60}")
    print(f"🚀 模块3B: 组队模式生成回复（按题轮询，两两对局）")
    print(f"{'=' * 60}\n")

    df_questions = pd.read_excel(questions_excel)
    required_cols = ['qid', 'query']
    missing_cols = [col for col in required_cols if col not in df_questions.columns]
    if missing_cols:
        raise ValueError(f"题目表缺少必需列: {', '.join(missing_cols)}")

    dep = _resolve_reply_source_deprioritize(source_deprioritize)
    df_questions = _order_questions_by_source_deprioritize(df_questions, dep)
    df_questions = _apply_reply_max_questions(df_questions, max_questions)

    has_history_context = 'history_context' in df_questions.columns
    has_session_turn = 'session_id' in df_questions.columns and 'turn_id' in df_questions.columns
    has_sysprompt = 'sysprompt' in df_questions.columns
    reply_system = (system_prompt or '').strip()
    if has_sysprompt:
        n_sp = df_questions['sysprompt'].apply(lambda v: bool(safe_str(v).strip())).sum()
        print(f"  题目级 sysprompt: {n_sp} 行非空（将合并进对战请求的 system）\n")

    clients: Dict[str, OAIClient] = {}
    client_lock = Lock()
    enriched_configs = []

    for cfg in model_configs:
        model_name = cfg['model']
        try:
            if cfg.get('provider'):
                if (
                    str(cfg.get('provider')).strip() == 'aimux'
                    and cfg.get('aimux_origin_code')
                    and str(cfg.get('aimux_origin_code')).strip()
                ):
                    provider_config = aimux_provider_for_origin(str(cfg['aimux_origin_code']).strip())
                else:
                    provider_config = get_provider(cfg['provider'])
            else:
                provider_config = get_provider_for_model(model_name)
            client_key = f"{provider_config.name}::{provider_config.base_url}"
            ec = {
                'model': model_name,
                'provider': provider_config.name,
                'client_key': client_key,
                'provider_config': provider_config,
                'enable_thinking': cfg.get('enable_thinking', False),
                'aimux_origin_code': (
                    str(cfg['aimux_origin_code']).strip()
                    if cfg.get('aimux_origin_code') and str(cfg.get('aimux_origin_code')).strip()
                    else None
                ),
            }
            sx = cfg.get(SOURCE_MODELS_EXCEL_KEY)
            if sx is not None and str(sx).strip():
                ec[SOURCE_MODELS_EXCEL_KEY] = str(sx).strip()
            enriched_configs.append(ec)
            if client_key not in clients:
                clients[client_key] = OAIClient(
                    base_url=provider_config.base_url, api_key=provider_config.api_key,
                    protocol=provider_config.protocol, auth_header=provider_config.auth_header,
                    auth_prefix=provider_config.auth_prefix, extra_headers=provider_config.extra_headers,
                    timeout=timeout
                )
        except Exception as e:
            print(f"  ❌ 模型 {model_name} 配置失败: {e}")

    # 候选池（过滤黑名单；可选排除裁判，使非裁判模型全部进入轮询对战）
    pool = []
    skipped_judge = 0
    for c in enriched_configs:
        if MODEL_BLACKLIST.is_blacklisted(c['provider'], c['model']):
            continue
        if (
            exclude_judge_from_paired_pool
            and judge_model
            and str(judge_model).strip()
            and _enriched_is_judge(c, judge_provider, judge_model, judge_aimux_origin_code)
        ):
            skipped_judge += 1
            continue
        pool.append(c)
    if not pool:
        raise RuntimeError("无可用模型（均被黑名单、或与裁判重合后为空、或初始化失败）")
    if skipped_judge:
        print(
            f"  对战池已排除裁判模型: provider={judge_provider!r} model={judge_model!r}"
            f"{' aimux_origin=' + repr(judge_aimux_origin_code) if judge_aimux_origin_code else ''} "
            f"（共跳过 {skipped_judge} 条配置项）\n"
        )
    print(
        f"  对战池: {len(pool)} 个模型  跨厂商优先配对: "
        f"{'是' if prefer_cross_vendor_pairing else '否（按轮询顺序相邻配对）'}\n"
    )

    results: List[dict] = []
    existing_keys = set()
    if os.path.exists(output_excel):
        try:
            df_existing = pd.read_excel(output_excel)
            if _is_wide_paired_replies_df(df_existing):
                results = _wide_paired_replies_to_long_records(df_existing)
                existing_keys = _long_side_records_to_existing_keys(results)
                print(f"💾 发现已有对战表（每场一行）: {len(df_existing)} 场，续跑按已完成模型跳过\n")
            elif not df_existing.empty and 'pair_id' in df_existing.columns and 'side' in df_existing.columns:
                for _, row in df_existing.iterrows():
                    results.append(row.to_dict())
                existing_keys = _long_side_records_to_existing_keys(results)
                print(f"💾 发现旧版按侧 replies: {len(results)} 条，将汇总为每场一行写入\n")
            elif not df_existing.empty and 'qid' in df_existing.columns and 'model' in df_existing.columns:
                for _, row in df_existing.iterrows():
                    if _reply_row_is_success(row):
                        qe = safe_str(row.get('qid', ''))
                        me = safe_str(row.get('model', ''))
                        if qe and me:
                            existing_keys.add((qe, me))
                print(
                    f"💾 发现非组队格式 replies: 成功 {len(existing_keys)} 个 (qid, model) 将跳过，"
                    f"失败行将补跑；输出将为新的每场一行宽表\n"
                )
        except Exception as e:
            print(f"⚠️  读取已有结果失败: {e}\n")

    # 题目顺序
    question_rows = []
    for _, row in df_questions.iterrows():
        qid = safe_str(row['qid'])
        query_raw = row['query']
        query = '\n'.join(str(i) for i in query_raw if i is not None) if isinstance(query_raw, list) else safe_str(query_raw)
        history_context_value = ''
        if has_history_context:
            raw_hc = row['history_context']
            if not pd.isna(raw_hc):
                history_context_value = str(raw_hc)
        session_id_val = safe_str(row.get('session_id', '')) if has_session_turn else ''
        turn_id_val = row.get('turn_id', '')
        if turn_id_val is not None and not pd.isna(turn_id_val):
            try:
                turn_id_val = int(float(turn_id_val))
            except (TypeError, ValueError):
                turn_id_val = ''
        else:
            turn_id_val = ''
        row_sp = safe_str(row['sysprompt']).strip() if has_sysprompt else ''
        question_rows.append({
            'qid': qid,
            'query': query,
            'history_context': history_context_value,
            'session_id': session_id_val,
            'turn_id': turn_id_val,
            'sysprompt': row_sp,
        })

    qid_to_query: Dict[str, str] = {q['qid']: q['query'] for q in question_rows}
    _inject_query_from_questions(results, qid_to_query)

    tasks = []
    # 按题轮询取 8 个：start = qi*per_question_max_models
    k = int(per_question_max_models) if per_question_max_models else 8
    for qi, q in enumerate(question_rows):
        picked = _round_robin_pick(pool, start=qi * k, k=k)
        if prefer_cross_vendor_pairing:
            pairs_q = _pair_picked_cross_vendor(picked)
        else:
            pairs_q = [(picked[i], picked[i + 1]) for i in range(0, len(picked) - 1, 2)]
        for pair_no, (a, b) in enumerate(pairs_q, start=1):
            pair_id = f"{q['qid']}::P{pair_no:02d}"

            if (q['qid'], a['model']) not in existing_keys:
                t = {
                    'qid': q['qid'], 'query': q['query'],
                    'provider': a['provider'], 'model': a['model'],
                    'model_name': a['model'],
                    'client_key': a['client_key'],
                    'enable_thinking': a.get('enable_thinking', False),
                    'history_context': q['history_context'],
                    'pair_id': pair_id, 'side': 'A',
                    'opponent_model': b['model'], 'opponent_provider': b['provider'],
                }
                if a.get('aimux_origin_code'):
                    t['aimux_origin_code'] = a.get('aimux_origin_code')
                if a.get(SOURCE_MODELS_EXCEL_KEY):
                    t[SOURCE_MODELS_EXCEL_KEY] = a[SOURCE_MODELS_EXCEL_KEY]
                row_sp = (q.get('sysprompt') or '').strip()
                eff_sys = merge_conversation_sysprompt(reply_system, row_sp) if row_sp else reply_system
                if eff_sys:
                    t['system_prompt'] = eff_sys
                if has_session_turn:
                    t['session_id'] = q['session_id']
                    t['turn_id'] = q['turn_id']
                tasks.append(t)

            if (q['qid'], b['model']) not in existing_keys:
                t = {
                    'qid': q['qid'], 'query': q['query'],
                    'provider': b['provider'], 'model': b['model'],
                    'model_name': b['model'],
                    'client_key': b['client_key'],
                    'enable_thinking': b.get('enable_thinking', False),
                    'history_context': q['history_context'],
                    'pair_id': pair_id, 'side': 'B',
                    'opponent_model': a['model'], 'opponent_provider': a['provider'],
                }
                if b.get('aimux_origin_code'):
                    t['aimux_origin_code'] = b.get('aimux_origin_code')
                if b.get(SOURCE_MODELS_EXCEL_KEY):
                    t[SOURCE_MODELS_EXCEL_KEY] = b[SOURCE_MODELS_EXCEL_KEY]
                row_sp = (q.get('sysprompt') or '').strip()
                eff_sys = merge_conversation_sysprompt(reply_system, row_sp) if row_sp else reply_system
                if eff_sys:
                    t['system_prompt'] = eff_sys
                if has_session_turn:
                    t['session_id'] = q['session_id']
                    t['turn_id'] = q['turn_id']
                tasks.append(t)

    print(f"📝 待处理任务: {len(tasks)} 条（组队模式）\n")
    if not tasks:
        if results:
            _save_paired_wide_outputs(results, output_excel, matches_excel, qid_to_query=qid_to_query)
        df_out = _long_side_records_to_wide_df(results)
        if qid_to_query and not df_out.empty:
            df_out = _wide_df_apply_query_map(df_out, qid_to_query)
        if df_out.empty and os.path.exists(output_excel):
            df_out = pd.read_excel(output_excel)
        return df_out, []

    marked_failed_for_excel: set = set()
    new_results = []
    failed_models_for_table = []
    failed_count = 0
    skipped_count = 0
    completed = 0

    def generate_task(task: dict):
        if MODEL_BLACKLIST.is_blacklisted(task['provider'], task['model']):
            return None
        with client_lock:
            client = clients[task['client_key']]
        reply, finish_reason, reasoning = generate_reply(
            client, task['model'], task['query'], temperature,
            enable_thinking=task.get('enable_thinking', False), retries=5,
            history_context=task.get('history_context', ''),
            system_prompt=task.get('system_prompt', ''),
        )
        reply_str = reply if isinstance(reply, str) else (str(reply) if reply is not None else '')
        if reply_str.startswith('<error'):
            err = {
                'qid': task['qid'], 'model': task['model_name'], 'provider': task['provider'],
                'status': 'error', 'reply': reply_str,
                'pair_id': task.get('pair_id', ''), 'side': task.get('side', ''),
                'opponent_model': task.get('opponent_model', ''), 'opponent_provider': task.get('opponent_provider', ''),
                'aimux_origin_code': task.get('aimux_origin_code', ''),
                'query': task.get('query', ''),
            }
            if task.get(SOURCE_MODELS_EXCEL_KEY):
                err[SOURCE_MODELS_EXCEL_KEY] = task[SOURCE_MODELS_EXCEL_KEY]
            return err

        reasoning_str = reasoning if isinstance(reasoning, str) else (str(reasoning) if reasoning else '') or ''
        out = {
            'qid': task['qid'], 'model': task['model_name'], 'provider': task['provider'],
            'reply': reply_str, 'reasoning': reasoning_str,
            'reply_len': len(reply_str),
            'reasoning_len': len(reasoning_str),
            'finish_reason': finish_reason,
            'enable_thinking': task.get('enable_thinking', False),
            'status': 'ok',
            'timestamp': pd.Timestamp.now(),
            'pair_id': task.get('pair_id', ''),
            'side': task.get('side', ''),
            'opponent_model': task.get('opponent_model', ''),
            'opponent_provider': task.get('opponent_provider', ''),
            'query': task.get('query', ''),
        }
        if task.get('aimux_origin_code'):
            out['aimux_origin_code'] = task.get('aimux_origin_code')
        if 'session_id' in task:
            out['session_id'] = task.get('session_id', '')
            out['turn_id'] = task.get('turn_id', '')
            out['history_context'] = task.get('history_context', '')
        return out

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_task, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="🔄 生成进度", ncols=100):
            try:
                result = future.result()
                if result is None:
                    skipped_count += 1
                else:
                    if result.get('status') == 'ok':
                        new_results.append(result)
                        existing_keys.add((str(result['qid']), str(result['model'])))
                    else:
                        failed_count += 1
                        row_fail = {'provider': result.get('provider', ''), 'model': result.get('model', '')}
                        oc = result.get('aimux_origin_code')
                        if oc:
                            row_fail['aimux_origin_code'] = oc
                        failed_models_for_table.append(row_fail)
                        err_reply = str(result.get('reply', '') or '')
                        excel_p = _resolve_models_excel_for_mark(result, models_excel_path, models_excel_paths)
                        if excel_p and not is_transient_gateway_error(err_reply):
                            key = (
                                excel_p,
                                row_fail.get('provider', ''),
                                row_fail.get('model', ''),
                                row_fail.get('aimux_origin_code', '') or '',
                            )
                            if key not in marked_failed_for_excel:
                                if mark_single_model_unavailable(
                                    excel_p,
                                    row_fail.get('provider', ''),
                                    row_fail.get('model', ''),
                                    aimux_origin_code=row_fail.get('aimux_origin_code'),
                                ):
                                    marked_failed_for_excel.add(key)
                completed += 1
                if checkpoint_interval > 0 and completed % checkpoint_interval == 0:
                    merged = results + new_results
                    wide = _long_side_records_to_wide_df(merged)
                    if _save_paired_wide_outputs(merged, output_excel, matches_excel, qid_to_query=qid_to_query):
                        tqdm.write(f"💾 检查点: 已保存 {len(wide)} 场对战（每场一行）")
            except Exception as e:
                tqdm.write(f"❌ 任务失败: {e}")
                failed_count += 1
                completed += 1

    results.extend(new_results)
    merged_final = results
    _save_paired_wide_outputs(merged_final, output_excel, matches_excel, qid_to_query=qid_to_query)
    df_final = _long_side_records_to_wide_df(merged_final)
    if qid_to_query and not df_final.empty:
        df_final = _wide_df_apply_query_map(df_final, qid_to_query)
    if df_final.empty and os.path.exists(output_excel):
        df_final = pd.read_excel(output_excel)

    print(f"\n{'=' * 60}")
    print("✅ 组队回复生成完成!")
    print(f"{'=' * 60}")
    print(f"  按侧 API 条数: {len(merged_final)}  本场新增成功侧: {len(new_results)}")
    print(f"  成功侧: {sum(1 for r in merged_final if r.get('status') == 'ok')}  失败: {failed_count}  跳过: {skipped_count}")
    print(f"  对战行数（宽表）: {len(df_final)}  replies: {output_excel}")
    if matches_excel and str(matches_excel).strip() and os.path.abspath(str(matches_excel).strip()) != os.path.abspath(output_excel):
        print(f"  matches: {matches_excel}")
    print(f"{'=' * 60}\n")
    return df_final, failed_models_for_table
