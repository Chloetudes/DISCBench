# -*- coding: utf-8 -*-
"""
从 instruction_quality_evaluation 阶段模型输出的 YAML 中解析约束，
按 sysprompt 固定权重与公式用 Python 重算百分制难度分，并与模型自报分比对。

公式（与 instruction_quality_evaluation.txt 一致）：
  分子 = Σ(约束难度分 × 约束权重)
  分母 = 流程步骤数 + 格式输出数 + 边界范围数 + 数量篇幅数（不含问题约束）
  百分制得分 = (分子 / 分母) × 10
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

# 与 sysprompt 2.1 一致
CONSTRAINT_TYPE_WEIGHTS: Dict[str, float] = {
    '教学约束': 0.5,
    '素材约束': 0.4,
    '流程步骤': 1.0,
    '格式输出': 0.7,
    '边界范围': 0.3,
    '数量篇幅': 0.3,
}

# 计入分母的有效执行约束类型
EFFECTIVE_DENOMINATOR_TYPES = frozenset({'流程步骤', '格式输出', '边界范围', '数量篇幅'})

_TASK_INTENT_TYPE_RE = re.compile(r"类型\s*[:：]\s*([^\n\r]+)")
_TASK_INTENT_HEAD_CHARS = 50

_TYPE_ALIASES = {
    '类型1': '教学约束', '类型2': '素材约束', '类型3': '流程步骤',
    '类型4': '格式输出', '类型5': '边界范围', '类型6': '数量篇幅',
}


def normalize_constraint_type(raw: Any) -> str:
    s = str(raw or '').strip()
    if not s:
        return ''
    if s in _TYPE_ALIASES:
        return _TYPE_ALIASES[s]
    for key in CONSTRAINT_TYPE_WEIGHTS:
        if key in s:
            return key
    return s


def _to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if isinstance(val, float) and val != val:  # NaN
            return None
        return float(val)
    s = str(val).strip()
    if not s or s.lower() in ('null', 'none', 'nan'):
        return None
    m = re.search(r'-?\d+(?:\.\d+)?', s)
    return float(m.group(0)) if m else None


def parse_task_intent_type_from_raw(
    text: str,
    *,
    head_chars: int = _TASK_INTENT_HEAD_CHARS,
) -> Optional[str]:
    """从 instruction_quality 模型输出中提取「任务意图 → 类型」。"""
    if not text or not str(text).strip():
        return None
    s = str(text)
    snippet = s[:head_chars] if head_chars and head_chars > 0 else s
    m = _TASK_INTENT_TYPE_RE.search(snippet)
    if not m and head_chars:
        m = _TASK_INTENT_TYPE_RE.search(s)
    if not m:
        return None
    t = m.group(1).strip()
    return t or None


def _extract_yaml_block(text: str) -> str:
    if not text:
        return ''
    s = str(text).strip()
    m = re.search(r'```(?:yaml|yml)?\s*([\s\S]*?)```', s, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 未闭合的 ```yaml 代码块（Excel 截断/export 常见）
    m_open = re.match(r'```(?:yaml|yml)?\s*([\s\S]*)', s, re.IGNORECASE)
    if m_open and ('约束列表:' in s or '评估结果:' in s):
        return m_open.group(1).strip()
    if '约束列表:' in s or '评估结果:' in s:
        return s
    return s


def _sanitize_yaml_block(block: str) -> str:
    """修复模型输出/Excel 导出中常见的 YAML 截断（如 null → nul）。"""
    if not block:
        return ''
    s = str(block)
    for key in ('问题', '问题类型', '依赖', '过滤原因'):
        s = re.sub(rf'(\b{re.escape(key)}\s*:\s*)nul\b', r'\1null', s)
    return s


def extract_constraints_from_raw_regex(text: str) -> List[Dict[str, Any]]:
    """YAML 解析失败时，从原始文本按约束块正则提取（仅用于计分，不提取百分制）。"""
    s = str(text or '')
    if not s.strip():
        return []
    blocks = re.split(r'(?=-\s*ID:\s*\S+)', s)
    out: List[Dict[str, Any]] = []
    for block in blocks:
        id_m = re.search(r'ID:\s*(\S+)', block)
        tm = re.search(r'类型:\s*([^\n]+)', block)
        dm = re.search(r'难度分:\s*([\d.]+)', block)
        if not tm or not dm:
            continue
        ctype = normalize_constraint_type(tm.group(1))
        if ctype not in CONSTRAINT_TYPE_WEIGHTS:
            continue
        prob_m = re.search(r'问题:\s*([^\n]+)', block)
        prob = prob_m.group(1).strip() if prob_m else None
        if prob and prob.lower() in ('nul', 'null'):
            prob = None
        row: Dict[str, Any] = {
            'ID': id_m.group(1) if id_m else None,
            '类型': ctype,
            '难度分': float(dm.group(1)),
            '问题': prob,
        }
        wm = re.search(r'权重:\s*([\d.]+)', block)
        if wm:
            row['权重'] = float(wm.group(1))
        out.append(row)
    return out


def _strip_ds_tags(val: Any) -> Optional[float]:
    """提取 <DS>...</DS> 或纯数字。"""
    if val is None:
        return None
    s = str(val).strip()
    m = re.search(r'<DS>\s*([^<]+?)\s*</DS>', s, re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    return _to_float(s)


def _merge_regex_constraints(parsed: Dict[str, Any], text: str) -> Dict[str, Any]:
    """约束列表为空时，用正则从原文补全（仍走 Python 公式计分）。"""
    if iter_constraints(parsed):
        return parsed
    cons = extract_constraints_from_raw_regex(text)
    if not cons:
        return parsed
    merged = dict(parsed)
    merged['约束列表'] = cons
    return merged


def parse_instruction_quality_yaml(text: str) -> Dict[str, Any]:
    """解析模型输出的 YAML；约束列表缺失时正则兜底。"""
    block = _sanitize_yaml_block(_extract_yaml_block(text))
    if not block:
        cons = extract_constraints_from_raw_regex(text)
        return {'约束列表': cons} if cons else {}
    if yaml is not None:
        try:
            data = yaml.safe_load(block)
            if isinstance(data, dict):
                return _merge_regex_constraints(data, text)
        except Exception:
            pass
    cons = extract_constraints_from_raw_regex(text)
    if cons:
        out: Dict[str, Any] = {'约束列表': cons}
        # 保留评估结果块供模型自报分比对，但不作为 difficulty_score 来源
        m_pct = re.search(r'百分制得分\s*:\s*(.+)', block)
        if m_pct:
            out.setdefault('评估结果', {}).setdefault('难度', {})['百分制得分'] = m_pct.group(1).strip()
        return out
    return {}


def _parse_instruction_quality_minimal(block: str) -> Dict[str, Any]:
    """无 PyYAML 时的最小兜底（仅保留模型自报分字段，不计入主难度分）。"""
    out: Dict[str, Any] = {'评估结果': {'难度': {}}}
    m = re.search(r'百分制得分\s*:\s*(.+)', block)
    if m:
        out['评估结果']['难度']['百分制得分'] = m.group(1).strip()
    return out


def _constraint_has_problem(c: Dict[str, Any]) -> bool:
    prob = c.get('问题')
    if prob is None:
        return False
    s = str(prob).strip().lower()
    return bool(s) and s not in ('null', 'none', '无', 'false', '否')


def iter_constraints(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = parsed.get('约束列表')
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return []


def compute_difficulty_score_py(
    parsed: Dict[str, Any],
    *,
    use_model_weights: bool = False,
) -> Dict[str, Any]:
    """
    按固定类型权重（默认）或模型填写的权重计算难度分。

    返回:
      numerator, denominator, difficulty_score_py, constraint_details, counts_by_type
    """
    constraints = iter_constraints(parsed)
    numerator = 0.0
    denominator = 0
    details: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {k: 0 for k in CONSTRAINT_TYPE_WEIGHTS}

    for c in constraints:
        if _constraint_has_problem(c):
            continue
        ctype = normalize_constraint_type(c.get('类型'))
        diff = _to_float(c.get('难度分'))
        if diff is None:
            continue
        if use_model_weights:
            weight = _to_float(c.get('权重'))
            if weight is None:
                weight = CONSTRAINT_TYPE_WEIGHTS.get(ctype, 0.0)
        else:
            weight = CONSTRAINT_TYPE_WEIGHTS.get(ctype, 0.0)
        weighted = diff * weight
        numerator += weighted
        if ctype in EFFECTIVE_DENOMINATOR_TYPES:
            denominator += 1
        if ctype in counts:
            counts[ctype] += 1
        details.append({
            'id': c.get('ID'),
            '类型': ctype,
            '难度分': diff,
            '权重': weight,
            '加权难度': round(weighted, 4),
        })

    score = round((numerator / denominator) * 10, 2) if denominator > 0 else None
    return {
        'numerator': round(numerator, 4),
        'denominator': denominator,
        'difficulty_score_py': score,
        'constraint_details': details,
        'counts_by_type': counts,
    }


def extract_model_reported_scores(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """从 YAML 评估结果块提取模型自报统计。"""
    eval_block = parsed.get('评估结果') or {}
    if not isinstance(eval_block, dict):
        eval_block = {}
    diff_block = eval_block.get('难度') or {}
    if not isinstance(diff_block, dict):
        diff_block = {}
    stats = eval_block.get('统计') or {}
    if not isinstance(stats, dict):
        stats = {}
    quality = eval_block.get('质量') or {}
    if not isinstance(quality, dict):
        quality = {}

    score_model = _strip_ds_tags(diff_block.get('百分制得分'))
    if score_model is None:
        score_model = _strip_ds_tags(diff_block.get('平均难度'))
        if score_model is not None:
            score_model = round(score_model * 10, 2)

    task = parsed.get('任务意图') or {}
    if not isinstance(task, dict):
        task = {}

    return {
        'difficulty_score_model': score_model,
        'model_total_weighted': _to_float(diff_block.get('总加权分')),
        'model_effective_count': _to_float(diff_block.get('有效约束数')),
        'model_avg_difficulty': _to_float(diff_block.get('平均难度')),
        'difficulty_grade_model': quality.get('等级'),
        'difficulty_desc_model': quality.get('难度描述'),
        'iq_qualified': quality.get('是否合格'),
        'iq_filter_reason': quality.get('过滤原因'),
        'L3': task.get('类型'),
        'task_objectivity': task.get('客观性'),
        'constraint_total': _to_float(stats.get('总约束数')),
    }


def score_to_grade(score: Optional[float]) -> str:
    if score is None:
        return ''
    if score <= 0:
        return 'E级'
    if score >= 80:
        return 'S级'
    if score >= 60:
        return 'A级'
    if score >= 40:
        return 'B级'
    if score >= 20:
        return 'C级'
    return 'D级'


def score_to_desc(score: Optional[float]) -> str:
    if score is None:
        return ''
    if score <= 0:
        return '不合格'
    if score >= 80:
        return '极难'
    if score >= 60:
        return '困难'
    if score >= 40:
        return '较难'
    if score >= 20:
        return '中等'
    return '简单'


def build_instruction_quality_row(
    raw_response: str,
    *,
    use_model_weights: bool = False,
) -> Dict[str, Any]:
    """单条：解析 + Python 计分 + 与模型分比对（difficulty_score 仅来自约束块公式）。"""
    parsed = parse_instruction_quality_yaml(raw_response)
    calc = compute_difficulty_score_py(parsed, use_model_weights=use_model_weights)
    reported = extract_model_reported_scores(parsed)

    score_py = calc['difficulty_score_py']
    parse_via = 'yaml' if iter_constraints(parsed) else ''
    if score_py is None:
        cons = extract_constraints_from_raw_regex(raw_response)
        if cons:
            calc = compute_difficulty_score_py({'约束列表': cons}, use_model_weights=use_model_weights)
            score_py = calc['difficulty_score_py']
            parsed = {'约束列表': cons, **({} if not parsed else parsed)}
            parse_via = 'regex'
    score_model = reported.get('difficulty_score_model')
    delta = None
    if score_py is not None and score_model is not None:
        delta = round(abs(score_py - score_model), 2)

    l3 = reported.get('L3') or parse_task_intent_type_from_raw(raw_response) or ''

    row = {
        'instruction_quality_raw': raw_response,
        'instruction_quality_parsed_json': json.dumps(parsed, ensure_ascii=False) if parsed else '',
        'difficulty_score': score_py,
        'difficulty_score_py': score_py,
        'difficulty_score_model': score_model,
        'difficulty_score_delta': delta,
        'difficulty_level': score_to_grade(score_py) or reported.get('difficulty_grade_model') or '',
        'difficulty_desc': score_to_desc(score_py) or reported.get('difficulty_desc_model') or '',
        'iq_numerator': calc['numerator'],
        'iq_denominator': calc['denominator'],
        'iq_constraint_json': json.dumps(calc['constraint_details'], ensure_ascii=False),
        'L3': l3,
        'iq_qualified': reported.get('iq_qualified'),
        'iq_filter_reason': reported.get('iq_filter_reason') or '',
        'iq_parse_ok': bool(iter_constraints(parsed)),
        'iq_status': 'scored_py' if score_py is not None else 'parse_failed',
    }
    if parse_via == 'regex' and score_py is not None:
        row['iq_error'] = 'YAML 解析失败; 已从约束块正则重算'
    row.update({f'iq_count_{k}': calc['counts_by_type'].get(k, 0) for k in CONSTRAINT_TYPE_WEIGHTS})
    return row


def merge_instruction_quality_into_questions(
    questions_path: str,
    quality_df,
    *,
    sheet_name=0,
    qid_col: str = 'qid',
) -> bool:
    """将指令质量列写回题目表（支持多 sheet，仅更新目标 sheet）。"""
    import os
    import pandas as pd

    iq_cols = [
        'instruction_quality_raw', 'difficulty_score', 'difficulty_score_py',
        'difficulty_score_model', 'difficulty_score_delta', 'difficulty_level',
        'difficulty_desc', 'iq_numerator', 'iq_denominator', 'iq_constraint_json',
        'instruction_quality_parsed_json', 'L3', 'iq_qualified', 'iq_filter_reason',
        'iq_parse_ok', 'iq_status', 'iq_error',
    ] + [f'iq_count_{k}' for k in CONSTRAINT_TYPE_WEIGHTS]

    merge_df = quality_df.copy()
    if qid_col not in merge_df.columns:
        merge_df[qid_col] = [f'row_{i:05d}' for i in range(len(merge_df))]
    merge_df[qid_col] = merge_df[qid_col].astype(str).str.strip()

    xf = pd.ExcelFile(questions_path)
    sheet_names = list(xf.sheet_names)
    if isinstance(sheet_name, str):
        target_sheet = sheet_name
    elif isinstance(sheet_name, int):
        target_sheet = sheet_names[sheet_name]
    else:
        target_sheet = sheet_names[0]

    sheets = {n: pd.read_excel(questions_path, sheet_name=n) for n in sheet_names}
    df = sheets[target_sheet]

    for c in iq_cols:
        if c in merge_df.columns:
            df = df.drop(columns=[c], errors='ignore')

    slim_cols = [c for c in iq_cols if c in merge_df.columns]
    slim = merge_df[[qid_col] + slim_cols + (['query'] if 'query' in merge_df.columns else [])].copy()

    has_qid = qid_col in df.columns and df[qid_col].notna().any()
    empty_qid = False
    if has_qid:
        eq = df[qid_col].astype(str).str.strip()
        empty_qid = eq.isin(('', 'nan', 'None')).all()
        has_qid = not empty_qid

    # 右表在「合并键」上重复会导致 pandas 笛卡尔积，行数暴涨（例如 1506 → 8000+）
    if has_qid and qid_col in slim.columns:
        n0 = len(slim)
        slim = slim.drop_duplicates(subset=[qid_col], keep='last')
        if len(slim) < n0:
            print(f"  ⚠️  instruction_quality 写回: 结果表重复 qid 已去重 {n0} → {len(slim)}")
    elif (not has_qid) and 'query' in df.columns and 'query' in slim.columns:
        n0 = len(slim)
        slim = slim.drop_duplicates(subset=['query'], keep='last')
        if len(slim) < n0:
            print(f"  ⚠️  instruction_quality 写回: 结果表重复 query 已去重 {n0} → {len(slim)}")

    if has_qid:
        df[qid_col] = df[qid_col].astype(str).str.strip()
        right = slim.drop(columns=['query'], errors='ignore')
        df = df.merge(right, on=qid_col, how='left')
    elif 'query' in df.columns and 'query' in slim.columns:
        sq = slim.drop_duplicates(subset=['query'], keep='last')
        iq_only = [c for c in slim_cols if c in sq.columns]
        tmp_qid = f'{qid_col}__from_iq'
        if qid_col in sq.columns:
            right = sq[['query', qid_col] + iq_only].copy()
            right = right.rename(columns={qid_col: tmp_qid})
        else:
            right = sq[['query'] + iq_only].copy()
            tmp_qid = None
        n_left = len(df)
        df = df.merge(right, on='query', how='left')
        if len(df) > n_left * 1.05:
            print(
                f"  ⚠️  merge 后行数仍异常: {n_left} → {len(df)}，"
                f"请检查题目表是否含大量重复 query。"
            )
        if tmp_qid and tmp_qid in df.columns:
            if qid_col not in df.columns:
                df[qid_col] = df[tmp_qid]
            else:
                o = df[qid_col].astype(str).str.strip()
                mask = o.eq('') | o.isin(('nan', 'None'))
                df.loc[mask, qid_col] = df.loc[mask, tmp_qid]
            df = df.drop(columns=[tmp_qid], errors='ignore')
    elif len(df) == len(slim):
        for c in [qid_col] + slim_cols:
            if c in slim.columns:
                df[c] = slim[c].values
    else:
        print('⚠️  无法对齐题目行（缺 qid/query 且行数不一致），跳过写回')
        return False

    sheets[target_sheet] = df

    try:
        os.makedirs(os.path.dirname(questions_path) or '.', exist_ok=True)
        tmp = questions_path + '.tmp.xlsx'
        with pd.ExcelWriter(tmp, engine='openpyxl') as writer:
            for name, sdf in sheets.items():
                sdf.to_excel(writer, sheet_name=name, index=False)
        import os as _os
        _os.replace(tmp, questions_path)
        return True
    except Exception as e:
        print(f'⚠️  写回题目表失败: {e}')
        return False
