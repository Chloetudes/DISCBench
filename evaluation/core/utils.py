# -*- coding: utf-8 -*-
import hashlib
import os
import re
import math
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd


def safe_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return '\n'.join(str(item) for item in value if item is not None)
    if isinstance(value, float):
        try:
            if math.isnan(value):
                return ""
        except Exception:
            pass
    return str(value)


def reply_vendor_family(
    provider: Optional[str],
    model: Optional[str],
    aimux_origin_code: Optional[str] = None,
) -> str:
    """
    粗粒度「厂商/系族」标签，用于同一题凑多条成功回复时尽量跨系（避免同一主干的两枚小改版对比差过小）。
    按 provider 网关 + 模型 id 关键字推断；Aimux 按原厂代码区分。
    """
    p = safe_str(provider).strip().lower()
    m = safe_str(model).strip().lower()
    oc = ""
    if aimux_origin_code is not None and not (isinstance(aimux_origin_code, float) and pd.isna(aimux_origin_code)):
        oc = str(aimux_origin_code).strip().lower()
    if p == "aimux" and oc:
        return f"aimux:{oc}"
    if p in ("routify_gpt", "routify_gpt_responses"):
        return "openai"
    if p == "routify_claude":
        return "anthropic"
    if p == "routify_gemini":
        return "google"
    if p in ("bailian", "bailian_thinking"):
        return "alibaba_qwen"
    if p == "zhipu":
        return "zhipu"
    if p == "openrouter":
        return "openrouter"
    if p == "aiarena":
        return "aiarena"
    if any(k in m for k in ("claude", "opus-4", "sonnet", "haiku")):
        return "anthropic"
    if "gemini" in m or "gemma" in m:
        return "google"
    if any(k in m for k in ("gpt-", "gpt_", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if "qwen" in m:
        return "alibaba_qwen"
    if "deepseek" in m:
        return "deepseek"
    if "kimi" in m or "moonshot" in m:
        return "moonshot"
    if "glm" in m or "chatglm" in m:
        return "zhipu"
    if "grok" in m:
        return "xai"
    if "doubao" in m or "seedance" in m:
        return "bytedance"
    if "minimax" in m or "hailuo" in m:
        return "minimax"
    if "kling" in m:
        return "kling"
    if p:
        return f"misc:{p}"
    return "misc:unknown"


# 对比评测常用「八大家」：每族至多选一条，便于跨系对比（与 reply_vendor_family 互补，含 Aimux 广场中文供应商写法）
BENCHMARK_EIGHT_FAMILY_ORDER: Tuple[str, ...] = (
    "glm",
    "gpt",
    "claude",
    "gemini",
    "doubao",
    "qwen",
    "deepseek",
    "kimi",
)
BENCHMARK_EIGHT_FAMILY_LABEL_ZH: Dict[str, str] = {
    "glm": "智谱 GLM",
    "gpt": "OpenAI GPT",
    "claude": "Anthropic Claude",
    "gemini": "Google Gemini",
    "doubao": "字节豆包",
    "qwen": "阿里通义 Qwen",
    "deepseek": "DeepSeek",
    "kimi": "月之暗面 Kimi",
}


def _norm_oc(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip().lower()


def _model_newness_key(model_id: str) -> Tuple[int, int, int, str]:
    """
    同族内偏「新」的启发式排序键（越大越新）：优先解析 id 中的日期 yyyymmdd，其次 v 主版本号，再比 id 长度与字典序。
    """
    s = (model_id or "").strip()
    sl = s.lower()
    date_int = 0
    m = re.search(r"(20\d{2})[-_/]?(\d{2})[-_/]?(\d{2})", sl)
    if m:
        try:
            date_int = int(m.group(1) + m.group(2) + m.group(3))
        except ValueError:
            date_int = 0
    if not date_int:
        m2 = re.search(r"(20\d{6})\b", sl)
        if m2:
            try:
                date_int = int(m2.group(1))
            except ValueError:
                date_int = 0
    v_major = 0
    mv = re.search(r"[\._-]v(\d+)\b", sl)
    if mv:
        try:
            v_major = int(mv.group(1))
        except ValueError:
            v_major = 0
    return (date_int, v_major, len(sl), sl)


def eval_benchmark_eight_family(
    provider: Optional[str],
    model: Optional[str],
    aimux_origin_code: Optional[str] = None,
) -> Optional[str]:
    """
    将一行模型映射到八大家之一；无法可靠归类时返回 None（不参与 auto8 均衡选型）。
    """
    m = safe_str(model).strip().lower()
    oc = _norm_oc(aimux_origin_code)

    def from_model_name() -> Optional[str]:
        if not m:
            return None
        if "deepseek" in m:
            return "deepseek"
        if "qwen" in m or "通义" in (model or ""):
            return "qwen"
        if "kimi" in m or "moonshot" in m:
            return "kimi"
        if "doubao" in m or "豆包" in (model or ""):
            return "doubao"
        if "gemini" in m or "gemma" in m:
            return "gemini"
        if any(k in m for k in ("gpt-", "gpt_", "o1", "o3", "o4", "chatgpt")):
            return "gpt"
        if any(k in m for k in ("claude", "opus-4", "sonnet", "haiku")):
            return "claude"
        if "glm" in m or "chatglm" in m:
            return "glm"
        return None

    hit = from_model_name()
    if hit:
        return hit

    oc_map = {
        "openai": "gpt",
        "anthropic": "claude",
        "google": "gemini",
        "deepseek": "deepseek",
        "moonshot": "kimi",
        "alibaba": "qwen",
        "alibaba (china)": "qwen",
        "qwen": "qwen",
        "zhipu": "glm",
        "z.ai": "glm",
        "zai": "glm",
        "智谱": "glm",
        "字节": "doubao",
        "字节跳动": "doubao",
        "doubao": "doubao",
        "volcengine": "doubao",
        "azure openai": "gpt",
    }
    if oc:
        if oc in oc_map:
            return oc_map[oc]
        if "字节" in oc or "volc" in oc or "doubao" in oc:
            return "doubao"
        if "阿里" in oc or "alibaba" in oc or "qwen" in oc or "通义" in oc:
            return "qwen"
        if "智谱" in oc or "zhipu" in oc or oc.startswith("z."):
            return "glm"

    vf = reply_vendor_family(provider, model, aimux_origin_code)
    vf_map = {
        "zhipu": "glm",
        "openai": "gpt",
        "anthropic": "claude",
        "google": "gemini",
        "bytedance": "doubao",
        "alibaba_qwen": "qwen",
        "deepseek": "deepseek",
        "moonshot": "kimi",
    }
    if vf in vf_map:
        return vf_map[vf]
    if vf.startswith("aimux:"):
        inner = vf.split(":", 1)[1].strip()
        if inner in oc_map:
            return oc_map[inner]
        if "字节" in inner or "volc" in inner or "doubao" in inner:
            return "doubao"
        if "阿里" in inner or "alibaba" in inner or "qwen" in inner or "通义" in inner:
            return "qwen"
        if "智谱" in inner or "zhipu" in inner or inner.startswith("z."):
            return "glm"
        return from_model_name()
    return None


def pick_balanced_eight_families(available: pd.DataFrame) -> Tuple[List[dict], List[str]]:
    """
    在已通过探测的 available 表上，按八大家各选 1 条：同族内优先较新的 model id，其次响应更快。
    返回 (selected 行 dict 列表, 未覆盖族的中文说明列表)。
    """
    if available is None or available.empty:
        return [], list(BENCHMARK_EIGHT_FAMILY_LABEL_ZH.values())

    rows: List[Tuple[str, Tuple[int, int, int, str], float, dict]] = []
    for _, row in available.iterrows():
        d = row.to_dict()
        oc = d.get("aimux_origin_code")
        fam = eval_benchmark_eight_family(
            d.get("provider"), d.get("model"), oc
        )
        if not fam:
            continue
        rt = d.get("response_time")
        try:
            rt_f = float(rt) if rt is not None and not (isinstance(rt, float) and pd.isna(rt)) else 9999.0
        except (TypeError, ValueError):
            rt_f = 9999.0
        mid = str(d.get("model", "") or "")
        rows.append((fam, _model_newness_key(mid), rt_f, d))

    picked: Dict[str, dict] = {}
    for fam in BENCHMARK_EIGHT_FAMILY_ORDER:
        cands = [t for t in rows if t[0] == fam]
        if not cands:
            continue
        cands.sort(
            key=lambda t: (-t[1][0], -t[1][1], -t[1][2], t[1][3], t[2])
        )
        best = cands[0][3]
        picked[fam] = best

    order_keys = [k for k in BENCHMARK_EIGHT_FAMILY_ORDER if k in picked]
    selected = [picked[k] for k in order_keys]
    missing_zh = [
        BENCHMARK_EIGHT_FAMILY_LABEL_ZH[k]
        for k in BENCHMARK_EIGHT_FAMILY_ORDER
        if k not in picked
    ]
    return selected, missing_zh


def filter_available_exclude_patterns(
    available: pd.DataFrame,
    exclude_patterns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """按模型名/provider/展示名子串排除（如 deepseek）。"""
    if available is None or available.empty or not exclude_patterns:
        return available
    patterns = [str(p).strip().lower() for p in exclude_patterns if str(p).strip()]

    def _row_excluded(row: pd.Series) -> bool:
        blob = " ".join(
            str(row.get(c, "") or "")
            for c in ("provider", "model", "展示名称", "aimux_origin_code")
            if c in row.index
        ).lower()
        return any(p in blob for p in patterns)

    mask = ~available.apply(_row_excluded, axis=1)
    return available.loc[mask].reset_index(drop=True)


def pick_n_cross_vendor_models(
    available: pd.DataFrame,
    n: int = 2,
    *,
    prefer_random: bool = True,
) -> Tuple[List[dict], List[str]]:
    """
    从可用池选 n 个模型，尽量来自不同八大家族（用于双厂商对比回复）。
    prefer_random=True 时在族间随机打散，避免总是列表前几项（常为 deepseek）。
    """
    import random

    if available is None or available.empty or n < 1:
        return [], []

    picked, missing = pick_balanced_eight_families(available)
    if prefer_random and len(picked) > 1:
        random.shuffle(picked)
    if len(picked) >= n:
        return picked[:n], missing

    seen_fam: set = set()
    selected: List[dict] = list(picked)
    for r in selected:
        fam = eval_benchmark_eight_family(r.get("provider"), r.get("model"), r.get("aimux_origin_code"))
        seen_fam.add(fam or r.get("model"))

    for _, row in available.iterrows():
        if len(selected) >= n:
            break
        d = row.to_dict()
        fam = eval_benchmark_eight_family(d.get("provider"), d.get("model"), d.get("aimux_origin_code"))
        key = fam or str(d.get("model", "")).strip()
        if key in seen_fam:
            continue
        seen_fam.add(key)
        selected.append(d)
    return selected[:n], missing


def _norm_for_fingerprint(value) -> str:
    """Normalize value for fingerprint: strip, treat NaN/None as empty."""
    s = safe_str(value)
    return s.replace("\r\n", "\n").replace("\r", "\n").strip()


def compute_input_fingerprint(row: dict, columns: list) -> str:
    """
    根据指定列计算行的输入指纹，用于增量更新：仅当指纹变化时才重新生成/覆盖。
    row: 字典或可下标对象；columns: 列名列表（顺序固定）。
    """
    parts = []
    for col in columns:
        val = row.get(col) if isinstance(row, dict) else getattr(row, col, None)
        parts.append(f"{col}={_norm_for_fingerprint(val)}")
    text = "|".join(parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sanitize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", text)
    text = text.replace("\u200B", "").replace("\u200C", "").replace("\u200D", "").replace("\uFEFF", "")
    text = text.replace("\u00A0", " ").replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def safe_save_excel(df: pd.DataFrame, output_path: str, max_retries: int = 3) -> bool:
    for attempt in range(max_retries):
        try:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            temp_path = output_path + '.tmp.xlsx'
            df.to_excel(temp_path, index=False, engine='openpyxl')
            pd.read_excel(temp_path, nrows=1)

            if os.path.exists(output_path):
                os.replace(temp_path, output_path)
            else:
                os.rename(temp_path, output_path)
            return True
        except Exception as e:
            print(f"⚠️  保存失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1.0)
            else:
                temp_path = output_path + '.tmp.xlsx'
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                return False
    return False
