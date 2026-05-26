# -*- coding: utf-8 -*-
"""回复模型多来源路由：从 idealab / Aimux 表展开候选，探测后选用可用通路。"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config import get_provider, get_provider_for_model, aimux_provider_for_origin
from clients.openai_client import OAIClient
from .models_from_excel import (
    SOURCE_MODELS_EXCEL_KEY,
    _load_models_table,
    _normalize_idealab_source,
    _str_to_bool,
    detect_models_table_format,
)
from .core.blacklist import is_permission_error, is_transient_gateway_error

# 可由 pipeline 通过 configure_route_policy() 覆盖（默认不限制渠道，以探针/运行通为准）
_ROUTE_DEPRIORITIZE: List[str] = []
_ROUTE_DROP_IF_ALTERNATIVES: bool = False
_ROUTE_PROBE_SKIP: List[str] = []
_ROUTE_FORBID_IF_OTHERS_EXIST: List[str] = []
_SKIP_LOGICAL_ONLY_FORBIDDEN_ROUTES: bool = False
_MODEL_ALIASES: Dict[str, List[str]] = {}
_VERSION_FALLBACKS: Dict[str, List[str]] = {}
_ALLOW_VENDOR_FAMILY_ROUTE_MATCH: bool = False
_MAX_ROUTES_PER_LOGICAL: int = 6
_MAX_BACKUP_ROUTES: int = 2
_AUTO_FAMILY_MATCH: bool = False

# config 未写 provider 时注入的常用通路（与裁判 routify 对齐）
_BUILTIN_ROUTE_SEEDS: Dict[str, List[Dict[str, Any]]] = {
    "glm-5.1": [
        {"provider": "bailian", "model": "glm-5.1", "route_label": "builtin/bailian"},
        {"provider": "routify_gpt", "model": "glm-5.1", "route_label": "builtin/routify_gpt"},
        {"provider": "bailian", "model": "glm-5", "route_label": "builtin/bailian-glm5"},
        {"provider": "routify_gpt", "model": "glm-5", "route_label": "builtin/routify-glm5"},
    ],
    "deepseek-v3.2-thinking": [
        {"provider": "routify_gpt", "model": "deepseek-v3.2-thinking", "enable_thinking": True, "route_label": "builtin/routify-ds-thinking"},
        {"provider": "routify_gpt", "model": "deepseek-v3.2", "route_label": "builtin/routify-ds32"},
        {"provider": "routify_gpt", "model": "deepseek-v3.2-chat", "route_label": "builtin/routify-ds32-chat"},
    ],
    "gpt-5.4-2026-03-05": [
        {"provider": "routify_gpt", "model": "gpt-5.4-2026-03-05", "route_label": "builtin/routify_gpt54"},
    ],
    "claude-sonnet-4-6": [
        {"provider": "routify_claude", "model": "claude-sonnet-4-6-20260217", "route_label": "builtin/routify_claude"},
        {"provider": "routify_claude", "model": "claude-sonnet-4-6", "route_label": "builtin/routify_claude46"},
    ],
    "gemini-3.1-pro-preview": [
        {"provider": "routify_gemini", "model": "gemini-3.1-pro-preview", "route_label": "builtin/routify_gemini31"},
        {"provider": "routify_gemini", "model": "gemini-3-pro-preview", "route_label": "builtin/routify_gemini3"},
    ],
    "kimi-k2.5": [
        {"provider": "idealab", "model": "bailian/kimi-k2.5", "route_label": "builtin/idealab-kimi"},
        {"provider": "routify_gpt", "model": "kimi-k2.5", "route_label": "builtin/routify-kimi"},
    ],
    "qwen3.6-plus": [
        {"provider": "bailian", "model": "qwen3.6-plus-2026-04-02", "route_label": "builtin/bailian-qwen"},
        {"provider": "bailian", "model": "qwen3.6-plus", "route_label": "builtin/bailian-qwen-base"},
    ],
    "doubao-seed-2-0-pro": [
        {"provider": "routify_gpt", "model": "doubao-seed-1-6-251015", "route_label": "builtin/routify-doubao16"},
        {"provider": "routify_gpt", "model": "doubao-seed-1-8-251228", "route_label": "builtin/routify-doubao18"},
        {"provider": "routify_gpt", "model": "doubao-seed-2-0-pro-260215", "route_label": "builtin/routify-doubao20"},
        {"provider": "aiarena", "model": "doubao-seed-1-6-251015", "route_label": "builtin/aiarena-doubao16"},
        {"provider": "aimux", "model": "doubao-seed-1-6-251015", "aimux_origin_code": "字节跳动", "route_label": "builtin/aimux-doubao16"},
        {"provider": "aimux", "model": "doubao-seed-2-0-pro-260215", "aimux_origin_code": "字节跳动", "route_label": "builtin/aimux-doubao20"},
    ],
}


def configure_route_policy(
    *,
    deprioritize_providers: Optional[List[str]] = None,
    drop_deprioritized_if_alternatives: Optional[bool] = None,
    probe_skip_providers: Optional[List[str]] = None,
    forbid_providers_if_other_routes_exist: Optional[List[str]] = None,
    skip_logical_only_forbidden_routes: Optional[bool] = None,
    model_aliases: Optional[Dict[str, List[str]]] = None,
    version_fallbacks: Optional[Dict[str, List[str]]] = None,
    allow_vendor_family_route_match: Optional[bool] = None,
    max_routes_per_logical: Optional[int] = None,
    max_backup_routes: Optional[int] = None,
    auto_family_match: Optional[bool] = None,
) -> None:
    """设置回复路由策略；默认少量候选、版本偏新、探针择通后批量补齐。"""
    global _ROUTE_DEPRIORITIZE, _ROUTE_DROP_IF_ALTERNATIVES, _ROUTE_PROBE_SKIP
    global _ROUTE_FORBID_IF_OTHERS_EXIST, _SKIP_LOGICAL_ONLY_FORBIDDEN_ROUTES
    global _MODEL_ALIASES, _VERSION_FALLBACKS, _ALLOW_VENDOR_FAMILY_ROUTE_MATCH, _AUTO_FAMILY_MATCH
    global _MAX_ROUTES_PER_LOGICAL, _MAX_BACKUP_ROUTES
    if deprioritize_providers is not None:
        _ROUTE_DEPRIORITIZE = [str(p).strip().lower() for p in deprioritize_providers if str(p).strip()]
    if drop_deprioritized_if_alternatives is not None:
        _ROUTE_DROP_IF_ALTERNATIVES = bool(drop_deprioritized_if_alternatives)
    if probe_skip_providers is not None:
        _ROUTE_PROBE_SKIP = [str(p).strip().lower() for p in probe_skip_providers if str(p).strip()]
    if forbid_providers_if_other_routes_exist is not None:
        _ROUTE_FORBID_IF_OTHERS_EXIST = [
            str(p).strip().lower() for p in forbid_providers_if_other_routes_exist if str(p).strip()
        ]
    if skip_logical_only_forbidden_routes is not None:
        _SKIP_LOGICAL_ONLY_FORBIDDEN_ROUTES = bool(skip_logical_only_forbidden_routes)
    if version_fallbacks is not None:
        _VERSION_FALLBACKS = {}
        for k, vals in version_fallbacks.items():
            key = str(k).strip()
            if not key:
                continue
            _VERSION_FALLBACKS[key] = [str(v).strip() for v in vals if str(v).strip()]
    if allow_vendor_family_route_match is not None:
        _ALLOW_VENDOR_FAMILY_ROUTE_MATCH = bool(allow_vendor_family_route_match)
    if max_routes_per_logical is not None:
        _MAX_ROUTES_PER_LOGICAL = max(1, int(max_routes_per_logical))
    if max_backup_routes is not None:
        _MAX_BACKUP_ROUTES = max(0, int(max_backup_routes))
    if model_aliases is not None:
        _MODEL_ALIASES = {}
        for k, vals in model_aliases.items():
            key = str(k).strip()
            if not key:
                continue
            raw = [str(v).strip() for v in vals if str(v).strip()]
            valid = _filter_same_version_aliases(key, raw)
            if valid:
                _MODEL_ALIASES[key] = valid
    if auto_family_match is not None:
        _AUTO_FAMILY_MATCH = bool(auto_family_match)


def _provider_name_norm(provider: str) -> str:
    return str(provider or "").strip().lower()


def _is_deprioritized_provider(provider: str) -> bool:
    p = _provider_name_norm(provider)
    if p in _ROUTE_DEPRIORITIZE:
        return True
    return any(p.startswith(d + "_") or d in p for d in _ROUTE_DEPRIORITIZE)


def _is_forbidden_if_others_provider(provider: str) -> bool:
    p = _provider_name_norm(provider)
    if not _ROUTE_FORBID_IF_OTHERS_EXIST:
        return False
    if p in _ROUTE_FORBID_IF_OTHERS_EXIST:
        return True
    return any(p.startswith(d + "_") or d in p for d in _ROUTE_FORBID_IF_OTHERS_EXIST)


def _split_forbidden_routes(
    routes: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    primary: List[Dict[str, Any]] = []
    forbidden: List[Dict[str, Any]] = []
    for r in routes:
        if _is_forbidden_if_others_provider(r.get("provider", "")):
            forbidden.append(r)
        else:
            primary.append(r)
    return primary, forbidden


def logical_name_from_config(cfg: Dict[str, Any]) -> str:
    for k in ("logical_model", "展示名称", "display_name", "model"):
        v = cfg.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _normalize_model_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def _version_identity_key(name: str) -> str:
    """
    同一「版本」的身份键：只去掉渠道路径前缀、日期/构建号后缀。
    保留 pro / lite / thinking / preview 等——后缀不同即不同版本。
    """
    s = str(name or "").strip().lower()
    if not s:
        return ""
    s = s.split("/")[-1]
    s = re.sub(r"-\d{4}-\d{2}-\d{2}[a-z]?$", "", s)
    s = re.sub(r"-\d{8}[a-z]?$", "", s)
    s = re.sub(r"-\d{6,}$", "", s)
    return _normalize_model_token(s)


def _filter_same_version_aliases(logical: str, aliases: List[str]) -> List[str]:
    """丢弃与逻辑名不是同一版本的别名（如 gpt-5.4-pro 不能给 gpt-5.4-2026-03-05 用）。"""
    lk = _version_identity_key(logical)
    if not lk:
        return []
    out: List[str] = []
    for a in aliases:
        ak = _version_identity_key(a)
        if ak == lk:
            out.append(a)
        else:
            print(
                f"  ⚠️  别名已忽略（不同版本）: {logical} ↛ {a}"
                f"（身份键 {lk} vs {ak}）"
            )
    return out


def _match_tier(logical: str, display: str, api_id: str) -> Optional[int]:
    """
    匹配档位（越小越优先）：0=展示名/api_id 完全一致；1=去符号后一致；2=带厂商前缀路径一致。
    禁止 glm-5 误匹配 glm-5.1、gpt-5-0807 误匹配 gpt-5.4-2026-03-05 等「前缀+版本号」误伤。
    """
    ln = str(logical or "").strip().lower()
    if not ln:
        return None
    ln_norm = _normalize_model_token(ln)
    best: Optional[int] = None

    def _consider(s_raw: str) -> None:
        nonlocal best
        s = str(s_raw or "").strip().lower()
        if not s:
            return
        variants = [s]
        if "/" in s:
            variants.append(s.split("/")[-1])
        for s in variants:
            if s == ln:
                best = 0 if best is None else min(best, 0)
                return
            if _normalize_model_token(s) == ln_norm:
                best = 1 if best is None else min(best, 1)
                continue
            # 禁止短 id 当长 id 的前缀版本（glm-5 ⊂ glm-5.1）
            if len(s) < len(ln) and ln.startswith(s):
                tail = ln[len(s) :]
                if tail and tail[0] in "._-":
                    continue
            if len(ln) < len(s) and s.startswith(ln):
                tail = s[len(ln) :]
                if tail and tail[0] in "._-":
                    continue
            if "/" in str(s_raw) and s == ln:
                best = 2 if best is None else min(best, 2)

    _consider(display)
    _consider(api_id)
    return best


def _dedupe_terms(terms: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for t in terms:
        s = str(t or "").strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _search_terms_for_logical(
    logical: str,
    extra_aliases: Optional[List[str]] = None,
) -> List[str]:
    """逻辑名 → 同版本别名 → 版本降级（较新版本 id 优先）。"""
    from .core.utils import _model_newness_key

    terms = [logical]
    terms.extend(_MODEL_ALIASES.get(logical, []))
    if extra_aliases:
        terms.extend(_filter_same_version_aliases(logical, list(extra_aliases)))
    fallbacks = list(_VERSION_FALLBACKS.get(logical, []))
    fallbacks.sort(key=_model_newness_key, reverse=True)
    terms.extend(fallbacks)
    return _dedupe_terms(terms)


def _sort_routes_for_probe(routes: List[Dict[str, Any]], logical: str) -> List[Dict[str, Any]]:
    from .core.utils import _model_newness_key

    def key(r: Dict[str, Any]) -> Tuple[int, int, Tuple[int, int, int, str], int, str]:
        tier = int(r.get("match_tier", 9))
        avail = 0 if r.get("avail") else 1
        mid = str(r.get("model", ""))
        return (tier, avail, _model_newness_key(mid), _provider_priority(str(r.get("provider", ""))), mid)

    return sorted(routes, key=key)


def _vendor_family_match_tier(logical: str, display: str, api_id: str) -> Optional[int]:
    """同厂商不同版本：用于从表里捞可降级 api（档位低于精确同版本）。"""
    if not _ALLOW_VENDOR_FAMILY_ROUTE_MATCH:
        return None
    from .core.utils import eval_benchmark_eight_family

    lf = eval_benchmark_eight_family("", logical, None)
    if not lf:
        return None
    for raw in (api_id, display):
        rf = eval_benchmark_eight_family("", str(raw or ""), None)
        if rf and rf == lf:
            return 5
    return None


def _same_version_as_logical(logical: str, display: str, api_id: str) -> bool:
    lk = _version_identity_key(logical)
    if not lk:
        return False
    for raw in (api_id, display):
        if _version_identity_key(str(raw or "")) == lk:
            return True
    return False


def _best_match_tier(
    logical: str,
    display: str,
    api_id: str,
    extra_aliases: Optional[List[str]] = None,
) -> Optional[int]:
    """
    匹配档位：0=同版本身份（可跨渠道）；1=字符串完全一致；2=同版本别名表命中。
    不同后缀（pro/lite/thinking/…）身份键不同 → 不匹配。
    """
    if _same_version_as_logical(logical, display, api_id):
        return 0
    terms = _search_terms_for_logical(logical, extra_aliases)
    best: Optional[int] = None
    for term in terms:
        t = _match_tier(term, display, api_id)
        if t is not None:
            best = t if best is None else min(best, t)
    if best is not None:
        return best
    return None


def _route_from_idealab_row(row: pd.Series, excel_path: str) -> Optional[Dict[str, Any]]:
    mid = str(row.get("api_model_id", "")).strip()
    if not mid:
        return None
    src_raw = str(row.get("来源", "")).strip()
    provider = _normalize_idealab_source(src_raw, mid)
    route: Dict[str, Any] = {
        "provider": provider,
        "model": mid,
        "enable_thinking": _str_to_bool(row.get("Thinking")) if "Thinking" in row.index else False,
        "route_label": f"idealab表/{src_raw or provider}",
        "table_source": src_raw or provider,
        "avail": str(row.get("可用状态", "")).strip() == "是",
        SOURCE_MODELS_EXCEL_KEY: excel_path,
    }
    disp = str(row.get("展示名称", "")).strip()
    if disp:
        route["展示名称"] = disp
    return route


def _route_from_aimux_row(row: pd.Series, excel_path: str) -> Optional[Dict[str, Any]]:
    mid = str(row.get("模型名称", "")).strip()
    origin = str(row.get("供应商", "")).strip()
    if not mid or not origin:
        return None
    return {
        "provider": "aimux",
        "model": mid,
        "aimux_origin_code": origin,
        "enable_thinking": _str_to_bool(row.get("Thinking")) if "Thinking" in row.index else False,
        "route_label": f"aimux/{origin}",
        "table_source": f"aimux:{origin}",
        "avail": str(row.get("可用状态", "")).strip() == "是",
        SOURCE_MODELS_EXCEL_KEY: excel_path,
        "展示名称": mid,
    }


def _provider_priority(provider: str) -> int:
    p = _provider_name_norm(provider)
    if _ROUTE_DEPRIORITIZE and _is_deprioritized_provider(p):
        return 90
    if p == "idealab":
        return 0
    if p == "bailian":
        return 1
    if p.startswith("routify"):
        return 2
    if p == "aimux":
        return 3
    return 5


def _apply_route_policy(routes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """可选的渠道过滤；默认不过滤，保留全部候选供探针/运行择通。"""
    if not routes:
        return routes
    if _ROUTE_FORBID_IF_OTHERS_EXIST:
        primary, forbidden = _split_forbidden_routes(routes)
        if primary and forbidden:
            names = ", ".join(_ROUTE_FORBID_IF_OTHERS_EXIST)
            print(
                f"  ℹ️  存在非 {names} 路由，已移除 {len(forbidden)} 条 {names}（正式回复不走该渠道）"
            )
            routes = primary
        elif forbidden and not primary:
            names = ", ".join(_ROUTE_FORBID_IF_OTHERS_EXIST)
            print(
                f"  ⚠️  仅余 {names} 路由（{len(forbidden)} 条）；"
                f"{'本模型将跳过' if _SKIP_LOGICAL_ONLY_FORBIDDEN_ROUTES else '仍将使用 aimux 兜底'}"
            )
            routes = forbidden
    if _ROUTE_DROP_IF_ALTERNATIVES and _ROUTE_DEPRIORITIZE:
        primary = [r for r in routes if not _is_deprioritized_provider(r.get("provider", ""))]
        deprioritized = [r for r in routes if _is_deprioritized_provider(r.get("provider", ""))]
        primary_avail = [r for r in primary if r.get("avail")]
        if primary_avail:
            dropped = len(deprioritized)
            if dropped:
                dep_names = ", ".join(_ROUTE_DEPRIORITIZE)
                print(
                    f"  ℹ️  已有可用的 idealab/routify/bailian 路由，跳过 {dropped} 条置后来源（{dep_names}）"
                )
            routes = primary
        elif primary and deprioritized:
            print(
                f"  ℹ️  idealab/routify 等同版本路由在表中均「不可用」，仍保留 "
                f"{len(deprioritized)} 条 aimux 等兜底路由（排序靠后）"
            )
            routes = primary + deprioritized
    return _sort_routes(routes)


def _sort_routes(routes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(r: Dict[str, Any]) -> Tuple[int, int, int, str]:
        avail = 0 if r.get("avail") else 1
        tier = int(r.get("match_tier", 9))
        prov = _provider_priority(str(r.get("provider", "")))
        return (avail, tier, prov, str(r.get("model", "")))

    out: List[Dict[str, Any]] = []
    seen: set = set()
    for r in sorted(routes, key=key):
        k = _route_dedupe_key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _route_dedupe_key(r: Dict[str, Any]) -> Tuple[str, str]:
    """同 provider+api_model 只保留一条（避免表内重复行浪费探针）。"""
    return (
        str(r.get("provider", "")).lower(),
        str(r.get("model", "")).lower(),
    )


def _merge_route_lists(*lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for routes in lists:
        for r in routes or []:
            k = _route_dedupe_key(r)
            if k in seen:
                continue
            seen.add(k)
            out.append(dict(r))
    return out


def _builtin_routes_for_logical(logical: str, cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if cfg and cfg.get("provider") and cfg.get("model"):
        br = {
            "provider": cfg["provider"],
            "model": cfg["model"],
            "route_label": "config.json",
            "avail": True,
            "match_tier": 0,
            "fills_logical": logical,
        }
        if cfg.get("enable_thinking"):
            br["enable_thinking"] = True
        out.append(br)
    for seed in _BUILTIN_ROUTE_SEEDS.get(logical, []):
        r = dict(seed)
        r.setdefault("avail", True)
        r.setdefault("match_tier", 1)
        r["fills_logical"] = logical
        out.append(r)
    return out


def _collect_routes_for_search_term(
    term: str,
    fills_logical: str,
    models_excel_paths: List[str],
    extra_aliases: Optional[List[str]] = None,
    *,
    max_per_term: int = 2,
) -> List[Dict[str, Any]]:
    term = str(term or "").strip()
    if not term:
        return []
    routes: List[Dict[str, Any]] = []
    for path in models_excel_paths:
        if not path or not os.path.isfile(path):
            continue
        abspath = os.path.abspath(path)
        try:
            fmt, _sheet, df = _load_models_table(abspath)
        except Exception:
            continue
        if fmt == "idealab":
            hits: List[Dict[str, Any]] = []
            for _, row in df.iterrows():
                tier = _best_match_tier(term, row.get("展示名称"), row.get("api_model_id"), extra_aliases)
                if tier is None:
                    continue
                rt = _route_from_idealab_row(row, abspath)
                if not rt or not rt.get("avail"):
                    continue
                rt["match_tier"] = tier
                rt["fills_logical"] = fills_logical
                rt["match_term"] = term
                hits.append(rt)
            routes.extend(_sort_routes_for_probe(hits, fills_logical)[:max_per_term])
        else:
            hits = []
            for _, row in df.iterrows():
                tier = _best_match_tier(term, row.get("模型名称"), row.get("模型名称"), extra_aliases)
                if tier is None:
                    continue
                rt = _route_from_aimux_row(row, abspath)
                if not rt or not rt.get("avail"):
                    continue
                rt["match_tier"] = tier
                rt["fills_logical"] = fills_logical
                rt["match_term"] = term
                hits.append(rt)
            routes.extend(_sort_routes_for_probe(hits, fills_logical)[:max_per_term])
    return routes


def collect_routes_for_logical_name(
    logical: str,
    models_excel_paths: List[str],
    extra_aliases: Optional[List[str]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    logical = str(logical or "").strip()
    if not logical:
        return []
    terms = _search_terms_for_logical(logical, extra_aliases)
    routes: List[Dict[str, Any]] = _builtin_routes_for_logical(logical, cfg)
    for term in terms:
        routes.extend(
            _collect_routes_for_search_term(
                term, logical, models_excel_paths, extra_aliases, max_per_term=2,
            )
        )
    routes = _merge_route_lists(routes)
    routes = _apply_route_policy(routes)
    routes = _sort_routes_for_probe(routes, logical)[:_MAX_ROUTES_PER_LOGICAL]
    if routes:
        pick = routes[0]
        print(
            f"  ℹ️  「{logical}」精简候选 {len(routes)} 条"
            f"（优先 {pick.get('provider')}/{pick.get('model')}；"
            f"检索词 {len(terms)} 个）"
        )
    return routes


def expand_reply_configs_with_routes(
    base_configs: List[Dict[str, Any]],
    models_excel_paths: List[str],
) -> List[Dict[str, Any]]:
    """为每个逻辑模型名展开多来源 routes；无表匹配时保留原单条配置。"""
    expanded: List[Dict[str, Any]] = []
    for cfg in base_configs:
        logical = logical_name_from_config(cfg)
        per_aliases = cfg.get("aliases") or cfg.get("reply_aliases")
        routes = (
            collect_routes_for_logical_name(logical, models_excel_paths, per_aliases, cfg=cfg)
            if logical
            else _builtin_routes_for_logical(logical, cfg)
        )
        if not routes:
            print(f"  ⚠️  未找到模型「{logical or cfg}」的任何路由，仍保留配置占位（运行时再试）")
            routes = _builtin_routes_for_logical(logical, cfg) or [dict(cfg)]
        if _SKIP_LOGICAL_ONLY_FORBIDDEN_ROUTES and not any(
            not _is_forbidden_if_others_provider(r.get("provider", "")) for r in routes
        ):
            print(
                f"  ⏭️  「{logical}」仅剩禁用渠道路由，已跳过"
            )
            continue
        item = dict(cfg)
        item["logical_model"] = logical or str(cfg.get("model", "")).strip()
        item["展示名称"] = item.get("展示名称") or item["logical_model"]
        item["routes"] = routes
        expanded.append(item)
    return expanded


def resolve_provider_config_for_route(route: Dict[str, Any]):
    if (
        str(route.get("provider", "")).strip() == "aimux"
        and route.get("aimux_origin_code")
        and str(route.get("aimux_origin_code")).strip()
    ):
        return aimux_provider_for_origin(str(route["aimux_origin_code"]).strip())
    if route.get("provider"):
        return get_provider(str(route["provider"]).strip())
    return get_provider_for_model(str(route["model"]).strip())


def probe_single_route(
    route: Dict[str, Any],
    *,
    timeout: int = 60,
    test_prompt: str = "Reply with exactly: OK",
) -> Tuple[bool, str]:
    """短探针测一条路由是否可调用（鉴权/连通）。"""
    try:
        pc = resolve_provider_config_for_route(route)
        client = OAIClient(
            base_url=pc.base_url,
            api_key=pc.api_key,
            protocol=pc.protocol,
            auth_header=pc.auth_header,
            auth_prefix=pc.auth_prefix,
            extra_headers=pc.extra_headers,
            timeout=timeout,
        )
        kwargs = {
            "model": route["model"],
            "messages": [{"role": "user", "content": test_prompt}],
            "temperature": 0,
        }
        if route.get("enable_thinking"):
            kwargs["enable_thinking"] = True
        if hasattr(client, "chat_with_meta"):
            text, _, _ = client.chat_with_meta(**kwargs)
        else:
            text = client.chat(**kwargs)
        text = str(text or "")
        if text.startswith("<error"):
            if is_permission_error(text) or is_transient_gateway_error(text):
                return False, text[:200]
            return False, text[:200]
        return True, text[:80]
    except Exception as e:
        return False, str(e)[:200]


def _finalize_route_ladder(
    routes: List[Dict[str, Any]],
    chosen: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """探针主路由置顶，保留至多 _MAX_ROUTES_PER_LOGICAL 条供运行期依次切换。"""
    if not routes:
        return []
    cap = max(1, _MAX_ROUTES_PER_LOGICAL)
    if not chosen:
        return routes[:cap]
    rest = [r for r in routes if _route_dedupe_key(r) != _route_dedupe_key(chosen)]
    return [chosen] + rest[: cap - 1]


def probe_and_select_routes(
    configs: List[Dict[str, Any]],
    *,
    timeout: int = 60,
    test_prompt: Optional[str] = None,
    probe_skip_providers: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """按版本从新到旧探针，首个可用即定为主路由；未再全表扫描。"""
    prompt = (test_prompt or "Reply with exactly: OK").strip()
    skip_set = {
        str(p).strip().lower()
        for p in (probe_skip_providers if probe_skip_providers is not None else _ROUTE_PROBE_SKIP)
        if str(p).strip()
    }
    chosen: Dict[str, Dict[str, Any]] = {}
    print(f"\n{'=' * 60}")
    print(
        f"  回复路由探针（每模型 ≤{_MAX_ROUTES_PER_LOGICAL} 候选，通一条置顶；"
        f"运行期最多试 {_MAX_ROUTES_PER_LOGICAL} 条）"
    )
    print(f"{'=' * 60}")

    def _probe_until_ok(logical: str, routes: List[Dict], *, allow_skipped: bool) -> Optional[Dict[str, Any]]:
        for route in routes:
            prov = _provider_name_norm(route.get("provider"))
            if skip_set and prov in skip_set and not allow_skipped:
                continue
            label = route.get("route_label") or "?"
            mid = route.get("model")
            print(f"      {label} → {route.get('provider')} / {mid} ...", end=" ", flush=True)
            ok, msg = probe_single_route(route, timeout=timeout, test_prompt=prompt)
            if ok:
                print("✅")
                return route
            kind = "网关" if is_transient_gateway_error(msg) else ("鉴权" if is_permission_error(msg) else "失败")
            print(f"❌ ({kind}: {msg[:50]})")
        return None

    for cfg in configs:
        logical = logical_name_from_config(cfg)
        routes = list(cfg.get("routes") or [])
        if not logical or not routes:
            continue
        print(f"\n  ▶ {logical}")
        ok_route = _probe_until_ok(logical, routes, allow_skipped=False)
        if not ok_route and skip_set:
            ok_route = _probe_until_ok(logical, routes, allow_skipped=True)
        ladder = _finalize_route_ladder(routes, ok_route)
        cfg["routes"] = ladder
        if ok_route:
            chosen[logical] = ok_route
            backups = len(ladder) - 1
            print(
                f"    ✅ 主路由 {ok_route.get('provider')}/{ok_route.get('model')}"
                + (f"  备降 {backups} 条" if backups else "")
            )
        else:
            print(f"    ⚠️  探针均未通，生成时仍按 {len(ladder)} 条备降顺序试")
    print(f"{'=' * 60}\n")
    return chosen


def sort_reply_configs_for_run(
    configs: List[Dict[str, Any]],
    chosen: Dict[str, Dict[str, Any]],
    run_order: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """探针通过者优先，再按 reply_model_run_order。"""
    order_index: Dict[str, int] = {}
    if run_order:
        for i, name in enumerate(run_order):
            key = str(name).strip()
            if key:
                order_index[key.lower()] = i

    def _tier(cfg: Dict[str, Any]) -> int:
        logical = logical_name_from_config(cfg) or ""
        if logical in chosen:
            return 0
        if cfg.get("routes"):
            return 1
        return 2

    def _sort_key(cfg: Dict[str, Any]) -> Tuple[int, int, str]:
        logical = logical_name_from_config(cfg) or ""
        return (
            _tier(cfg),
            order_index.get(logical.lower(), 999),
            logical,
        )

    out = sorted(configs, key=_sort_key)
    smooth = [logical_name_from_config(c) for c in out if _tier(c) == 0]
    if smooth:
        print(f"  ℹ️  优先生成（探针已通过）: {', '.join(smooth)}")
    deferred = [logical_name_from_config(c) for c in out if _tier(c) >= 2]
    if deferred:
        print(f"  ℹ️  靠后（无候选路由）: {', '.join(deferred)}")
    later = [logical_name_from_config(c) for c in out if _tier(c) == 1]
    if later:
        print(f"  ℹ️  探针未过、生成时再试: {', '.join(later)}")
    return out


def apply_chosen_routes_to_configs(
    configs: List[Dict[str, Any]],
    chosen: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """routes 已在探针阶段收成主路由+少量备降；此处仅标记 chosen_route。"""
    out: List[Dict[str, Any]] = []
    for cfg in configs:
        c = dict(cfg)
        logical = logical_name_from_config(cfg)
        c["chosen_route"] = chosen.get(logical)
        c["routes"] = list(c.get("routes") or [])
        out.append(c)
    return out
