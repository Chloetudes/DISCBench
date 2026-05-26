import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from datetime import datetime, timedelta

# 标记该行模型来自哪张 Excel（用于失败时写回可用状态列）
SOURCE_MODELS_EXCEL_KEY = "_source_models_excel"


def _load_models_table(path: str) -> Tuple[str, str, pd.DataFrame]:
    """
    读取模型清单 Excel，返回 (格式, 数据 sheet 名, DataFrame)。
    格式：idealab（展示名称/api_model_id/来源）或 aimux_catalog（模型名称 + 供应商）。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型清单不存在: {path}")

    xls = pd.ExcelFile(path)
    sheet = "idealab_models" if "idealab_models" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet)
    cols = set(df.columns)
    if {"展示名称", "api_model_id", "来源"}.issubset(cols):
        return "idealab", sheet, df
    if "模型名称" in cols:
        if "供应商" not in cols:
            raise ValueError(f"Aimux 目录表需含「供应商」列（原厂代码）: {path}")
        return "aimux_catalog", sheet, df
    raise ValueError(
        f"无法识别模型表格式: {path}。需 idealab 列（展示名称,api_model_id,来源）"
        f"或 Aimux 列（模型名称,供应商）。"
    )


def detect_models_table_format(path: str) -> str:
    """返回模型表格式标识：idealab 或 aimux_catalog。"""
    fmt, _sheet, _df = _load_models_table(path)
    return fmt


def _fallback_routify_provider_for_modelrouter(model_name: str) -> str:
    """
    表里写 modelRouter 但未在 MODEL_PROVIDER_MAPPING 命中时的兜底网关。
    Routify 多走 OpenAI 兼容协议，多数第三方 id 落在 routify_gpt；Claude/Gemini/GPT-5.2 等单独分流。
    """
    m = str(model_name).strip().lower()
    if any(k in m for k in ("claude", "opus-4", "sonnet", "haiku")):
        return "routify_claude"
    if "gemini" in m or "gemma" in m:
        return "routify_gemini"
    if "gpt-5" in m and ("chat" in m or m.endswith("latest") or "codex" in m):
        return "routify_gpt_responses"
    if "codex" in m and "gpt" in m:
        return "routify_gpt_responses"
    return "routify_gpt"


def _normalize_idealab_source(raw_provider, model_name: str) -> str:
    """
    兼容旧 idealab_models 表里把来源写成泛化名 `modelRouter` / `router` 的情况。
    此时按模型名回落到 config.MODEL_PROVIDER_MAPPING 中的真实 provider（如 routify_gpt）；
    若映射中无该 model id，则按模型名字符串启发式落到 routify_*，避免留下非法名「modelRouter」。
    """
    provider = str(raw_provider or "").strip()
    if provider.lower() not in (
        "modelrouter",
        "model_router",
        "router",
        "routify",  # 表里写「走 Routify 网关、按模型名选协议」时与 modelRouter 同义
    ):
        return provider
    try:
        from config import get_provider_for_model
        return get_provider_for_model(str(model_name).strip()).name
    except Exception:
        try:
            from config import get_provider

            guess = _fallback_routify_provider_for_modelrouter(model_name)
            get_provider(guess)
            return guess
        except Exception:
            return provider


def _str_to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("是", "yes", "y", "true", "1")


def _load_models_from_excel_single(
    path: str,
    provider: Optional[str] = None,
    only_available: bool = False,
) -> List[Dict]:
    """从单个 Excel 加载模型行（不含 _source_models_excel 标记）。"""
    fmt, _sheet, df = _load_models_table(path)

    if fmt == "idealab":
        if provider:
            df = df[df["来源"] == provider]
    else:
        if provider:
            pv = str(provider).strip().lower()
            df = df[df["供应商"].astype(str).str.strip().str.lower() == pv]

    if only_available and "可用状态" in df.columns:
        mask = df["可用状态"].astype(str).str.strip()
        df = df[mask == "是"]

    if df.empty:
        return []

    configs: List[Dict] = []

    if fmt == "idealab":
        has_thinking = "Thinking" in df.columns
        for _, row in df.iterrows():
            model_name = str(row["api_model_id"]).strip()
            if not model_name:
                continue
            provider_name = _normalize_idealab_source(row.get("来源", ""), model_name)
            cfg: Dict = {
                "provider": provider_name,
                "model": model_name,
                "enable_thinking": _str_to_bool(row["Thinking"]) if has_thinking else False,
            }
            display_name = str(row.get("展示名称", "")).strip()
            if display_name:
                cfg["展示名称"] = display_name
            configs.append(cfg)
        return configs

    # aimux_catalog
    has_thinking = "Thinking" in df.columns
    for _, row in df.iterrows():
        model_name = str(row["模型名称"]).strip()
        origin = str(row.get("供应商", "")).strip()
        if not model_name or not origin:
            continue
        cfg = {
            "provider": "aimux",
            "model": model_name,
            "aimux_origin_code": origin,
            "enable_thinking": _str_to_bool(row["Thinking"]) if has_thinking else False,
        }
        desc_raw = row.get("描述")
        desc = "" if pd.isna(desc_raw) else str(desc_raw).strip()
        cfg["展示名称"] = desc if desc else model_name
        configs.append(cfg)

    return configs


DEFAULT_JUDGE_FAMILY_KEYWORDS = ("gpt", "claude", "gemini")


def _model_matches_judge_families(model_name: str, families: Sequence[str]) -> bool:
    m = str(model_name or "").lower()
    return any(str(f).lower() in m for f in families if str(f).strip())


def load_judge_family_models_from_excel(
    path: Union[str, Sequence[str], None],
    families: Optional[Sequence[str]] = None,
    only_available: bool = False,
) -> List[Dict]:
    """
    从模型表（可多文件合并）加载 GPT / Claude / Gemini 系列（按模型 id 子串），
    保留各渠道 provider（idealab、routify_*、aimux 等），不限于单一来源。
    """
    kw = tuple(families or DEFAULT_JUDGE_FAMILY_KEYWORDS)
    all_rows = load_models_from_excel(path, provider=None, only_available=only_available)
    return [c for c in all_rows if _model_matches_judge_families(c.get("model"), kw)]


def load_models_from_excel(
    path: Union[str, Sequence[str], None],
    provider: Optional[str] = None,
    only_available: bool = False,
) -> List[Dict]:
    """
    从模型表格加载待测模型列表。支持两种格式：
    - idealab：列 展示名称 / api_model_id / 来源
    - Aimux 目录（如 data/models.xlsx）：列 模型名称 / 供应商（原厂代码，用于 base_url）

    path 可为单个文件路径，或路径列表（按顺序合并；同一 provider+model(+aimux_origin) 只保留首次出现）。
    每项会附带 ``_source_models_excel``：该行来源的 Excel 绝对路径，供写回可用状态与失败标记。

    参数：
    - provider: idealab 时过滤「来源」；Aimux 目录时过滤「供应商」（如 openai）
    - only_available: 为 True 时，仅加载「可用状态」为“是”的行（若表中有该列）
    """
    if path is None:
        return []
    if isinstance(path, (list, tuple)):
        out: List[Dict] = []
        seen: set = set()
        for raw_p in path:
            p = str(raw_p or "").strip()
            if not p:
                continue
            chunk = _load_models_from_excel_single(p, provider=provider, only_available=only_available)
            for c in chunk:
                oc = str(c.get("aimux_origin_code", "") or "").strip().lower()
                key = (
                    str(c.get("provider", "")).strip().lower(),
                    str(c.get("model", "")).strip(),
                    oc,
                )
                if key in seen:
                    continue
                seen.add(key)
                c2 = dict(c)
                c2[SOURCE_MODELS_EXCEL_KEY] = os.path.abspath(p) if not os.path.isabs(p) else p
                out.append(c2)
        return out

    p1 = str(path).strip()
    if not p1:
        return []
    chunk = _load_models_from_excel_single(p1, provider=provider, only_available=only_available)
    abs_p = os.path.abspath(p1) if not os.path.isabs(p1) else p1
    for c in chunk:
        c[SOURCE_MODELS_EXCEL_KEY] = abs_p
    return chunk


def try_use_cached_availability_multi(
    paths: List[str],
    model_configs: List[Dict],
    max_age_days: int = 14,
    provider: Optional[str] = None,
) -> Tuple[bool, Optional[pd.DataFrame]]:
    """
    多表场景：每张表分别判断是否可走「可用状态 + 最后测试时间」缓存；
    仅当**每一张有模型行的表**都返回可跳过时，才合并为 cached_df 并跳过 API 探测。
    """
    if not paths or not model_configs:
        return False, None
    if len(paths) == 1:
        return try_use_cached_availability(
            paths[0], model_configs, max_age_days=max_age_days, provider=provider
        )
    parts: List[pd.DataFrame] = []
    for p in paths:
        subset = [c for c in model_configs if c.get(SOURCE_MODELS_EXCEL_KEY) == p]
        if not subset:
            continue
        ok, df_p = try_use_cached_availability(p, subset, max_age_days=max_age_days, provider=provider)
        if not ok or df_p is None or df_p.empty:
            return False, None
        df2 = df_p.copy()
        df2[SOURCE_MODELS_EXCEL_KEY] = p
        parts.append(df2)
    if not parts:
        return False, None
    return True, pd.concat(parts, ignore_index=True)


def update_availability_for_test_results(
    paths: List[str],
    test_results: pd.DataFrame,
    provider: Optional[str] = None,
) -> None:
    """
    将测试结果写回 Excel「可用状态 / 最后测试时间」。
    若 test_results 含 ``_source_models_excel`` 列（多表合并探测），按路径拆分后分别写回对应文件。
    """
    if test_results is None or test_results.empty:
        return
    col = SOURCE_MODELS_EXCEL_KEY
    if col not in test_results.columns or bool(test_results[col].isna().all()):
        if paths:
            update_availability_in_excel(paths[0], test_results, provider=provider)
        return
    for p in paths:
        sub = test_results[test_results[col].astype(str) == str(p)]
        if sub.empty:
            continue
        update_availability_in_excel(
            p, sub.drop(columns=[col], errors="ignore"), provider=provider
        )


def mark_failed_models_on_excel_paths(
    paths: List[str],
    replies_df: Optional[pd.DataFrame] = None,
    provider: Optional[str] = None,
    blacklisted_models: Optional[list] = None,
) -> int:
    """对多张模型表依次尝试标记失败模型（无匹配行则跳过）。"""
    total = 0
    for p in paths:
        total += mark_failed_models_from_replies(
            p, replies_df=replies_df, provider=provider, blacklisted_models=blacklisted_models
        )
    return total


def _cfg_matches_excel_provider_filter(cfg: Dict, provider: Optional[str]) -> bool:
    if not provider or not str(provider).strip():
        return True
    p = str(provider).strip()
    if str(cfg.get("provider", "")).strip() == "aimux" and str(cfg.get("aimux_origin_code", "")).strip():
        return str(cfg["aimux_origin_code"]).strip().lower() == p.lower()
    return str(cfg.get("provider", "")).strip() == p


def _cached_availability_row_for_config(
    path: str,
    cfg: Dict,
    max_age_days: int = 14,
    provider: Optional[str] = None,
) -> Optional[Dict]:
    """
    单条模型：若 Excel 中「最后测试时间」在 max_age_days 内，返回与 test_all_models 同结构的 dict；
    否则返回 None（需 API 探测）。
    """
    if not cfg or not path or not os.path.exists(path):
        return None
    if not _cfg_matches_excel_provider_filter(cfg, provider):
        return None
    fmt, _sheet, df = _load_models_table(path)
    if "可用状态" not in df.columns or "最后测试时间" not in df.columns:
        return None

    prov = str(cfg.get("provider", "")).strip()
    model_id = str(cfg.get("model", "")).strip()
    if fmt == "idealab":
        src_norm = df.apply(
            lambda row: _normalize_idealab_source(row.get("来源", ""), row.get("api_model_id", "")),
            axis=1,
        )
        mask = (src_norm.astype(str).str.strip() == prov) & (
            df["api_model_id"].astype(str).str.strip() == model_id
        )
    else:
        oc = str(cfg.get("aimux_origin_code", "")).strip()
        if not oc:
            return None
        mask = (df["供应商"].astype(str).str.strip() == oc) & (
            df["模型名称"].astype(str).str.strip() == model_id
        )
    matched = df[mask]
    if matched.empty:
        return None
    row = matched.iloc[0]
    last_str = row.get("最后测试时间")
    if pd.isna(last_str) or not str(last_str).strip():
        return None
    try:
        if isinstance(last_str, datetime):
            last_dt = last_str
        else:
            last_dt = pd.to_datetime(last_str)
        if datetime.now() - last_dt.to_pydatetime() > timedelta(days=max_age_days):
            return None
    except Exception:
        return None

    avail_str = str(row.get("可用状态", "")).strip()
    available = avail_str == "是"
    out = {
        "provider": prov,
        "model": model_id,
        "available": available,
        "response_time": None,
        "error": None if available else "（使用表格缓存状态）",
        "enable_thinking": _str_to_bool(row.get("Thinking", False)) if "Thinking" in df.columns else False,
    }
    if fmt == "idealab":
        out["展示名称"] = str(row.get("展示名称", "")).strip()
    else:
        desc_raw = row.get("描述")
        desc = "" if pd.isna(desc_raw) else str(desc_raw).strip()
        out["展示名称"] = desc if desc else model_id
        out["aimux_origin_code"] = str(cfg.get("aimux_origin_code", "")).strip()
    if cfg.get(SOURCE_MODELS_EXCEL_KEY):
        out[SOURCE_MODELS_EXCEL_KEY] = cfg[SOURCE_MODELS_EXCEL_KEY]
    return out


def split_configs_by_excel_cache(
    model_configs: List[Dict],
    max_age_days: int = 14,
    provider: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    按条拆分：表格缓存仍有效 → 进入 cached_df；否则进入待 API 探测列表。
  用于裁判选型：14 天内已测过的 GPT/Claude/Gemini 不重复打 API。
    """
    cached_rows: List[Dict] = []
    to_probe: List[Dict] = []
    for cfg in model_configs:
        path = cfg.get(SOURCE_MODELS_EXCEL_KEY) or ""
        row = _cached_availability_row_for_config(path, cfg, max_age_days, provider) if path else None
        if row is not None:
            cached_rows.append(row)
        else:
            to_probe.append(cfg)
    cached_df = pd.DataFrame(cached_rows) if cached_rows else pd.DataFrame()
    return cached_df, to_probe


def try_use_cached_availability(
    path: str,
    model_configs: List[Dict],
    max_age_days: int = 14,
    provider: Optional[str] = None,
) -> Tuple[bool, Optional[pd.DataFrame]]:
    """
    若表格中所有待测模型的「最后测试时间」距当前不足 max_age_days 天，则返回 (True, cached_df)，
    可直接用表格记录的可用状态，跳过 API 检测。否则返回 (False, None) 表示需要重新跑可用性测试。
    """
    if not model_configs or not os.path.exists(path):
        return False, None
    rows_out = []
    for cfg in model_configs:
        if not _cfg_matches_excel_provider_filter(cfg, provider):
            continue
        row = _cached_availability_row_for_config(path, cfg, max_age_days, provider)
        if row is None:
            return False, None
        rows_out.append(row)
    if not rows_out:
        return False, None
    return True, pd.DataFrame(rows_out)


def update_availability_in_excel(path: str,
                                 test_results: pd.DataFrame,
                                 provider: Optional[str] = None) -> None:
    """
    将模型可用性测试结果写回表格。

    - idealab：匹配键 (来源, api_model_id) 对应测试结果 (provider, model)。
    - Aimux 目录：匹配键 (供应商, 模型名称) 对应 (aimux_origin_code, model)；provider 为 aimux。
    - provider 参数不为空时：idealab 按测试结果里的 provider 过滤；Aimux 目录按 aimux_origin_code（原厂）过滤。
    - 写回列：「可用状态」「最后测试时间」（若无则新增）。
    """
    if test_results is None or test_results.empty:
        return

    fmt, data_sheet, df = _load_models_table(path)

    results = test_results.copy()
    results["provider"] = results["provider"].astype(str).str.strip()
    results["model"] = results["model"].astype(str).str.strip()
    if provider:
        p = str(provider).strip().lower()
        if fmt == "aimux_catalog" and "aimux_origin_code" in results.columns:
            results = results[
                (results["provider"].str.lower() == "aimux")
                & (results["aimux_origin_code"].astype(str).str.strip().str.lower() == p)
            ]
        else:
            results = results[results["provider"].str.lower() == p]

    availability_map: Dict[Tuple[str, str], bool] = {}
    for _, row in results.iterrows():
        prov = str(row["provider"]).strip().lower()
        mid = str(row["model"]).strip()
        if prov == "aimux" and "aimux_origin_code" in results.columns:
            raw_oc = row.get("aimux_origin_code")
            if pd.notna(raw_oc) and str(raw_oc).strip():
                oc = str(raw_oc).strip().lower()
                availability_map[(oc, mid)] = bool(row["available"])
                continue
        availability_map[(prov, mid)] = bool(row["available"])
    if not availability_map:
        return

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if "可用状态" not in df.columns:
        df["可用状态"] = ""
    if "最后测试时间" not in df.columns:
        df["最后测试时间"] = ""

    if fmt == "idealab":
        for idx, row in df.iterrows():
            mid = str(row["api_model_id"]).strip()
            src = _normalize_idealab_source(row["来源"], mid).strip().lower()
            key = (src, mid)
            if key in availability_map:
                available = availability_map[key]
                df.at[idx, "可用状态"] = "是" if available else "否"
                df.at[idx, "最后测试时间"] = now_str
    else:
        for idx, row in df.iterrows():
            src = str(row["供应商"]).strip().lower()
            mid = str(row["模型名称"]).strip()
            key = (src, mid)
            if key in availability_map:
                available = availability_map[key]
                df.at[idx, "可用状态"] = "是" if available else "否"
                df.at[idx, "最后测试时间"] = now_str

    xls = pd.ExcelFile(path)
    other_sheets: Dict[str, pd.DataFrame] = {}
    for name in xls.sheet_names:
        if name != data_sheet:
            other_sheets[name] = pd.read_excel(path, sheet_name=name)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=data_sheet)
        for name, odf in other_sheets.items():
            odf.to_excel(writer, index=False, sheet_name=name)


def mark_single_model_unavailable(
    models_excel_path: str,
    provider: str,
    model: str,
    aimux_origin_code: Optional[str] = None,
) -> bool:
    """
    立即将单个模型在表格中标记为不可用。用于首次失败时即时更新，即使用户中断运行也能生效。
    Aimux 目录表须传入 aimux_origin_code（与「供应商」一致）才能匹配行。
    返回是否成功写入。
    """
    if not models_excel_path or not os.path.exists(models_excel_path):
        return False
    row = {"provider": str(provider).strip(), "model": str(model).strip(), "available": False}
    if aimux_origin_code and str(aimux_origin_code).strip():
        row["aimux_origin_code"] = str(aimux_origin_code).strip()
    df_fail = pd.DataFrame([row])
    update_availability_in_excel(models_excel_path, df_fail, provider=None)
    return True


def mark_failed_models_from_replies(
    models_excel_path: str,
    replies_df: Optional[pd.DataFrame] = None,
    provider: Optional[str] = None,
    blacklisted_models: Optional[list] = None,
) -> int:
    """
    将调用失败的模型在表格中标记为不可用（可用状态=否）。
    来源：(1) 回复表中 status=='error' 的记录；(2) 本次运行被黑名单跳过的模型（连接/权限等首次失败即跳过）。
    下次加载 only_available=True 时，这些模型将不再出现在可选清单中。
    返回被标记为不可用的模型数量。
    """
    if not os.path.exists(models_excel_path):
        return 0
    to_mark = []
    # 1. 从回复表提取 status=='error' 的 (provider, model)，Aimux 多原厂时带 aimux_origin_code
    if replies_df is not None and not replies_df.empty and "status" in replies_df.columns and "provider" in replies_df.columns:
        failed = replies_df[replies_df["status"] == "error"]
        failed_with_provider = failed[failed["provider"].notna() & (failed["provider"].astype(str).str.strip() != "")]
        if not failed_with_provider.empty:
            dup_cols = ["provider", "model"]
            if "aimux_origin_code" in failed_with_provider.columns:
                dup_cols.append("aimux_origin_code")
            sub = failed_with_provider[dup_cols].drop_duplicates()
            for _, row in sub.iterrows():
                item = {"provider": str(row["provider"]).strip(), "model": str(row["model"]).strip()}
                if "aimux_origin_code" in row.index:
                    oc = row.get("aimux_origin_code")
                    if pd.notna(oc) and str(oc).strip():
                        item["aimux_origin_code"] = str(oc).strip()
                to_mark.append(item)
    # 2. 从黑名单提取（首次调用即失败被跳过的模型，无回复表记录）
    if blacklisted_models:
        for item in blacklisted_models:
            p = str(item.get("provider", "")).strip()
            m = str(item.get("model", "")).strip()
            if p and m:
                entry = {"provider": p, "model": m}
                oc = item.get("aimux_origin_code")
                if oc is not None and str(oc).strip():
                    entry["aimux_origin_code"] = str(oc).strip()
                to_mark.append(entry)
    if not to_mark:
        return 0
    df_fail = pd.DataFrame(to_mark)
    dedupe_cols = ["provider", "model"]
    if "aimux_origin_code" in df_fail.columns:
        dedupe_cols.append("aimux_origin_code")
    df_fail = df_fail.drop_duplicates(subset=dedupe_cols)
    if provider:
        p = str(provider).strip()
        if "aimux_origin_code" in df_fail.columns and not df_fail.empty:
            oc_col = df_fail["aimux_origin_code"].astype(str).str.strip().str.lower()
            prov_col = df_fail["provider"].astype(str).str.strip()
            df_fail = df_fail[
                (prov_col.str.lower() == p.lower())
                | ((prov_col.str.lower() == "aimux") & (oc_col == p.lower()))
            ]
        else:
            df_fail = df_fail[df_fail["provider"].astype(str).str.strip() == p]
    if df_fail.empty:
        return 0
    df_fail["available"] = False
    update_availability_in_excel(models_excel_path, df_fail, provider=provider)
    return len(df_fail)

