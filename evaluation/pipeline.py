# -*- coding: utf-8 -*-
"""
DISCBench 精简评测管线：仅支持论文复现三环节 + 裁判选型。

  Stage 1  evaluate_instructions  — 指令质量评估
  Stage 2  generate_replies       — 多模型生成回复
  Stage 3  evaluate_replies       — LLM 裁判评估回复
  可选     test_judge_models       — 交互选择裁判模型
"""
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_pipeline_file = os.path.abspath(__file__)
_evaluation_pkg_dir = os.path.dirname(_pipeline_file)
_project_root = os.path.dirname(_evaluation_pkg_dir)

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import JUDGE_CANDIDATE_MODELS, MODEL_PROVIDER_MAPPING, get_provider_for_model

from .core.blacklist import MODEL_BLACKLIST
from .core.utils import (
    BENCHMARK_EIGHT_FAMILY_LABEL_ZH,
    eval_benchmark_eight_family,
    filter_available_exclude_patterns,
    pick_balanced_eight_families,
    pick_n_cross_vendor_models,
    safe_save_excel,
)
from .managers.constraint_library import ConstraintLibraryManager
from .managers.directory import DirectoryManager
from .managers.sysprompt import SyspromptManager
from .models_from_excel import (
    SOURCE_MODELS_EXCEL_KEY,
    detect_models_table_format,
    load_judge_family_models_from_excel,
    load_models_from_excel,
    mark_failed_models_on_excel_paths,
    split_configs_by_excel_cache,
    try_use_cached_availability_multi,
    update_availability_for_test_results,
)
from .stages.stage1_quality import batch_evaluate_instruction_quality
from .stages.stage3_reply import batch_generate_replies, batch_generate_replies_paired_round_robin
from .stages.stage4_evaluate import batch_evaluate_responses_with_cache
from .testing.model_tester import ModelAvailabilityTester, expand_reply_model_configs

SUPPORTED_STAGES = (
    "test_judge_models",
    "evaluate_instructions",
    "generate_replies",
    "evaluate_replies",
)

STAGE_ORDER = {
    "test_judge_models": 0,
    "evaluate_instructions": 1,
    "generate_replies": 2,
    "evaluate_replies": 3,
}


class PipelineManager:
    STAGE_DEFINITIONS = {
        "test_judge_models": {
            "name": "裁判模型可用性测试与选型",
            "input_files": [],
            "output_files": [],
        },
        "evaluate_instructions": {
            "name": "指令质量评估",
            "input_files": ["data/questions.jsonl"],
            "output_files": ["data/questions.jsonl", "output/stage1_quality/"],
        },
        "generate_replies": {
            "name": "多模型回复生成",
            "input_files": ["data/questions.jsonl"],
            "output_files": ["data/replies.jsonl"],
        },
        "evaluate_replies": {
            "name": "回复裁判评估",
            "input_files": ["data/questions.jsonl", "data/replies.jsonl"],
            "output_files": ["data/replies.jsonl"],
        },
    }

    def __init__(self, config: dict):
        self.config = config
        pr = config.get("project_root_dir")
        if pr and str(pr).strip():
            pr = str(pr).strip()
            self.project_root = os.path.abspath(pr if os.path.isabs(pr) else os.path.join(_project_root, pr))
        else:
            self.project_root = _project_root
        base = config.get("output_base_dir", "output")
        project_id = config.get("project_id") or ""
        self.dir_manager = DirectoryManager(
            base_dir=base, project_id=project_id, project_root=self.project_root
        )
        sysprompt_path = config.get("sysprompt_excel", "data/sysprompts.xlsx")
        if sysprompt_path and not os.path.isabs(sysprompt_path):
            sysprompt_path = os.path.join(self.project_root, sysprompt_path)
        self.sysprompt_manager = SyspromptManager(sysprompt_path)
        self.constraint_library = ConstraintLibraryManager(
            self._resolve_path(self.dir_manager.get_path("library", "constraint_library.xlsx"))
        )

    # ---------- 路径解析 ----------

    def _resolve_path(self, path_str: str) -> str:
        if not path_str or os.path.isabs(path_str):
            return path_str or ""
        return os.path.join(self.project_root, path_str)

    def _batch_suffix(self) -> str:
        batch = (self.config.get("data_batch") or "").strip()
        return f"_{batch}" if batch else ""

    def _resolve_questions_excel_for_evaluate(self) -> str:
        custom = self.config.get("questions_excel")
        if custom and str(custom).strip():
            path = str(custom).strip()
            return path if os.path.isabs(path) else self._resolve_path(path)
        suffix = self._batch_suffix()
        return self._resolve_path(
            self.dir_manager.get_path("questions", f"questions_complete{suffix}.xlsx")
        )

    def _resolve_replies_excel(self) -> str:
        raw = self.config.get("replies_excel")
        if not raw:
            suffix = self._batch_suffix()
            return self._resolve_path(
                self.dir_manager.get_path("replies", f"replies{suffix}.xlsx")
            )
        if isinstance(raw, list):
            paths = [
                (p if os.path.isabs(p) else self._resolve_path(str(p).strip()))
                for p in raw
                if p and str(p).strip()
            ]
            if len(paths) == 1:
                return paths[0]
            if not paths:
                suffix = self._batch_suffix()
                return self._resolve_path(
                    self.dir_manager.get_path("replies", f"replies{suffix}.xlsx")
                )
            dfs = []
            for p in paths:
                if not os.path.exists(p):
                    print(f"  ⚠️  回复表不存在，跳过: {p}")
                    continue
                xls = pd.ExcelFile(p)
                sheet = next(
                    (s for s in ("Sheet1", "replies") if s in xls.sheet_names),
                    xls.sheet_names[0],
                )
                df = pd.read_excel(p, sheet_name=sheet)
                if {"qid", "model", "reply"}.issubset(df.columns):
                    dfs.append(df)
            if not dfs:
                raise FileNotFoundError("未成功读取任何回复表（需含 qid, model, reply）")
            merged = pd.concat(dfs, ignore_index=True)
            out_path = self._resolve_path(
                self.dir_manager.get_path("replies", "merged_replies.xlsx")
            )
            safe_save_excel(merged, out_path)
            print(f"  📂 已合并 {len(paths)} 个回复表 → {out_path}  共 {len(merged)} 行\n")
            return out_path
        path = str(raw).strip()
        return path if os.path.isabs(path) else self._resolve_path(path)

    def _resolve_one_models_excel(self, rel: str) -> Optional[str]:
        rel = str(rel or "").strip()
        if not rel:
            return None
        if os.path.isabs(rel):
            return rel if os.path.exists(rel) else None
        path = os.path.join(self.project_root, rel)
        if not os.path.exists(path):
            legacy = os.path.join(_project_root, rel)
            if os.path.exists(legacy):
                path = legacy
        return path if os.path.exists(path) else None

    def _resolve_models_excel_paths(self) -> List[str]:
        raw = self.config.get("models_excel")
        out: List[str] = []
        if isinstance(raw, (list, tuple)):
            for item in raw:
                p = self._resolve_one_models_excel(str(item))
                if p:
                    out.append(p)
        elif raw and str(raw).strip():
            p = self._resolve_one_models_excel(str(raw))
            if p:
                out.append(p)
        else:
            p = self._resolve_one_models_excel("data/models.xlsx")
            out = [p] if p else []
        seen: set = set()
        return [p for p in out if not (p in seen or seen.add(p))]

    def _resolve_judge_models_excel_paths(self) -> List[str]:
        raw = self.config.get("judge_models_excel")
        if isinstance(raw, (list, tuple)):
            return [p for item in raw if (p := self._resolve_one_models_excel(str(item)))]
        if raw and str(raw).strip():
            p = self._resolve_one_models_excel(str(raw))
            return [p] if p else []
        return self._resolve_models_excel_paths()

    def _get_sorted_stages(self, stages: List[str]) -> List[str]:
        for stage in stages:
            if stage not in SUPPORTED_STAGES:
                raise ValueError(
                    f"未知阶段: {stage}。本精简版仅支持: {', '.join(SUPPORTED_STAGES)}"
                )
        return sorted(stages, key=lambda s: STAGE_ORDER.get(s, 99))

    # ---------- 裁判选型 ----------

    def _force_model_selection(self) -> bool:
        return bool(self.config.get("force_model_selection"))

    def _resolve_judge_provider_model(self) -> tuple:
        cfg = self.config
        explicit_provider = cfg.get("provider")
        model = cfg.get("model")
        if explicit_provider and model:
            return explicit_provider, model
        if model:
            try:
                pc = get_provider_for_model(model)
                return pc.name, model
            except ValueError:
                pass
        return explicit_provider, model

    def _judge_cache_path(self) -> str:
        return self._resolve_path(
            self.dir_manager.get_path("library", "judge_availability_cache.json")
        )

    def _try_use_judge_cache(self, provider: str, model: str, max_age_days: int) -> Optional[bool]:
        if not provider or not model:
            return None
        path = self._judge_cache_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            return None
        entry = cache.get(f"{provider}:{model}")
        if not entry or not isinstance(entry, dict):
            return None
        last_str = entry.get("last_tested")
        if not last_str:
            return None
        try:
            last_dt = datetime.fromisoformat(last_str.replace("Z", "+00:00"))
            if getattr(last_dt, "tzinfo", None):
                last_dt = last_dt.replace(tzinfo=None)
        except (ValueError, TypeError, OSError):
            return None
        if datetime.now() - last_dt > timedelta(days=max_age_days):
            return None
        return bool(entry.get("available", False))

    def _save_judge_cache(self, provider: str, model: str, available: bool) -> None:
        path = self._judge_cache_path()
        cache = {}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception:
                pass
        cache[f"{provider}:{model}"] = {
            "last_tested": datetime.now().isoformat(),
            "available": available,
        }
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _get_judge_candidate_configs(self) -> List[Dict]:
        cfg = self.config
        custom = cfg.get("judge_model_configs")
        if custom:
            return expand_reply_model_configs(custom)

        excel_paths: List[str] = []
        if cfg.get("judge_models_from_excel"):
            excel_paths = self._resolve_judge_models_excel_paths()
        configs: List[Dict] = []
        only_available = bool(cfg.get("judge_only_available", False))
        family_kw = cfg.get("judge_model_family_keywords") or ["gpt", "claude", "gemini"]
        if cfg.get("judge_models_load_all"):
            for p in excel_paths:
                configs.extend(load_models_from_excel(p, provider=None, only_available=only_available))
        elif cfg.get("judge_models_family_filter", True) and excel_paths:
            configs = load_judge_family_models_from_excel(
                excel_paths, families=family_kw, only_available=only_available
            )
        elif excel_paths:
            suppliers = cfg.get("judge_models_suppliers") or ["Anthropic", "Google", "openai"]
            for sup in suppliers:
                for p in excel_paths:
                    configs.extend(
                        load_models_from_excel(p, provider=sup, only_available=only_available)
                    )

        seen: set = set()
        deduped: List[Dict] = []
        for c in configs:
            key = (
                str(c.get("provider", "")).strip().lower(),
                str(c.get("model", "")).strip(),
                str(c.get("aimux_origin_code", "")).strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)

        if cfg.get("judge_include_builtin_channels", True):
            for model_name in JUDGE_CANDIDATE_MODELS:
                if model_name not in MODEL_PROVIDER_MAPPING:
                    continue
                try:
                    pc = get_provider_for_model(model_name)
                    key = (pc.name.lower(), model_name.lower(), "")
                    if key not in seen:
                        seen.add(key)
                        deduped.append({"provider": pc.name, "model": model_name})
                except ValueError:
                    pass
        return deduped

    def _fetch_available_judge_pool(self) -> Optional[pd.DataFrame]:
        cfg = self.config
        max_age_days = cfg.get("availability_test_max_age_days", 14)
        configs = self._get_judge_candidate_configs()
        if not configs:
            print("⚠️  裁判模型候选为空。请配置 judge_model_configs 或 models_excel。")
            return None

        test_results = None
        force_refresh = bool(cfg.get("availability_test_force_refresh", False))
        configs_to_probe = list(configs)
        cached_df = pd.DataFrame()

        if cfg.get("judge_models_from_excel") and not force_refresh:
            cached_df, configs_to_probe = split_configs_by_excel_cache(
                configs, max_age_days=max_age_days, provider=None
            )
            if len(cached_df) and not configs_to_probe:
                test_results = cached_df
                print(f"  📋 裁判可用状态均来自表格缓存（< {max_age_days} 天）\n")

        if test_results is None:
            judge_workers = int(cfg.get("judge_test_max_workers") or cfg.get("max_workers") or 3)
            judge_prompt = cfg.get("judge_probe_test_prompt") or "Reply with exactly: OK"
            tester = ModelAvailabilityTester(
                timeout=cfg.get("test_timeout", 30),
                test_prompt=judge_prompt,
            )
            probed = (
                tester.test_all_models(configs_to_probe, max_workers=judge_workers)
                if configs_to_probe
                else pd.DataFrame()
            )
            parts = [df for df in (cached_df, probed) if df is not None and not df.empty]
            test_results = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            judge_paths = self._resolve_judge_models_excel_paths()
            if judge_paths and probed is not None and not probed.empty:
                try:
                    update_availability_for_test_results(judge_paths, probed, provider=None)
                except Exception as e:
                    print(f"  ⚠️ 写回裁判可用状态失败: {e}")

        available = test_results[test_results["available"]].sort_values("response_time").reset_index(drop=True)
        if available.empty:
            print("❌ 裁判模型候选均不可用，请检查配置或网络。")
            return None
        return available

    @staticmethod
    def _sort_judge_available_for_display(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        deprioritize = cfg.get("judge_deprioritize_providers") or ["aimux"]
        dep_set = {str(p).strip().lower() for p in deprioritize if str(p).strip()}

        def _rank(row) -> tuple:
            prov = str(row.get("provider", "")).strip().lower()
            rt = row.get("response_time")
            rt_v = float(rt) if rt is not None and not pd.isna(rt) else 9999.0
            for d in dep_set:
                if prov == d or prov.startswith(d + "_") or d in prov:
                    return (1, rt_v)
            return (0, rt_v)

        out = df.copy()
        out["_sort_rank"] = out.apply(_rank, axis=1)
        return out.sort_values("_sort_rank").drop(columns=["_sort_rank"]).reset_index(drop=True)

    def _filter_available_pool(self, available: pd.DataFrame, scope: str) -> pd.DataFrame:
        pats = self.config.get(f"{scope}_exclude_model_patterns") or self.config.get("exclude_model_patterns")
        if pats is None and scope == "reply":
            pats = ["deepseek"]
        if not pats or available is None or available.empty:
            return available
        if isinstance(pats, str):
            pats = [pats.strip()] if pats.strip() else []
        filtered = filter_available_exclude_patterns(available, pats)
        return filtered.reset_index(drop=True) if not filtered.empty else available.reset_index(drop=True)

    def check_and_select_judge_model(self) -> Optional[Dict[str, str]]:
        cfg = self.config
        max_age_days = cfg.get("availability_test_max_age_days", 14)
        judge_provider, judge_model = self._resolve_judge_provider_model()
        if judge_provider and judge_model and not self._force_model_selection():
            cached = self._try_use_judge_cache(judge_provider, judge_model, max_age_days)
            if cached is True:
                print(
                    f"  📋 使用裁判缓存（{judge_provider}/{judge_model}，"
                    f"< {max_age_days} 天），跳过 API 检测\n"
                )
                self._judge_checked_this_run = True
                return {"provider": judge_provider, "model": judge_model}

        available = self._fetch_available_judge_pool()
        if available is None:
            return None
        available = self._filter_available_pool(available, "judge")
        available = self._sort_judge_available_for_display(available, cfg)

        print(f"\n{'=' * 60}")
        print(f"📋 可用裁判模型（共 {len(available)} 项）")
        print(f"{'=' * 60}")
        for i, row in available.iterrows():
            rt = row.get("response_time")
            rt_str = f" {rt:.2f}s" if rt is not None and not pd.isna(rt) else ""
            print(f"  {i + 1:3d}. {row['provider']:20s} / {row['model']:35s}{rt_str}")
        print(f"{'=' * 60}")
        print("  请输入裁判编号（直接回车=随机 1 个）：", end="")
        try:
            choice = input().strip()
        except EOFError:
            choice = ""
        if not choice:
            picked, _ = pick_n_cross_vendor_models(available, n=1, prefer_random=True)
            if not picked:
                return None
            selected = pd.Series(picked[0])
        else:
            idx = max(0, min(int(choice) - 1, len(available) - 1)) if choice.isdigit() else 0
            selected = available.iloc[idx]
        self.config["provider"] = selected["provider"]
        self.config["model"] = selected["model"]
        self._judge_checked_this_run = True
        self._save_judge_cache(selected["provider"], selected["model"], True)
        print(f"\n✅ 已选择裁判: {selected['provider']} / {selected['model']}")
        return {"provider": selected["provider"], "model": selected["model"]}

    # ---------- 回复模型选型 ----------

    @staticmethod
    def _rows_to_reply_configs(rows: List[dict]) -> List[Dict]:
        out: List[Dict] = []
        for row in rows:
            item = {
                "provider": row["provider"],
                "model": row["model"],
                "enable_thinking": row.get("enable_thinking", False),
            }
            oc = row.get("aimux_origin_code")
            if oc is not None and str(oc).strip():
                item["aimux_origin_code"] = str(oc).strip()
            src_ex = row.get(SOURCE_MODELS_EXCEL_KEY)
            if src_ex is not None and str(src_ex).strip():
                item[SOURCE_MODELS_EXCEL_KEY] = str(src_ex).strip()
            out.append(item)
        return out

    def check_and_select_reply_models(self) -> List[Dict]:
        cfg = self.config
        model_configs = cfg.get("reply_model_configs") or []
        models_excel_paths: List[str] = []
        if cfg.get("use_models_from_excel"):
            models_excel_paths = self._resolve_models_excel_paths()
            if models_excel_paths:
                try:
                    model_configs = load_models_from_excel(
                        models_excel_paths, provider=cfg.get("models_excel_provider")
                    )
                    print(f"  📂 已从表格加载待测模型: {len(model_configs)} 个\n")
                except Exception as e:
                    print(f"  ⚠️ 从表格加载失败: {e}\n")

        if not model_configs:
            print("⚠️  待测模型为空。请配置 use_models_from_excel 或 reply_model_configs。")
            return []

        max_age_days = cfg.get("availability_test_max_age_days", 14)
        test_results = None
        force_refresh = bool(
            cfg.get("availability_test_force_refresh", False) or self._force_model_selection()
        )
        if models_excel_paths and not force_refresh:
            can_skip, cached_df = try_use_cached_availability_multi(
                models_excel_paths,
                model_configs,
                max_age_days=max_age_days,
                provider=cfg.get("models_excel_provider"),
            )
            if can_skip and cached_df is not None:
                test_results = cached_df
                print(f"  📋 使用表格缓存（< {max_age_days} 天），跳过 API 检测\n")

        if test_results is None:
            tester = ModelAvailabilityTester(
                timeout=cfg.get("test_timeout", 30),
                test_prompt=cfg.get("test_prompt"),
            )
            test_results = tester.test_all_models(model_configs, max_workers=3)
            if models_excel_paths and not test_results.empty:
                try:
                    update_availability_for_test_results(
                        models_excel_paths,
                        test_results,
                        provider=cfg.get("models_excel_provider"),
                    )
                except Exception as e:
                    print(f"  ⚠️ 写回表格失败: {e}")

        available = (
            test_results[test_results["available"] == True]
            .sort_values("response_time")
            .reset_index(drop=True)
        )
        if available.empty:
            print("❌ 当前无可用回复模型。")
            return []
        available = self._filter_available_pool(available, "reply")

        if bool(cfg.get("reply_auto_eight_families", False)):
            picked, missing = pick_balanced_eight_families(available)
            if not picked:
                print("⚠️  auto8 未选出任何模型。")
                return []
            reply_configs = self._rows_to_reply_configs(picked)
            self.config["reply_model_configs"] = reply_configs
            print(f"\n  ✅ auto8 已选 {len(reply_configs)} 个模型")
            if missing:
                print(f"  ⚠️ 未覆盖: {', '.join(missing)}")
            return reply_configs

        print(f"\n{'=' * 60}")
        print(f"📋 可用回复模型（共 {len(available)} 项）")
        print(f"{'=' * 60}")
        for i, row in available.iterrows():
            pv = str(row.get("provider", "") or "")
            mid = str(row.get("model", "") or "")
            fam = eval_benchmark_eight_family(pv, mid, row.get("aimux_origin_code"))
            fam_note = f" | {BENCHMARK_EIGHT_FAMILY_LABEL_ZH.get(fam, fam)}" if fam else ""
            print(f"  {i + 1:3d}. [{pv:18s}] {mid}{fam_note}")
        print(f"{'=' * 60}")
        print("  输入编号（逗号分隔）；all=全选；auto8=八大家各 1 条；回车=all：", end="")
        try:
            choice = input().strip() or "all"
        except EOFError:
            choice = "all"

        if choice.lower() == "auto8":
            picked, missing = pick_balanced_eight_families(available)
            selected = picked
            if missing:
                print(f"  ⚠️ 未覆盖: {', '.join(missing)}")
        elif choice.lower() == "all":
            selected = available.to_dict("records")
        else:
            selected = []
            for part in choice.split(","):
                part = part.strip()
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(available):
                        selected.append(available.iloc[idx].to_dict())

        reply_configs = self._rows_to_reply_configs(selected)
        self.config["reply_model_configs"] = reply_configs
        print(f"\n✅ 已选择 {len(reply_configs)} 个模型参与批量回复。")
        return reply_configs

    # ---------- 阶段执行 ----------

    def _ensure_judge_ready(self, stage: str) -> Tuple[str, str]:
        cfg = self.config
        judge_stages = ("evaluate_instructions", "evaluate_replies")
        if stage not in judge_stages:
            return self._resolve_judge_provider_model()

        if cfg.get("check_judge_before_use") and not getattr(self, "_judge_checked_this_run", False):
            if cfg.get("skip_judge_validation"):
                print("  ⚠️  check_judge_before_use=true，将交互选择裁判（忽略 skip_judge_validation）\n")
            if not self.check_and_select_judge_model():
                raise RuntimeError("裁判模型选择取消或无可选模型。")

        judge_provider, judge_model = self._resolve_judge_provider_model()
        if not judge_provider or not judge_model:
            raise RuntimeError(
                "未配置裁判模型。请在 config 中设置 provider/model，"
                "或在 stages 中加入 test_judge_models。"
            )
        print(f"  裁判模型: {judge_provider}/{judge_model}\n")
        return judge_provider, judge_model

    def execute_stage(self, stage: str) -> None:
        stage_def = self.STAGE_DEFINITIONS[stage]
        print(f"\n{'=' * 60}")
        print(f"🚀 执行阶段: {stage_def['name']} ({stage})")
        print(f"{'=' * 60}\n")

        cfg = self.config
        dm = self.dir_manager
        sp = self.sysprompt_manager

        if stage == "test_judge_models":
            if not self.check_and_select_judge_model():
                raise RuntimeError("裁判模型选择取消或无可选模型。")
            return

        judge_provider, judge_model = self._ensure_judge_ready(stage)

        if stage == "evaluate_instructions":
            iq_cfg = cfg.get("instruction_quality") or {}
            input_excel = iq_cfg.get("input_excel") or cfg.get("questions_excel")
            input_excel = (
                input_excel
                if os.path.isabs(str(input_excel))
                else self._resolve_path(str(input_excel))
            )
            out_name = iq_cfg.get("output_excel") or "evaluated_instructions.xlsx"
            output_excel = (
                out_name
                if os.path.isabs(str(out_name)) or "/" in str(out_name)
                else self._resolve_path(dm.get_path("stage1_quality", out_name))
            )
            questions_excel = iq_cfg.get("questions_excel") or cfg.get("questions_excel")
            if questions_excel:
                questions_excel = (
                    questions_excel
                    if os.path.isabs(str(questions_excel))
                    else self._resolve_path(str(questions_excel))
                )
            _iq_src = iq_cfg.get("source_include") or (cfg.get("data_filters") or {}).get("source_include")
            batch_evaluate_instruction_quality(
                input_excel=input_excel,
                output_excel=output_excel,
                provider=judge_provider,
                model=judge_model,
                sysprompt_manager=sp,
                constraint_library=self.constraint_library,
                temperature=cfg.get("evaluation_temperature", 0.3),
                max_workers=iq_cfg.get("max_workers", cfg.get("max_workers", 4)),
                checkpoint_interval=cfg.get("checkpoint_interval", 10),
                timeout=cfg["timeout"],
                input_sheet=iq_cfg.get("input_sheet", 0),
                questions_excel=questions_excel,
                questions_sheet=iq_cfg.get("questions_sheet", iq_cfg.get("input_sheet", 0)),
                merge_back=iq_cfg.get("merge_back", bool(questions_excel)),
                qid_col=iq_cfg.get("qid_column", "qid"),
                qid_prefix=iq_cfg.get("qid_prefix", "iq"),
                use_model_weights=bool(iq_cfg.get("use_model_weights", False)),
                rescoring_only=bool(iq_cfg.get("rescoring_only", False)),
                force_rerun=bool(iq_cfg.get("force_rerun", False)),
                source_include=_iq_src,
            )

        elif stage == "generate_replies":
            from .reply_model_routes import (
                apply_chosen_routes_to_configs,
                configure_route_policy,
                expand_reply_configs_with_routes,
                probe_and_select_routes,
                sort_reply_configs_for_run,
            )

            configure_route_policy(
                deprioritize_providers=cfg.get("reply_deprioritize_providers", []),
                model_aliases=cfg.get("reply_model_aliases"),
                version_fallbacks=cfg.get("reply_version_fallbacks"),
                max_routes_per_logical=cfg.get("reply_max_routes_per_logical", 6),
                max_backup_routes=cfg.get("reply_max_backup_routes", 2),
            )
            if cfg.get("check_models_before_reply", True):
                if not self.check_and_select_reply_models():
                    print("⚠️  未选择任何模型，跳过批量回复。")
                    return
            elif cfg.get("use_models_from_excel"):
                paths = self._resolve_models_excel_paths()
                if paths:
                    only_available = bool(cfg.get("reply_models_only_available", True))
                    reply_configs = load_models_from_excel(
                        paths,
                        provider=cfg.get("models_excel_provider"),
                        only_available=only_available,
                    )
                    if reply_configs:
                        cfg["reply_model_configs"] = reply_configs
                        print(f"  📂 回复模型：已加载 {len(reply_configs)} 个（only_available={only_available}）\n")

            if cfg.get("reply_multi_source_fallback", True):
                paths_ms = self._resolve_models_excel_paths()
                base_rc = cfg.get("reply_model_configs") or []
                if base_rc and paths_ms:
                    expanded = expand_reply_configs_with_routes(base_rc, paths_ms)
                    if cfg.get("reply_probe_routes_before_run", True):
                        chosen = probe_and_select_routes(
                            expanded,
                            timeout=int(cfg.get("test_timeout", 60)),
                            probe_skip_providers=cfg.get("reply_probe_skip_providers"),
                        )
                        expanded = apply_chosen_routes_to_configs(expanded, chosen)
                    expanded = sort_reply_configs_for_run(expanded, {}, cfg.get("reply_model_run_order"))
                    cfg["reply_model_configs"] = expanded

            if not cfg.get("reply_model_configs"):
                print("⚠️  无待测模型，跳过批量回复。")
                return

            questions_for_reply = self._resolve_questions_excel_for_evaluate()
            if not questions_for_reply or not os.path.exists(questions_for_reply):
                raise FileNotFoundError(f"题目表不存在: {questions_for_reply}")
            print(f"  📂 题目表: {os.path.basename(questions_for_reply)}\n")

            models_excel_paths = self._resolve_models_excel_paths()
            models_excel_path = models_excel_paths[0] if models_excel_paths else None
            _q_sheet = cfg.get("questions_sheet") or (cfg.get("instruction_quality") or {}).get("questions_sheet")
            _src_inc = cfg.get("reply_source_include") or (cfg.get("data_filters") or {}).get("source_include")
            reply_mode = (cfg.get("reply_generation_mode") or "full").strip().lower()

            if reply_mode in ("paired_round_robin", "paired", "b"):
                jp, jm, joc = cfg.get("provider"), cfg.get("model"), None
                batch_sfx = self._batch_suffix()
                matches_path = cfg.get("matches_excel") or self._resolve_path(
                    dm.get_path("replies", f"matches{batch_sfx}.xlsx")
                )
                df_replies, failed_from_run = batch_generate_replies_paired_round_robin(
                    questions_excel=questions_for_reply,
                    model_configs=cfg["reply_model_configs"],
                    output_excel=self._resolve_replies_excel(),
                    matches_excel=matches_path,
                    per_question_max_models=int(cfg.get("per_question_max_models", 8) or 8),
                    temperature=cfg.get("reply_temperature", 0.6),
                    max_workers=cfg.get("max_workers", 5),
                    checkpoint_interval=cfg.get("checkpoint_interval", 10),
                    timeout=cfg["timeout"],
                    models_excel_path=models_excel_path,
                    models_excel_paths=models_excel_paths or None,
                    exclude_judge_from_paired_pool=bool(cfg.get("paired_exclude_judge", True)),
                    judge_provider=jp,
                    judge_model=jm,
                    judge_aimux_origin_code=joc,
                )
            else:
                df_replies, failed_from_run = batch_generate_replies(
                    questions_excel=questions_for_reply,
                    model_configs=cfg["reply_model_configs"],
                    output_excel=self._resolve_replies_excel(),
                    temperature=cfg.get("reply_temperature", 0.6),
                    max_workers=cfg.get("max_workers", 5),
                    checkpoint_interval=cfg.get("checkpoint_interval", 10),
                    timeout=int(cfg.get("reply_timeout") or cfg.get("timeout", 120)),
                    models_excel_path=models_excel_path,
                    models_excel_paths=models_excel_paths or None,
                    source_include=_src_inc,
                    questions_sheet=_q_sheet,
                    existing_model_aliases=cfg.get("reply_existing_model_aliases"),
                    reply_skip_by_vendor_family=bool(cfg.get("reply_skip_by_vendor_family", True)),
                )

            blacklisted = MODEL_BLACKLIST.get_all()
            all_failed = blacklisted + (failed_from_run or [])
            if models_excel_paths and all_failed:
                n = mark_failed_models_on_excel_paths(
                    models_excel_paths, replies_df=None, blacklisted_models=all_failed
                )
                if n > 0:
                    print(f"  ✅ 已将 {n} 个失败模型标记为不可用\n")

        elif stage == "evaluate_replies":
            questions_file = self._resolve_questions_excel_for_evaluate()
            replies_file = self._resolve_replies_excel()
            if not questions_file or not os.path.exists(questions_file):
                raise FileNotFoundError(f"题目表不存在: {questions_file}")
            if not replies_file:
                raise FileNotFoundError("未配置 replies_excel")
            print(
                f"  📌 题目表: {os.path.basename(questions_file)}  "
                f"回复表: {os.path.basename(replies_file)}\n"
            )
            _trim = cfg.get("trim_replies_to_question_qids", True)
            if isinstance(_trim, str):
                _trim = _trim.strip().lower() not in ("false", "0", "no", "否")
            _q_sheet_ev = cfg.get("questions_sheet") or (cfg.get("instruction_quality") or {}).get("questions_sheet")
            batch_evaluate_responses_with_cache(
                questions_excel=questions_file,
                replies_excel=replies_file,
                output_excel=replies_file,
                provider=judge_provider,
                model=judge_model,
                sysprompt_manager=sp,
                batch_id=cfg.get("batch_id"),
                data_filters=cfg.get("data_filters"),
                temperature=cfg.get("evaluation_temperature", 0.3),
                max_workers=cfg.get("max_workers", 5),
                checkpoint_interval=cfg.get("checkpoint_interval", 10),
                timeout=cfg["timeout"],
                overwrite_mode=cfg.get("overwrite_mode", "skip"),
                inject_expert_opinions_in_evaluation=cfg.get(
                    "inject_expert_opinions_in_evaluation", True
                ),
                skip_ref_in_evaluation=cfg.get("skip_ref_in_evaluation", False),
                trim_replies_to_question_qids=bool(_trim),
                questions_sheet=_q_sheet_ev,
                existing_model_aliases=cfg.get("reply_existing_model_aliases"),
                eval_deduplicate_by_logical_model=cfg.get(
                    "eval_deduplicate_by_logical_model", True
                ),
            )

        else:
            raise ValueError(f"未实现的阶段: {stage}")

    def _print_path_summary(self, sorted_stages: List[str]) -> None:
        lines = []
        if "evaluate_instructions" in sorted_stages:
            iq = self.config.get("instruction_quality") or {}
            inp = iq.get("input_excel") or self.config.get("questions_excel") or "—"
            out = iq.get("output_excel") or "output/stage1_quality/"
            lines.append(("evaluate_instructions", inp, out))
        if "generate_replies" in sorted_stages:
            lines.append((
                "generate_replies",
                self.config.get("questions_excel") or "—",
                self.config.get("replies_excel") or "output/replies/",
            ))
        if "evaluate_replies" in sorted_stages:
            lines.append((
                "evaluate_replies",
                f"{self.config.get('questions_excel')} + {self.config.get('replies_excel')}",
                self.config.get("replies_excel") or "—",
            ))
        if lines:
            print("📂 输入/输出路径摘要")
            print(f"{'=' * 60}")
            for label, inp, out in lines:
                print(f"  【{label}】")
                print(f"    输入: {inp}")
                print(f"    输出: {out}")
            print(f"{'=' * 60}\n")

    def run(self, stages: List[str], preserve_judge_selection: bool = False) -> bool:
        sorted_stages = self._get_sorted_stages(stages)
        print(f"\n{'=' * 60}")
        print(f"🎯 DISCBench 评测流程  阶段数: {len(sorted_stages)}")
        print(f"{'=' * 60}")
        for i, stage in enumerate(sorted_stages, 1):
            print(f"  {i}. {self.STAGE_DEFINITIONS[stage]['name']} ({stage})")
        print(f"{'=' * 60}\n")

        self._print_path_summary(sorted_stages)
        try:
            if not preserve_judge_selection:
                self._judge_checked_this_run = False
            for stage in sorted_stages:
                self.execute_stage(stage)
            print(f"\n{'=' * 60}")
            print("✅ 流程执行完毕！")
            print(f"{'=' * 60}\n")
            return True
        except RuntimeError as e:
            print(f"\n❌ 流程终止: {e}")
            return False
        except Exception as e:
            print(f"\n❌ 执行失败: {e}")
            import traceback
            traceback.print_exc()
            return False
