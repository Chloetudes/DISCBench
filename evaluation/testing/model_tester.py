# -*- coding: utf-8 -*-
"""
模型可用性测试：支持单 provider、多 providers（同一模型多源）。
配置中可全量罗列所有模型/版本，测试后按可用清单交互选择，再进入批量回复。

严谨性说明：
- 默认使用「回复式」探针（与真实题目长度接近），通过即表示能完成一次类回复请求，尽量保证批量回复时选取的模型可用。
- test_timeout 建议与批量 timeout 一致或更大；若仍出现 RemoteDisconnected，可降低 max_workers 或增大 timeout。
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from config import get_provider, get_provider_for_model, aimux_provider_for_origin
from clients.openai_client import OAIClient
from ..core.utils import safe_save_excel
from ..models_from_excel import SOURCE_MODELS_EXCEL_KEY

# 默认用「回复式」探针：长度与真实题目接近，通过即表示能完成一次类回复请求，减少批量时 RemoteDisconnected
DEFAULT_REPLY_LIKE_PROMPT = (
    "请根据以下要求写一段话：主题为数据安全与隐私保护，字数约100字，需包含三个要点，"
    "并采用分点列举的格式。不要输出多余解释，直接给出正文。"
)


def expand_reply_model_configs(model_configs: List[Dict]) -> List[Dict]:
    """
    将 reply_model_configs 展开为 (provider, model) 列表，便于逐项测试。
    - { "model": "x" } → 使用 get_provider_for_model(x) 得到 1 项
    - { "model": "x", "provider": "p1" } → 1 项 (p1, x)
    - { "model": "x", "providers": ["p1", "p2"] } → 2 项 (p1,x), (p2,x)
    """
    expanded = []
    for cfg in model_configs:
        model_name = cfg.get("model")
        if not model_name:
            continue
        enable_thinking = cfg.get("enable_thinking", False)
        extra = {k: v for k, v in cfg.items() if k in ("展示名称", "aimux_origin_code", SOURCE_MODELS_EXCEL_KEY)}
        if "providers" in cfg and cfg["providers"]:
            for p in cfg["providers"]:
                expanded.append({"provider": p, "model": model_name, "enable_thinking": enable_thinking, **extra})
        elif cfg.get("provider"):
            expanded.append({
                "provider": cfg["provider"],
                "model": model_name,
                "enable_thinking": enable_thinking,
                **extra,
            })
        else:
            try:
                p = get_provider_for_model(model_name).name
                expanded.append({"provider": p, "model": model_name, "enable_thinking": enable_thinking, **extra})
            except Exception:
                expanded.append({"provider": "unknown", "model": model_name, "enable_thinking": enable_thinking, **extra})
    return expanded


class ModelAvailabilityTester:
    """模型可用性探针：默认使用回复式探针（与真实题目长度接近），通过表示能完成类回复请求。"""

    def __init__(self, timeout: int = 30, test_prompt: Optional[str] = None):
        self.timeout = timeout
        self.test_prompt = test_prompt if test_prompt is not None else DEFAULT_REPLY_LIKE_PROMPT

    def test_single_model(
        self, provider: str, model: str, aimux_origin_code: Optional[str] = None
    ) -> Dict:
        try:
            if provider == "aimux" and aimux_origin_code and str(aimux_origin_code).strip():
                pc = aimux_provider_for_origin(str(aimux_origin_code).strip())
            else:
                pc = get_provider(provider)
            client = OAIClient(
                base_url=pc.base_url,
                api_key=pc.api_key,
                protocol=pc.protocol,
                auth_header=pc.auth_header,
                auth_prefix=pc.auth_prefix,
                extra_headers=pc.extra_headers,
                timeout=self.timeout
            )
            messages = [{"role": "user", "content": self.test_prompt}]
            start_time = time.time()
            response = client.chat(model=model, messages=messages, temperature=0.3)
            elapsed_time = time.time() - start_time

            if isinstance(response, str):
                is_error = (
                    response.startswith('<error') or
                    response.startswith('<!DOCTYPE') or
                    response.startswith('<html') or
                    'IRC-001' in response or
                    '没有当前模型的权限' in response or
                    'permission denied' in response.lower() or
                    'unauthorized' in response.lower()
                )
                if is_error:
                    return {'provider': provider, 'model': model, 'available': False,
                            'error': response[:200], 'response_time': elapsed_time}

            return {
                'provider': provider, 'model': model, 'available': True,
                'error': None, 'response_time': elapsed_time,
                'response_preview': response[:100] if isinstance(response, str) else str(response)[:100]
            }
        except Exception as e:
            return {'provider': provider, 'model': model, 'available': False,
                    'error': str(e), 'response_time': None}

    def test_all_models(self, model_configs: List[Dict], max_workers: int = 3) -> pd.DataFrame:
        """测试所有模型；model_configs 支持 provider / providers 列表，会先展开再逐项测试。"""
        expanded = expand_reply_model_configs(model_configs)
        print(f"\n{'=' * 60}")
        print(f"🧪 模型可用性测试")
        print(f"{'=' * 60}\n")
        print(f"  配置项: {len(model_configs)}  展开后待测: {len(expanded)}")
        print(f"  并发数: {max_workers}  超时: {self.timeout}秒  探针: 回复式（{len(self.test_prompt)}字），通过即表示能完成类回复请求\n")

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for cfg in expanded:
                provider = cfg.get("provider") or "unknown"
                model = cfg["model"]
                if provider == "unknown":
                    results.append({"provider": "unknown", "model": model, "available": False,
                                    "error": "未配置 provider 且无法从 get_provider_for_model 解析", "response_time": None})
                    continue
                oc = cfg.get("aimux_origin_code")
                oc_arg = str(oc).strip() if oc is not None and str(oc).strip() else None
                future = executor.submit(
                    self.test_single_model, provider, model, oc_arg
                )
                futures[future] = cfg

            for future in tqdm(as_completed(futures), total=len(futures), desc="🔄 测试进度", ncols=100):
                try:
                    row = future.result()
                    cfg = futures[future]
                    row["enable_thinking"] = cfg.get("enable_thinking", False)
                    if "展示名称" in cfg:
                        row["展示名称"] = cfg.get("展示名称", "")
                    if cfg.get("aimux_origin_code") is not None and str(cfg.get("aimux_origin_code", "")).strip():
                        row["aimux_origin_code"] = str(cfg["aimux_origin_code"]).strip()
                    src_ex = cfg.get(SOURCE_MODELS_EXCEL_KEY)
                    if src_ex and str(src_ex).strip():
                        row[SOURCE_MODELS_EXCEL_KEY] = str(src_ex).strip()
                    results.append(row)
                except Exception as e:
                    cfg = futures[future]
                    r = {"provider": cfg.get("provider", "unknown"), "model": cfg["model"],
                         "available": False, "error": str(e), "response_time": None,
                         "enable_thinking": cfg.get("enable_thinking", False)}
                    if "展示名称" in cfg:
                        r["展示名称"] = cfg.get("展示名称", "")
                    if cfg.get("aimux_origin_code") is not None and str(cfg.get("aimux_origin_code", "")).strip():
                        r["aimux_origin_code"] = str(cfg["aimux_origin_code"]).strip()
                    src_ex = cfg.get(SOURCE_MODELS_EXCEL_KEY)
                    if src_ex and str(src_ex).strip():
                        r[SOURCE_MODELS_EXCEL_KEY] = str(src_ex).strip()
                    results.append(r)

        df = pd.DataFrame(results)
        available_count = df['available'].sum()
        unavailable_count = len(df) - available_count

        print(f"\n{'=' * 60}")
        print(f"✅ 测试完成 — 可用: {available_count}  不可用: {unavailable_count}")
        print(f"{'=' * 60}")
        if available_count > 0:
            print("  💡 以上模型已通过回复式探针，将用于批量回复。若仍出现断连，可降低 max_workers 或增大 timeout。")

        if available_count > 0:
            print(f"\n✅ 可用模型列表:")
            for _, row in df[df['available']].sort_values('response_time').iterrows():
                print(f"  ✓ {row['provider']:20s} / {row['model']:35s} ({row['response_time']:.2f}秒)")

        if unavailable_count > 0:
            print(f"\n❌ 不可用模型列表:")
            for _, row in df[~df['available']].iterrows():
                error_preview = str(row['error'])[:80] + "..." if len(str(row['error'])) > 80 else str(row['error'])
                print(f"  ✗ {row['provider']:20s} / {row['model']:35s}")
                print(f"    原因: {error_preview}")

        print(f"{'=' * 60}\n")
        return df


def select_best_judge_model(test_results: pd.DataFrame,
                             preferred_models: Optional[List[str]] = None) -> Optional[Dict[str, str]]:
    available_models = test_results[test_results['available'] == True]
    if len(available_models) == 0:
        print("❌ 没有可用的模型")
        return None

    if preferred_models is None:
        preferred_models = [
            "claude-sonnet-4-5-20250929", "claude-opus-4-5-20251101",
            "gpt-5.2-chat-latest", "gpt-4o",
            "gemini-3-pro-preview", "deepseek-v3.2-exp", "qwen3-max"
        ]

    for model_name in preferred_models:
        match = available_models[available_models['model'] == model_name]
        if len(match) > 0:
            row = match.iloc[0]
            print(f"✅ 选择裁判模型: {row['provider']} / {row['model']}")
            return {'provider': row['provider'], 'model': row['model']}

    fastest = available_models.sort_values('response_time').iloc[0]
    print(f"✅ 选择裁判模型（最快响应）: {fastest['provider']} / {fastest['model']}")
    return {'provider': fastest['provider'], 'model': fastest['model']}
