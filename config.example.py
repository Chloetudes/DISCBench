# -*- coding: utf-8 -*-
"""
config.example.py — 复制为 config.py 并填入 API Key。
Stage 1–3 调用 LLM 必需；离线统计 scripts/run_stats.sh 不需要本文件。
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    protocol: str = "openai"
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer"
    extra_headers: Optional[Dict[str, str]] = None
    timeout: int = 120


def _env(key: str, default: str = "") -> str:
    return (os.environ.get(key) or default).strip()


# ---------- OpenAI 兼容 Provider（按需增删）----------
OTHER_CONFIGS = {
    "openai": ProviderConfig(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key=_env("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
    ),
    "dashscope": ProviderConfig(
        name="dashscope",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=_env("DASHSCOPE_API_KEY", "YOUR_DASHSCOPE_API_KEY"),
    ),
    "idealab": ProviderConfig(
        name="idealab",
        base_url=_env("IDEALAB_BASE_URL", "https://idealab.alibaba-inc.com/api/openai/v1"),
        api_key=_env("IDEALAB_API_KEY", "YOUR_IDEALAB_API_KEY"),
    ),
}

ROUTIFY_CONFIGS: Dict[str, ProviderConfig] = {}
ALL_CONFIGS = {**ROUTIFY_CONFIGS, **OTHER_CONFIGS}

# 模型名 → provider（与 data/models.xlsx 中「模型名称」对齐；Stage2/3 实际多为 Aimux 路由 + 本映射兜底）
MODEL_PROVIDER_MAPPING = {
    "gpt-5.4-2026-03-05": "openai",
    "claude-sonnet-4-6": "openai",
    "deepseek-v3.2-thinking": "openai",
    "doubao-seed-2-0-pro": "openai",
    "gemini-3.1-pro-preview": "openai",
    "glm-5.1": "openai",
    "kimi-k2.5": "openai",
    "qwen3.6-plus": "openai",
    "GLM-4.7-flash": "openai",
    "qwen3.5-27b": "openai",
    "step-3.5-flash": "openai",
    "Hunyuan-T1-20250822": "openai",
}

JUDGE_CANDIDATE_MODELS = list(MODEL_PROVIDER_MAPPING.keys())
DEFAULT_TIMEOUT = 120


def aimux_provider_for_origin(origin_code: str) -> ProviderConfig:
    """Aimux 路由占位；无 Aimux 时可改用 openai provider。"""
    _ = origin_code
    base = _env("AIMUX_OPENAI_BASE_URL", "https://aimux.alibaba-inc.com/v1")
    return ProviderConfig(
        name="aimux",
        base_url=base.rstrip("/"),
        api_key=_env("AIMUX_API_KEY", "YOUR_AIMUX_API_KEY"),
    )


def get_provider(provider_name: str) -> ProviderConfig:
    if provider_name not in ALL_CONFIGS:
        raise ValueError(f"未知 provider: {provider_name}，请在 config.py 中配置")
    return ALL_CONFIGS[provider_name]


def get_provider_for_model(model_name: str) -> ProviderConfig:
    if model_name not in MODEL_PROVIDER_MAPPING:
        raise ValueError(f"未配置模型 {model_name} 的 provider 映射")
    return get_provider(MODEL_PROVIDER_MAPPING[model_name])


def get_all_model_configs() -> List[Dict[str, str]]:
    return [{"model": m} for m in MODEL_PROVIDER_MAPPING]
