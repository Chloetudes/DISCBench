# -*- coding: utf-8 -*-
"""
项目配置加载：支持项目级 config 覆盖全局配置。
- 项目配置文件固定放在 **项目根下的 data/**：`data/config.json` 或 `data/config.yaml`（与 sysprompts.xlsx 等同目录，便于集中管理）
- 可选：`EVALUATION_PROJECT_CONFIG` / `EVAL_PROJECT_CONFIG` 指向 `data/` 下另一文件名或绝对路径，单次运行替代默认 `config.json`。
- 可选：`EVALUATION_DATA_BATCH` / `DATA_BATCH` 指定 `data_batch`，并展开配置中的 `{data_batch}`、`{eval_batch_id}` 占位符（用于 config_prof_expert.json 等通用模板）。
- 可选：`EVALUATION_EVAL_BATCH_ID` 覆盖本轮评测批次列名（如 `batch_prof_0523_gpt54`）；`EVALUATION_JUDGE_PROVIDER` / `EVALUATION_JUDGE_MODEL` 覆盖裁判。
- 产物目录仍为 **output/**（或 output/<project_id>/）：questions、replies、reports 等，不含主配置文件
- 嵌套：project_id 有值时，产物根 = output_base_dir / project_id /
- stages 支持预设代号，减少配置劳动量
"""
import os
import json

# 阶段预设代号：可用 "stages": "full" 或 "stages": ["criteria", "eval_only"]
STAGE_PRESETS = {
    "full": [
        "generate_instructions", "extract_instructions", "evaluate_instructions",
        "expand_multiturn", "promote_to_questions", "generate_criteria",
        "generate_references", "generate_replies", "evaluate_replies",
        "analyze_results", "generate_report",
    ],
    "criteria_ref": ["generate_criteria", "generate_references"],
    "criteria_ref_reply": ["generate_criteria", "generate_references", "generate_replies"],
    "eval_only": ["evaluate_replies", "analyze_results", "generate_report"],
    "reply_eval": ["generate_replies", "evaluate_replies", "analyze_results", "generate_report"],
    "criteria": ["generate_criteria"],
    "references": ["generate_references"],
    "reply": ["generate_replies"],
    "eval": ["evaluate_replies"],
    "analyze": ["analyze_results", "generate_report"],
    # 仅跑专家数据审核统计：只执行 analyze_results（需配合 analysis.stats_only: true），输出一张「专家数据质量与一致性」
    "expert_stats": ["analyze_results"],
    # 专家新题快速检验：若有新题则自动 生成标准→参考→回复→评测→统计
    "expert_quick_check": ["expert_quick_check"],
    # 种子表逐行合成题目 + 多模型回复（如 emo：prompt_generation + generation.see_excel）
    "emo_synth_reply": ["synthesize_from_seeds", "generate_replies"],
    # 仅探测 models_excel 中全部模型行并写回「可用状态」（DISCBench 默认单表 data/models.xlsx）
    "retest_models": ["test_models"],
    # 种子 query+rubrics → 改编题目表（需 data_clone 配置；兼容预设名 surge_clone）
    "data_clone": ["data_clone"],
    "surge_clone": ["data_clone"],
    # 三裁判合议（需 config.tripartite_eval.judges，见 agents/docs/05_TRIPARTITE_JUDGE_NEGOTIATION.md）
    "tripartite_eval": ["evaluate_replies_tripartite"],
    # 多裁判 Agent：与 tripartite_eval 相同阶段名，便于 stages 预设
    "multi_judge_tripartite": ["evaluate_replies_tripartite"],
    # 实验：多批次 eval_*_raw 考点比对 + 分歧考点仲裁（需 batch_arbit_eval，见 agents/docs/03 §5.5）
    "batch_arbit_eval": ["evaluate_replies_batch_arbit"],
    # 标准 + 参考答案门禁（替代一次式 generate_references；需 tripartite_eval + reference_tripartite_gate.writer_* 可选）
    "criteria_ref_gate": ["generate_criteria", "reference_tripartite_gate"],
}


def expand_stages(raw) -> list:
    """
    将 stages 解析为阶段列表。支持：
    - 字符串：预设代号，如 "full"、"criteria_ref"、"eval_only"
    - 列表：可混合预设代号与单阶段名，如 ["criteria", "eval_only"] 或 ["generate_criteria"]
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        s = (item or "").strip()
        if not s:
            continue
        if s in STAGE_PRESETS:
            result.extend(STAGE_PRESETS[s])
        else:
            result.append(s)
    return result


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并：override 的值覆盖 base，嵌套 dict 递归合并。"""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _substitute_batch_tokens(obj, tokens: dict):
    """递归替换配置中的 {data_batch}、{eval_batch_id} 等占位符。"""
    if isinstance(obj, str):
        out = obj
        for key, val in tokens.items():
            if val is not None:
                out = out.replace(f"{{{key}}}", str(val))
        return out
    if isinstance(obj, dict):
        return {k: _substitute_batch_tokens(v, tokens) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_batch_tokens(v, tokens) for v in obj]
    return obj


def _has_unresolved_batch_tokens(obj) -> bool:
    if isinstance(obj, str) and "{" in obj and "data_batch" in obj:
        return True
    if isinstance(obj, dict):
        return any(_has_unresolved_batch_tokens(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_unresolved_batch_tokens(v) for v in obj)
    return False


def _apply_data_batch(config: dict) -> dict:
    """
    解析 data_batch：优先 EVALUATION_DATA_BATCH / DATA_BATCH 环境变量，其次 config.data_batch。
    将 {data_batch}、{eval_batch_id} 占位符展开到全配置。
    """
    env_batch = (
        os.environ.get("EVALUATION_DATA_BATCH")
        or os.environ.get("DATA_BATCH")
        or ""
    ).strip()
    cfg_batch = str(config.get("data_batch") or "").strip()
    if cfg_batch in ("{data_batch}", ""):
        cfg_batch = ""
    batch = env_batch or cfg_batch
    if not batch:
        if _has_unresolved_batch_tokens(config):
            raise ValueError(
                "配置含 {data_batch} 占位符但未指定批次。"
                "请设置环境变量 EVALUATION_DATA_BATCH=prof_0523（或 config.data_batch）。"
            )
        return config
    tokens = {
        "data_batch": batch,
        "eval_batch_id": f"batch_{batch}",
    }
    config = _substitute_batch_tokens(config, tokens)
    config["data_batch"] = batch
    return config


def _apply_runtime_env_overrides(config: dict) -> dict:
    """单次运行覆盖：EVALUATION_EVAL_BATCH_ID、EVALUATION_JUDGE_PROVIDER、EVALUATION_JUDGE_MODEL。"""
    eval_batch = (
        os.environ.get("EVALUATION_EVAL_BATCH_ID")
        or os.environ.get("EVAL_BATCH_ID")
        or ""
    ).strip()
    if eval_batch:
        config["eval_batch_id"] = eval_batch
        config["batch_id"] = eval_batch
        if isinstance(config.get("analysis"), dict):
            config["analysis"]["eval_batch_id"] = eval_batch
        if isinstance(config.get("report"), dict):
            config["report"]["eval_batch_id"] = eval_batch

    provider = (os.environ.get("EVALUATION_JUDGE_PROVIDER") or "").strip()
    model = (os.environ.get("EVALUATION_JUDGE_MODEL") or "").strip()
    if provider:
        config["provider"] = provider
    if model:
        config["model"] = model
    return config


def get_project_output_directory(project_root: str, output_base_dir: str, project_id: str) -> str:
    """
    项目产物根目录（其下含 questions/、replies/、reports/ 等；**不含** config.json，配置在 data/）。
    规则：有 project_id 则多一层子目录，否则为扁平 output_base_dir。
    """
    flat_base = os.path.join(project_root, output_base_dir)
    pid = (project_id or "").strip()
    if pid:
        return os.path.join(flat_base, pid)
    return flat_base


def load_config_file(path: str) -> dict:
    """加载单个 JSON 或 YAML 配置文件。不存在或读失败则返回空 dict。"""
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        if not content.strip():
            return {}
        if path.endswith('.json'):
            return json.loads(content)
        try:
            import yaml
            return yaml.safe_load(content) or {}
        except ImportError:
            pass
    except Exception as e:
        print(f"⚠️  加载项目配置失败 {path}: {e}")
    return {}


def load_project_config(config_dir: str) -> dict:
    """
    从目录加载 config.json 或 config.yaml。不存在则返回空 dict。
    config_dir: 含配置文件的目录（通常为项目根的 data/，绝对路径）
    """
    for name in ('config.json', 'config.yaml'):
        path = os.path.join(config_dir, name)
        if os.path.isfile(path):
            return load_config_file(path)
    return {}


def resolve_config(
    global_config: dict,
    project_root: str,
    output_base_dir: str = "output",
) -> dict:
    """
    解析最终配置：合并全局配置与 **项目根/data/config.json**（或 config.yaml），
    或通过 EVALUATION_PROJECT_CONFIG 指定的配置文件。
    - 配置文件路径：project_root/data/config.json | data/config.yaml（与业务输入、sysprompt 同放 data/）
    - 产物路径：project_root/output/... 仍由 project_id + output_base_dir 决定（与配置文件位置无关）
    - sysprompt_excel 未在项目 config 里显式设置时，若 data/sysprompts.xlsx 存在则自动使用
    - 项目 config 中的相对路径相对于 project_root 解析（与 Pipeline 一致）
    """
    config = dict(global_config)
    project_id = (config.get('project_id') or '').strip()
    # 阶段预设展开（有无项目均执行）
    raw_stages = config.get('stages')
    if raw_stages is not None:
        expanded = expand_stages(raw_stages)
        if expanded:
            config['stages'] = expanded

    data_dir = os.path.join(project_root, 'data')
    env_override = (
        os.environ.get("EVALUATION_PROJECT_CONFIG")
        or os.environ.get("EVAL_PROJECT_CONFIG")
        or ""
    ).strip()
    explicit_cfg_path = ""
    if env_override:
        explicit_cfg_path = (
            env_override
            if os.path.isabs(env_override)
            else os.path.join(data_dir, env_override)
        )
    has_default_cfg = (
        os.path.isfile(os.path.join(data_dir, 'config.json'))
        or os.path.isfile(os.path.join(data_dir, 'config.yaml'))
    )
    if not has_default_cfg and not (explicit_cfg_path and os.path.isfile(explicit_cfg_path)):
        return config

    if explicit_cfg_path and os.path.isfile(explicit_cfg_path):
        project_cfg = load_config_file(explicit_cfg_path)
        rel = os.path.relpath(explicit_cfg_path, start=os.getcwd())
        print(f"  📖 已加载项目配置（EVALUATION_PROJECT_CONFIG）: {rel}")
    else:
        if env_override and explicit_cfg_path and not os.path.isfile(explicit_cfg_path):
            print(
                f"⚠️  EVALUATION_PROJECT_CONFIG 未找到: {explicit_cfg_path}，"
                f"回退为 data/config.json|yaml\n"
            )
        project_cfg = load_project_config(data_dir)
        if project_cfg and not (explicit_cfg_path and os.path.isfile(explicit_cfg_path)):
            cfg_name = (
                'config.json'
                if os.path.isfile(os.path.join(data_dir, 'config.json'))
                else 'config.yaml'
            )
            rel = os.path.relpath(os.path.join(data_dir, cfg_name), start=os.getcwd())
            print(f"  📖 已加载项目配置: {rel}")
    if project_cfg:
        config = _deep_merge(config, {k: v for k, v in project_cfg.items()
                                      if not (isinstance(k, str) and k.startswith('_'))})

    config = _apply_data_batch(config)

    config = _apply_runtime_env_overrides(config)

    # 项目级 sysprompt：若未在项目 config 中显式设置，且 data/sysprompts.xlsx 存在则使用
    if not project_cfg or 'sysprompt_excel' not in project_cfg:
        default_sysprompt = os.path.join(data_dir, 'sysprompts.xlsx')
        if os.path.isfile(default_sysprompt):
            config['sysprompt_excel'] = os.path.abspath(default_sysprompt)
            print(f"  📖 使用项目提示词: {default_sysprompt}")
    elif config.get('sysprompt_excel') and not os.path.isabs(config['sysprompt_excel']):
        # 与 PipelineManager._resolve_path 一致：相对路径相对于 project_root（如 projects/tom/），
        # 而非 outputs/<project_id>/，以便与 data/sysprompts.xlsx、init 脚手架一致。
        config['sysprompt_excel'] = os.path.abspath(
            os.path.join(project_root, config['sysprompt_excel'])
        )

    # 合并后再次展开 stages（项目 config 可能覆盖了 stages）
    raw_stages = config.get('stages')
    if raw_stages is not None:
        expanded = expand_stages(raw_stages)
        if expanded:
            config['stages'] = expanded

    # ========= 配置归一化：减少重复参数 =========
    # data_batch：文件后缀批次（questions_{batch}.xlsx / replies_{batch}.xlsx），与 eval_batch_id 不同概念
    # eval_batch_id / batch_id / analysis.eval_batch_id / report.eval_batch_id：本质都是“使用哪一列 eval_{id}”
    # 统一为一个来源：优先 root.eval_batch_id，其次 root.batch_id，其次 analysis/report 中已有值
    def _pick_first(*vals):
        for v in vals:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
        return ''

    analysis_cfg = config.get('analysis') if isinstance(config.get('analysis'), dict) else {}
    report_cfg = config.get('report') if isinstance(config.get('report'), dict) else {}

    resolved_eval_batch_id = _pick_first(
        config.get('eval_batch_id'),
        config.get('batch_id'),
        analysis_cfg.get('eval_batch_id'),
        report_cfg.get('eval_batch_id'),
    )
    if resolved_eval_batch_id:
        # 统计/报告默认使用同一批次；若用户需要刻意分开，仍可显式覆盖（这里不强制覆盖非空值）
        config['eval_batch_id'] = resolved_eval_batch_id
        if isinstance(config.get('analysis'), dict):
            config['analysis'].setdefault('eval_batch_id', resolved_eval_batch_id)
        else:
            config['analysis'] = {'eval_batch_id': resolved_eval_batch_id}
        if isinstance(config.get('report'), dict):
            config['report'].setdefault('eval_batch_id', resolved_eval_batch_id)
        else:
            config['report'] = {'eval_batch_id': resolved_eval_batch_id}
        # evaluate_replies 默认 batch_id 与 eval_batch_id 对齐，避免“评估写一列、统计读另一列”
        if not (config.get('batch_id') and str(config.get('batch_id')).strip()):
            config['batch_id'] = resolved_eval_batch_id

    return config
