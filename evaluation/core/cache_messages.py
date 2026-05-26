# -*- coding: utf-8 -*-
import json
from typing import List, Dict


def detect_provider_type(provider: str, model: str) -> str:
    provider_lower = provider.lower()
    model_lower = model.lower()

    if 'claude' in model_lower or 'anthropic' in provider_lower:
        return 'claude'
    if 'gpt' in model_lower or 'openai' in provider_lower:
        return 'openai'
    if 'gemini' in model_lower or 'google' in provider_lower:
        return 'gemini'
    return 'other'


def _build_reference_note(reference_type: str) -> str:
    if reference_type in ('human', 'reality'):
        return """
参考答案使用说明：
此参考答案是质量基准，不是唯一正确答案
模型回复可以与参考答案表达方式不同（emoji、语气、措辞等）
模型回复可以与参考答案格式不同（换行、标点、组织方式等）
评分依据是评分标准，不是参考答案
只要符合评分标准，即使与参考答案不同也不应扣分
只有在信息缺失、信息错误、违反约束时才扣分
"""
    return """
参考答案使用说明：
此参考答案由模型生成，仅作辅助参考
不应作为评分依据
仅用于启发思路和对比质量
模型回复质量应完全基于评分标准判断
"""


def _format_history_context(history_context: str) -> str:
    if not history_context or history_context.strip() in ('', '[]', 'nan', 'none', 'null'):
        return ''

    try:
        history = json.loads(history_context)
        if not isinstance(history, list) or len(history) == 0:
            return ''

        lines = ['【历史对话记录】']
        for turn in history:
            turn_num = turn.get('turn', '')
            user_text = turn.get('user', '')
            assistant_text = turn.get('assistant', '')
            lines.append(f"\n[第{turn_num}轮]")
            if user_text:
                lines.append(f"用户：{user_text}")
            if assistant_text:
                lines.append(f"助手：{assistant_text}")

        return '\n'.join(lines)
    except (json.JSONDecodeError, TypeError):
        return ''


def merge_conversation_sysprompt(base_sys_prompt: str, conversation_sysprompt: str) -> str:
    """题目级角色/场景设定优先置于裁判 system 之前（与 base 评测 sysprompt 拼接）。"""
    c = (conversation_sysprompt or "").strip()
    b = (base_sys_prompt or "").strip()
    if c and b:
        return f"{c}\n\n---\n\n{b}"
    return c or b


def _build_cached_context(
    query: str,
    evaluation_criteria: str,
    reference: str,
    reference_type: str,
    history_context: str = '',
    expert_opinions: str = '',
    conversation_sysprompt: str = '',
    ground_truth: str = '',
) -> str:
    history_section = _format_history_context(history_context)

    context = f"请评估以下AI模型对指令的回复质量。\n"

    if history_section:
        context += f"\n{history_section}\n"

    if expert_opinions and str(expert_opinions).strip():
        context += f"""
【专家参考（仅供对齐打分尺度）】
{expert_opinions.strip()}

"""
    sp = (conversation_sysprompt or "").strip()
    if sp:
        context += f"""【题目级系统提示（答题模型按此设定 + 下述用户指令作答）】
{sp}

"""
    context += f"""【原始指令 / 用户 query】
{query}

【评估标准】
{evaluation_criteria}
"""
    gt = (ground_truth or "").strip()
    if gt and gt.lower() not in ('nan', 'none', 'null'):
        context += f"""
【ground_truth（数据方提供的答题方向或核心要点，未必是完整可提交答案；评分以评估标准为准，勿与下述「参考答案」混为一谈）】
{gt}

"""
    has_reference = (
        reference
        and reference.lower() != 'nan'
        and reference.strip()
    )
    if has_reference:
        context += f"""
【参考答案】
{reference}

{_build_reference_note(reference_type)}
"""
    return context


def _build_reply_content(reply: str, model_name: str) -> str:
    return f"""
【待评估回复】
模型: {model_name}
内容:
{reply}

请严格按照system prompt中的格式要求输出评估结果。
"""


def build_cached_messages_claude(
    sys_prompt: str, query: str, evaluation_criteria: str,
    reference: str, reply: str, model_name: str,
    reference_type: str = 'model', history_context: str = '',
    expert_opinions: str = '', conversation_sysprompt: str = '',
    ground_truth: str = '',
) -> List[Dict]:
    cached_context = _build_cached_context(
        query, evaluation_criteria, reference, reference_type, history_context, expert_opinions,
        conversation_sysprompt=conversation_sysprompt,
        ground_truth=ground_truth,
    )
    return [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": cached_context,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        },
        {"role": "user", "content": _build_reply_content(reply, model_name)}
    ]


def build_cached_messages_openai(
    sys_prompt: str, query: str, evaluation_criteria: str,
    reference: str, reply: str, model_name: str,
    reference_type: str = 'model', history_context: str = '',
    expert_opinions: str = '', conversation_sysprompt: str = '',
    ground_truth: str = '',
) -> List[Dict]:
    cached_context = _build_cached_context(
        query, evaluation_criteria, reference, reference_type, history_context, expert_opinions,
        conversation_sysprompt=conversation_sysprompt,
        ground_truth=ground_truth,
    )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": cached_context},
        {"role": "user", "content": _build_reply_content(reply, model_name)}
    ]


def build_cached_messages_gemini(
    sys_prompt: str, query: str, evaluation_criteria: str,
    reference: str, reply: str, model_name: str,
    reference_type: str = 'model', history_context: str = '',
    expert_opinions: str = '', conversation_sysprompt: str = '',
    ground_truth: str = '',
) -> List[Dict]:
    cached_context = _build_cached_context(
        query, evaluation_criteria, reference, reference_type, history_context, expert_opinions,
        conversation_sysprompt=conversation_sysprompt,
        ground_truth=ground_truth,
    )
    return [
        {"role": "system", "content": sys_prompt, "cache": True},
        {"role": "user", "content": cached_context, "cache": True},
        {"role": "user", "content": _build_reply_content(reply, model_name)}
    ]


def build_cached_messages(
    provider_type: str, sys_prompt: str, query: str,
    evaluation_criteria: str, reference: str, reply: str,
    model_name: str, reference_type: str = 'model',
    history_context: str = '', expert_opinions: str = '',
    conversation_sysprompt: str = '',
    ground_truth: str = '',
) -> List[Dict]:
    builders = {
        'claude': build_cached_messages_claude,
        'openai': build_cached_messages_openai,
        'gemini': build_cached_messages_gemini,
    }
    builder = builders.get(provider_type)
    if builder:
        return builder(
            sys_prompt, query, evaluation_criteria, reference,
            reply, model_name, reference_type, history_context, expert_opinions,
            conversation_sysprompt,
            ground_truth,
        )

    cached_context = _build_cached_context(
        query, evaluation_criteria, reference, reference_type, history_context, expert_opinions,
        conversation_sysprompt=conversation_sysprompt,
        ground_truth=ground_truth,
    )
    user_content = cached_context + _build_reply_content(reply, model_name)
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content}
    ]
