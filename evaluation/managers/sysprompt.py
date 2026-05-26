# -*- coding: utf-8 -*-
import os
import pandas as pd
from ..core.utils import safe_str, sanitize_text

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))


class SyspromptManager:
    def __init__(self, sysprompt_excel: str):
        self.sysprompts = self._load(sysprompt_excel)

    def _load(self, excel_path: str) -> dict:
        sysprompts = {}

        if os.path.isabs(excel_path):
            abs_excel_path = excel_path
        else:
            abs_excel_path = os.path.join(_PROJECT_ROOT, excel_path)

        txt_dir = os.path.join(os.path.dirname(abs_excel_path), "sysprompts")
        if os.path.isdir(txt_dir):
            for filename in os.listdir(txt_dir):
                if filename.endswith(".txt"):
                    stage_key = filename[:-4]
                    txt_path = os.path.join(txt_dir, filename)
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                        if content:
                            sysprompts[stage_key] = sanitize_text(content)
                    except Exception as e:
                        print(f"⚠️  读取 {txt_path} 失败: {e}")

            if sysprompts:
                print(f"📖 从 {txt_dir}/ 加载了 {len(sysprompts)} 个 txt sysprompt")

        print(f"📖 读取Sysprompt配置: {abs_excel_path}")

        if not os.path.exists(abs_excel_path):
            print(f"⚠️  Sysprompt Excel 文件不存在，将仅使用 txt 文件配置")
            if sysprompts:
                print(f"✅ 共加载 {len(sysprompts)} 个 sysprompt:")
                for stage, content in sysprompts.items():
                    print(f"  - {stage}: 已配置({len(content)}字)")
                print()
            return sysprompts

        try:
            df = pd.read_excel(abs_excel_path)
        except Exception as e:
            print(f"⚠️  无法读取Sysprompt文件: {e}")
            return sysprompts

        if 'stage' not in df.columns or 'sysprompt' not in df.columns:
            print(f"⚠️  Sysprompt表必须包含 stage 和 sysprompt 列")
            return sysprompts

        for _, row in df.iterrows():
            stage = safe_str(row['stage']).strip()
            raw_value = row['sysprompt']
            sysprompt = safe_str(raw_value).strip()
            if not stage:
                continue
            is_empty = not sysprompt or sysprompt.lower() in ('nan', 'none', 'null', '')
            if is_empty:
                if stage not in sysprompts:
                    print(f"  ⚠️  stage={stage} Excel中sysprompt为空，"
                          f"如需配置请在 data/sysprompts/{stage}.txt 中写入内容")
            else:
                sysprompts[stage] = sanitize_text(sysprompt)

        print(f"✅ 加载 {len(sysprompts)} 个Sysprompt:")
        for stage, content in sysprompts.items():
            print(f"  - {stage}: 已配置({len(content)}字)")
        print()

        return sysprompts

    def get(self, stage: str, default: str = "") -> str:
        return self.sysprompts.get(stage, default)
