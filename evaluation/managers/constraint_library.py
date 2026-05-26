# -*- coding: utf-8 -*-
import json
import os
from threading import Lock
from typing import Dict, List, Optional

import pandas as pd

from ..core.types import Constraint
from ..core.utils import safe_str, safe_save_excel


class ConstraintLibraryManager:
    def __init__(self, library_path: str):
        self.library_path = library_path
        self.constraints: Dict[str, Constraint] = {}
        self.lock = Lock()
        self._load()

    def _load(self):
        if not os.path.exists(self.library_path):
            print(f"⚠️  约束库文件不存在: {self.library_path}，将创建新的约束库")
            return

        try:
            df = pd.read_excel(self.library_path)
            required_cols = ['id', 'content', 'type', 'subtype', 'weight', 'evaluation_method']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"⚠️  约束库缺少必需列: {', '.join(missing_cols)}")
                return

            with self.lock:
                for _, row in df.iterrows():
                    params = {}
                    if 'params' in df.columns:
                        params_str = safe_str(row['params'])
                        if params_str and params_str.lower() != 'nan':
                            try:
                                params = json.loads(params_str)
                            except Exception:
                                params = {}

                    constraint = Constraint(
                        id=safe_str(row['id']),
                        content=safe_str(row['content']),
                        type=safe_str(row['type']),
                        subtype=safe_str(row['subtype']),
                        weight=float(row['weight']) if pd.notna(row['weight']) else 1.0,
                        evaluation_method=safe_str(row['evaluation_method']),
                        params=params
                    )
                    self.constraints[constraint.id] = constraint

            print(f"✅ 加载约束库: {len(self.constraints)} 条约束")
        except Exception as e:
            print(f"⚠️  加载约束库失败: {e}")

    def save(self):
        with self.lock:
            if not self.constraints:
                print("⚠️  约束库为空，跳过保存")
                return

            data = [
                {
                    'id': c.id,
                    'content': c.content,
                    'type': c.type,
                    'subtype': c.subtype,
                    'weight': c.weight,
                    'evaluation_method': c.evaluation_method,
                    'params': json.dumps(c.params, ensure_ascii=False) if c.params else ''
                }
                for c in self.constraints.values()
            ]
            df = pd.DataFrame(data)
            if safe_save_excel(df, self.library_path):
                print(f"✅ 约束库已保存: {self.library_path}")
            else:
                print(f"❌ 约束库保存失败")

    def add(self, constraint: Constraint):
        with self.lock:
            self.constraints[constraint.id] = constraint

    def get(self, constraint_id: str) -> Optional[Constraint]:
        with self.lock:
            return self.constraints.get(constraint_id)

    def get_all(self) -> List[Constraint]:
        with self.lock:
            return list(self.constraints.values())

    def get_by_type(self, constraint_type: str) -> List[Constraint]:
        with self.lock:
            return [c for c in self.constraints.values() if c.type == constraint_type]

    def get_by_subtype(self, subtype: str) -> List[Constraint]:
        with self.lock:
            return [c for c in self.constraints.values() if c.subtype == subtype]

    def remove(self, constraint_id: str):
        with self.lock:
            self.constraints.pop(constraint_id, None)

    def clear(self):
        with self.lock:
            self.constraints.clear()

    def print_summary(self):
        with self.lock:
            total = len(self.constraints)
            by_type: Dict[str, int] = {}
            by_subtype: Dict[str, int] = {}
            for c in self.constraints.values():
                by_type[c.type] = by_type.get(c.type, 0) + 1
                by_subtype[c.subtype] = by_subtype.get(c.subtype, 0) + 1

        print(f"\n{'=' * 60}")
        print(f"📚 约束库统计")
        print(f"{'=' * 60}")
        print(f"  总约束数: {total}")
        if by_type:
            print(f"\n  按类型分布:")
            for ctype, count in sorted(by_type.items()):
                print(f"    - {ctype}: {count} 条")
        if by_subtype:
            print(f"\n  按子类型分布:")
            for subtype, count in sorted(by_subtype.items()):
                print(f"    - {subtype}: {count} 条")
        print(f"{'=' * 60}\n")
