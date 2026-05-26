# -*- coding: utf-8 -*-
import os


class DirectoryManager:
    """
    目录管理：支持 project_id 按项目批次隔离输出。
    - project_id 为空：base_dir 下直接建 questions/replies/reports 等
    - project_id 有值（如 "prof", "batch_400"）：base_dir/{project_id}/questions 等，同一项目所有输出集中在一个文件夹
    - project_root：项目根目录（绝对路径），传入时 dirs 为绝对路径，确保与 cwd 无关
    """
    def __init__(self, base_dir: str = "output", project_id: str = None, project_root: str = None):
        self.base_dir = base_dir
        self.project_id = (project_id or "").strip()
        self._root = f"{base_dir}/{self.project_id}" if self.project_id else base_dir
        prefix = os.path.abspath(project_root) if project_root else ""
        self.dirs = {
            "stage0_generation": _join(prefix, f"{self._root}/stage0_generation"),
            "stage0.5_extraction": _join(prefix, f"{self._root}/stage0.5_extraction"),
            "stage0.7_multiturn": _join(prefix, f"{self._root}/stage0.7_multiturn"),
            "stage1_quality": _join(prefix, f"{self._root}/stage1_quality"),
            "questions": _join(prefix, f"{self._root}/questions"),
            "replies": _join(prefix, f"{self._root}/replies"),
            "evaluations": _join(prefix, f"{self._root}/evaluations"),
            "library": _join(prefix, f"{base_dir}/library"),
            "reports": _join(prefix, f"{self._root}/reports"),
        }
        self._create_directories()

    def _create_directories(self):
        for dir_path in self.dirs.values():
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

    def get_path(self, stage: str, filename: str) -> str:
        if stage not in self.dirs:
            raise ValueError(f"未知的阶段目录: {stage}，可用目录: {list(self.dirs.keys())}")
        return os.path.join(self.dirs[stage], filename)


def _join(prefix: str, path: str) -> str:
    if not prefix:
        return path
    return os.path.normpath(os.path.join(prefix, path))
