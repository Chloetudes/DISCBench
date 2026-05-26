# -*- coding: utf-8 -*-
import sys
import os

_this_init_file = os.path.abspath(__file__)
_evaluation_pkg_dir = os.path.dirname(_this_init_file)
_project_root = os.path.dirname(_evaluation_pkg_dir)

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
