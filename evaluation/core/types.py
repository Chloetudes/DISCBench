# -*- coding: utf-8 -*-
from dataclasses import dataclass, field


@dataclass
class Constraint:
    id: str
    content: str
    type: str
    subtype: str
    weight: float
    evaluation_method: str
    params: dict = field(default_factory=dict)
