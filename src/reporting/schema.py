"""结构化辅助分析报告的数据接口。"""

from dataclasses import asdict, dataclass, field
from typing import Any, List


@dataclass
class Observation:
    id: str
    evidence_path: str
    value: Any


@dataclass
class Explanation:
    id: str
    text: str
    observation_ids: List[str]
    knowledge_ids: List[str]


@dataclass
class ReportLimitation:
    code: str
    message: str


@dataclass
class ECGReport:
    summary: str

    observations: List[Observation] = field(default_factory=list)
    explanations: List[Explanation] = field(default_factory=list)
    limitations: List[ReportLimitation] = field(default_factory=list)

    schema_version: str = "0.1.0"
    report_type: str = "ecg_auxiliary_analysis"

    def to_dict(self):
        return asdict(self)