"""Analysis lifecycle metadata; no probabilities or clinical claims."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AnalysisRecord:
    analysis_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now)
    finished_at: Optional[str] = None
    status: str = "running"
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_type: Optional[str] = None
    schema_version: str = "1.0"
