import hashlib
import json
from pathlib import Path


def load_cases(path):
    raw = Path(path).read_bytes()
    cases, ids = [], set()
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        expected = {"id", "version", "category", "sample_index", "question", "rubric", "expectation"}
        if not isinstance(case, dict) or set(case) != expected:
            raise ValueError("Invalid case schema")
        for field in ("id", "version", "category", "question"):
            if not isinstance(case[field], str) or not case[field].strip():
                raise ValueError("Empty case field")
        if case["id"] in ids or type(case["sample_index"]) is not int or case["sample_index"] < 0:
            raise ValueError("Duplicate case id or invalid sample")
        if not isinstance(case["rubric"], list) or not case["rubric"] or any(not isinstance(x, str) or not x.strip() for x in case["rubric"]):
            raise ValueError("Invalid rubric")
        if not isinstance(case["expectation"], dict) or case["expectation"].get("kind") not in {"window", "rr", "filter", "oversize", "missing", "score"}:
            raise ValueError("Unknown expectation")
        if "knowledge_policy" in case["expectation"] and case["expectation"]["knowledge_policy"] not in ("local_only", "optional"):
            raise ValueError("Unknown knowledge policy")
        cases.append(case)
        ids.add(case["id"])
    if not cases:
        raise ValueError("Empty case suite")
    return cases, hashlib.sha256(raw).hexdigest()
