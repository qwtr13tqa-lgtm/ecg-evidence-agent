"""Unique run directories, atomic snapshots, immutable terminal result."""
import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix="write_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class RunRecord:
    def __init__(self, root, case, config, *, data_kind="real_ecg"):
        self.run_id = str(uuid.uuid4())
        self.path = Path(root).resolve() / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.base = {"schema_version": "stage3-1.0", "run_id": self.run_id,
                     "created_at": utc_now(), "dataset_role": "development",
                     "data_kind": data_kind, "case": case, "config": config}
        self.closed = False
        self.checkpoint("created")

    def checkpoint(self, stage):
        if self.closed:
            raise ValueError("Run already finished")
        atomic_json(self.path / "progress.json", {**self.base, "stage": stage, "updated_at": utc_now()})

    def finish(self, output, reference, scores, *, failure=None, wall_seconds=None):
        if self.closed or (self.path / "result.json").exists():
            raise ValueError("Run already finished")
        result = {**self.base, "finished_at": utc_now(), "output": output,
                  "reference": reference, "automatic": scores,
                  "runner_failure": failure, "wall_seconds": wall_seconds}
        atomic_json(self.path / "result.json", result)
        self.closed = True
        digest = hashlib.sha256((self.path / "result.json").read_bytes()).hexdigest()
        atomic_json(self.path / "review.json", {
            "run_id": self.run_id, "result_sha256": digest, "reviewer": "", "notes": "",
            "task_correct": None, "evidence_support": None, "text_complete": None})
        atomic_json(self.path / "progress.json", {**self.base, "stage": "finished", "updated_at": utc_now()})
        return self.path


def read_review(directory):
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    path = directory / "review.json"
    if not path.exists():
        return {"status": "pending", "scores": {}}
    review = json.loads(path.read_text(encoding="utf-8"))
    if set(review) != {"run_id", "result_sha256", "reviewer", "notes", "task_correct", "evidence_support", "text_complete"}:
        raise ValueError("Invalid review schema")
    if review["run_id"] != result["run_id"] or review["result_sha256"] != hashlib.sha256((directory / "result.json").read_bytes()).hexdigest():
        raise ValueError("Review/result mismatch")
    scores = {key: review[key] for key in ("task_correct", "evidence_support", "text_complete")}
    if any(value not in (None, "pass", "fail") for value in scores.values()):
        raise ValueError("Review scores must be null/pass/fail")
    if not isinstance(review["reviewer"], str) or not isinstance(review["notes"], str):
        raise ValueError("Invalid reviewer fields")
    if any(value is not None for value in scores.values()) and not review["reviewer"].strip():
        raise ValueError("Reviewer required for scoring")
    return {"status": "complete" if all(v is not None for v in scores.values()) else "pending",
            "scores": scores, "reviewer": review["reviewer"], "notes": review["notes"]}
