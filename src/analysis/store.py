"""Bounded process-local snapshots. Not persistent storage or user authentication."""
from copy import deepcopy
from threading import RLock
from .record import AnalysisRecord, utc_now


class AnalysisStore:
    def __init__(self, capacity=32):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._records = {}
        self._results = {}
        self._lock = RLock()

    def begin(self, metadata=None):
        with self._lock:
            if len(self._records) >= self.capacity:
                raise RuntimeError("Analysis store full; explicitly discard old records")
            record = AnalysisRecord(metadata=deepcopy(metadata or {}))
            self._records[record.analysis_id] = record
            return record.analysis_id

    def complete(self, analysis_id, result, metadata=None):
        with self._lock:
            record = self._records[analysis_id]
            if record.status != "running":
                raise ValueError("Analysis is already terminal")
            if result.analysis_id != analysis_id:
                raise ValueError("Result belongs to another analysis")
            snapshot = deepcopy(result)
            record.metadata.update(deepcopy(metadata or {}))
            self._results[analysis_id] = snapshot
            record.status = "succeeded"
            record.finished_at = utc_now()

    def fail(self, analysis_id, error_type):
        with self._lock:
            record = self._records[analysis_id]
            if record.status != "running":
                raise ValueError("Analysis is already terminal")
            record.status = "failed"
            record.error_type = str(error_type)
            record.finished_at = utc_now()

    def get_record(self, analysis_id):
        with self._lock:
            return deepcopy(self._records[analysis_id])

    def get_result(self, analysis_id):
        with self._lock:
            if self._records[analysis_id].status != "succeeded":
                raise ValueError("Analysis has no successful result")
            return deepcopy(self._results[analysis_id])

    def list_records(self):
        with self._lock:
            return deepcopy(list(self._records.values()))

    def discard(self, analysis_id):
        with self._lock:
            if self._records[analysis_id].status == "running":
                raise ValueError("Cannot discard a running analysis")
            del self._records[analysis_id]
            self._results.pop(analysis_id, None)
