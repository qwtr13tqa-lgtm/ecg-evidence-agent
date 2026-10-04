"""Application-level binding, not a multi-user security boundary.

Only trusted application code constructs an executor. The model supplies a tool
name and arguments, never an analysis_id or a store. A deleted binding fails.
"""
import hashlib
import json
import time
from copy import deepcopy
from threading import Lock
from . import ecg_tools
from src.analysis.model_decision import get_model_decision
from .window_alignment import inspect_recent_rr_alignment, inspect_window_rr_alignment


class ECGToolExecutor:
    def __init__(self, store, analysis_id):
        store.get_result(analysis_id)  # bind only completed analyses
        self._store = store
        self._analysis_id = analysis_id
        self._history = []
        self._lock = Lock()

    def history(self):
        with self._lock:
            return deepcopy(self._history)

    def execute(self, name, arguments=None):
        started = time.perf_counter()
        tools = {
            'get_model_decision': (get_model_decision, set()),
            'inspect_recent_rr_alignment': (inspect_recent_rr_alignment, {'duration_seconds','lead'}),
            'inspect_window_rr_alignment': (inspect_window_rr_alignment, {'start_sample','end_sample','lead'}),
            "get_analysis_summary": (ecg_tools.get_analysis_summary, set()),
            "inspect_recent_error": (
                ecg_tools.inspect_recent_error, {"duration_seconds", "lead"},
            ),
            "inspect_error_window": (
                ecg_tools.inspect_error_window,
                {"start_sample", "end_sample", "lead"},
            ),
            "inspect_rr_intervals": (
                ecg_tools.inspect_rr_intervals, {"offset", "limit"},
            ),
        }
        response = {"analysis_id": self._analysis_id, "ok": False}
        safe_args = None
        try:
            if not isinstance(name, str) or name not in tools:
                raise ValueError("Unknown tool")
            args = {} if arguments is None else arguments
            function, allowed = tools[name]
            if not isinstance(args, dict) or not set(args) <= allowed:
                raise ValueError("Unexpected arguments; analysis binding is fixed")
            safe_args = json.loads(json.dumps(args, allow_nan=False))
            result = self._store.get_result(self._analysis_id)
            data = function(result, **safe_args)
            json.dumps(data, allow_nan=False)
            identity = json.dumps([name, safe_args], sort_keys=True)
            suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
            response.update({
                "ok": True, "data": data,
                "evidence_id": f"{self._analysis_id}:{suffix}",
                "tool_version": "1.0",
            })
        except (ValueError, TypeError):
            response["error"] = "INVALID_ARGUMENTS_OR_UNAVAILABLE_DATA"
        except KeyError:
            response["error"] = "ANALYSIS_NOT_FOUND"
        except Exception:
            response["error"] = "TOOL_EXECUTION_FAILED"
        event = {
            "analysis_id": self._analysis_id,
            "tool": name if isinstance(name, str) else "invalid",
            "arguments": safe_args, "ok": response["ok"],
            "elapsed_seconds": time.perf_counter() - started,
            "evidence_id": response.get("evidence_id"),
            "error": response.get("error"),
        }
        with self._lock:
            self._history.append(event)
        return response
