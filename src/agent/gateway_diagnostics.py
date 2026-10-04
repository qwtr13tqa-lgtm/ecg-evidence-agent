"""Metadata-only per-request snapshots. No request/response text or credentials."""
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def identifier(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value) else None


def number(value):
    return value if type(value) is int and value >= 0 else None


class GatewayDiagnostics:
    def __init__(self, root=None):
        self.root = Path(root or os.getenv("ECG_GATEWAY_DIAG_DIR") or
                         Path(__file__).resolve().parents[2] / "evaluation/gateway_diagnostics")
        self.session_id = str(uuid.uuid4())

    def save(self, record):
        temporary = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.root / (record["request_id_local"] + ".json")
            fd, temporary = tempfile.mkstemp(prefix="pending_", suffix=".tmp", dir=self.root)
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded(record))
            os.replace(temporary, path)
            return True
        except Exception as exc:
            print("GATEWAY_DIAG_WRITE_FAILED:", type(exc).__name__, flush=True)
            return False
        finally:
            if temporary:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass

    def begin(self, payload, timeout, base_url):
        messages = payload["messages"]
        analysis_id = None
        for message in messages:
            # Bootstrap has an analysis UUID. Do not retain arbitrary user JSON fields.
            content = message.get("content")
            if message.get("role") == "user" and isinstance(content, str):
                try:
                    obj = json.loads(content)
                    if isinstance(obj, dict) and "bootstrap" in obj:
                        analysis_id = str(uuid.UUID(obj["analysis_id"]))
                        break
                except (ValueError, TypeError, KeyError):
                    pass
        body = encoded(payload)
        record = {"schema_version": "gateway-diag-1.0", "session_id": self.session_id,
            "request_id_local": str(uuid.uuid4()), "analysis_id": analysis_id,
            "started_at": now(), "status": "started", "elapsed_seconds": None,
            "model": payload["model"], "timeout_seconds": timeout,
            "max_tokens": payload["max_tokens"], "temperature": payload["temperature"],
            "tool_choice": payload["tool_choice"], "sdk_retries": 0, "stream": False,
            "endpoint_sha256": hashlib.sha256(base_url.encode()).hexdigest(),
            "logical_payload_utf8_bytes": len(body), "payload_sha256": hashlib.sha256(body).hexdigest(),
            "message_count": len(messages), "tools_count": len(payload["tools"]),
            "tool_definitions_utf8_bytes": len(encoded(payload["tools"])),
            "messages": [{"index": i, "role": m.get("role") if m.get("role") in
                          ("system", "user", "assistant", "tool", "developer") else "other",
                          "content_characters": len(m["content"]) if isinstance(m.get("content"), str) else None,
                          "logical_message_utf8_bytes": len(encoded(m))} for i, m in enumerate(messages)],
            "provider_request_id": None, "http_status": None, "finish_reason": None,
            "usage": {}, "time_to_first_token_seconds": None}
        self.save(record)
        print("GATEWAY_REQUEST:", record["request_id_local"], "messages=", len(messages),
              "logical_bytes=", len(body), flush=True)
        return record

    def response(self, record, response):
        record["provider_request_id"] = identifier(getattr(response, "_request_id", None))
        usage = getattr(response, "usage", None)
        record["usage"] = {key: number(getattr(usage, key, None))
                           for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        details = getattr(usage, "completion_tokens_details", None)
        record["usage"]["reasoning_tokens"] = number(getattr(details, "reasoning_tokens", None))
        choices = getattr(response, "choices", None)
        if choices:
            choice = choices[0]
            record["finish_reason"] = identifier(getattr(choice, "finish_reason", None))
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None)
            record["response_content_characters"] = len(content) if isinstance(content, str) else None
            calls = getattr(message, "tool_calls", None) or []
            record["response_tool_count"] = len(calls)
            record["response_tool_argument_characters"] = [len(c.function.arguments)
                if isinstance(getattr(getattr(c, "function", None), "arguments", None), str) else None for c in calls]

    def finish(self, record, started, status, exc=None):
        record.update(status=status, finished_at=now(), elapsed_seconds=time.perf_counter()-started)
        if exc is not None:
            record["error_type"] = type(exc).__name__
            record["provider_request_id"] = identifier(getattr(exc, "request_id", None)) or record["provider_request_id"]
            record["http_status"] = number(getattr(exc, "status_code", None))
        self.save(record)
        print("GATEWAY_RESULT:", record["request_id_local"], status,
              round(record["elapsed_seconds"], 3), "seconds", flush=True)
