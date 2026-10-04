"""OpenAI-compatible native tools adapter. SDK retries disabled."""
import os
import time
from .gateway_diagnostics import GatewayDiagnostics


class ToolGateway:
    def __init__(self, *, timeout=180, max_tokens=1800):
        from .public_config import gateway_config
        config = gateway_config()
        key = config["ECG_API_KEY"]
        from openai import OpenAI
        self.model = config["ECG_MODEL"]
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.base_url = config["ECG_BASE_URL"]
        self.diagnostics = GatewayDiagnostics()
        self.client = OpenAI(api_key=key,
            base_url=self.base_url,
            timeout=timeout, max_retries=0)

    def complete_with_timeout(self, messages, tools, timeout):
        return self.complete(messages, tools, request_timeout=timeout)

    def complete(self, messages, tools, *, request_timeout=None):
        payload = dict(model=self.model, messages=messages, tools=tools,
                       tool_choice="auto", temperature=0, max_tokens=self.max_tokens)
        if not tools:
            payload.pop("tools"); payload.pop("tool_choice")
        effective_timeout=self.timeout if request_timeout is None else request_timeout
        record = self.diagnostics.begin(payload, effective_timeout, self.base_url)
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**payload, timeout=effective_timeout)
        except BaseException as exc:
            self.diagnostics.finish(record, started, "request_failed", exc)
            raise
        try:
            self.diagnostics.response(record, response)
            result = self._adapt(response)
            result["request_metadata"] = {"max_tokens": self.max_tokens,
                "message_count": len(messages),
                "message_bytes": len(__import__("json").dumps(messages, ensure_ascii=False).encode("utf-8"))}
        except Exception as exc:
            self.diagnostics.finish(record, started, "response_adapter_failed", exc)
            raise
        self.diagnostics.finish(record, started, "response_received")
        return result

    @staticmethod
    def _adapt(response):
        if not response.choices:
            raise ValueError("missing choices")
        choice = response.choices[0]
        message = choice.message
        calls = [{"id": call.id, "type": call.type,
                  "function": {"name": call.function.name,
                               "arguments": call.function.arguments}}
                 for call in (message.tool_calls or [])]
        usage = getattr(response, "usage", None)
        usage_data = {key: getattr(usage, key, None) for key in
                      ("prompt_tokens", "completion_tokens", "total_tokens")}
        return {"usage": usage_data, "finish_reason": choice.finish_reason,
                "content": message.content, "tool_calls": calls}

    def close(self):
        self.client.close()
