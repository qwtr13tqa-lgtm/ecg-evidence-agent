"""Parse final answer text only. Never repair JSON or parse tool arguments here."""
import json
import re
from .tool_protocol import strict_json


class AnswerParseError(ValueError):
    def __init__(self, reason, details):
        super().__init__(reason)
        self.details = {"parse_reason": reason, **details}


def parse_answer(text):
    details = {"content_type": type(text).__name__}
    if not isinstance(text, str):
        raise AnswerParseError("CONTENT_NOT_STRING", details)
    details.update({"content_length": len(text), "has_code_fence": "```" in text,
                    "outer_fence_removed": False})
    body = text.strip()
    if not body:
        raise AnswerParseError("EMPTY_CONTENT", details)
    # Only a complete standalone block is allowed. No explanation before/after it.
    if body.startswith("```"):
        match = re.fullmatch(r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```", body, re.IGNORECASE)
        if match is None:
            raise AnswerParseError("INVALID_OUTER_FENCE", details)
        body = match.group(1)
        details["outer_fence_removed"] = True
    try:
        result = strict_json(body)
    except json.JSONDecodeError as exc:
        raise AnswerParseError("JSON_SYNTAX_ERROR", {
            **details, "json_line": exc.lineno, "json_column": exc.colno,
            "json_position": exc.pos, "position_reference": "parsed_body"}) from None
    except (ValueError, RecursionError):
        # Duplicate keys, nonfinite constants/overflow, or excessive nesting.
        raise AnswerParseError("STRICT_JSON_REJECTED", details) from None
    if not isinstance(result, dict):
        raise AnswerParseError("TOP_LEVEL_NOT_OBJECT", details)
    return result
