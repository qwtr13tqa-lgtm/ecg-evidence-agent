"""报告结构、数值及引用完整性校验。

不校验医学正确性、自然语言推理或引用蕴含关系。
"""

import json
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class ValidationResult:
    errors: List[str] = field(default_factory=list)

    @property
    def passed(self):
        return not self.errors

    def to_dict(self):
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "scope": "structure_values_references_only",
            "semantic_support_checked": False,
            "medical_correctness_checked": False,
        }


def resolve_pointer(context, pointer):
    """解析 JSON Pointer；缺失或非法路径直接报错。"""
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("Expected an absolute JSON Pointer")

    value = context

    for token in pointer[1:].split("/"):
        if re.search(r"~(?![01])", token):
            raise ValueError("Invalid JSON Pointer escape")

        token = token.replace("~1", "/").replace("~0", "~")

        if isinstance(value, dict):
            value = value[token]

        elif isinstance(value, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", token):
                raise ValueError("Invalid array index")
            value = value[int(token)]

        else:
            raise ValueError("Cannot traverse a scalar value")

    return value


def validate_report(report, context):
    """接收普通字典，便于未来直接检查 LLM 的 JSON 输出。"""
    result = ValidationResult()
    errors = result.errors

    # 严格限定为标准 JSON 数据。
    try:
        report = json.loads(json.dumps(report, allow_nan=False))
        context = json.loads(json.dumps(context, allow_nan=False))
    except (TypeError, ValueError, OverflowError):
        errors.append("Report or context is not valid JSON data")
        return result

    required = {
        "summary",
        "observations",
        "explanations",
        "limitations",
        "schema_version",
        "report_type",
    }

    if not isinstance(report, dict) or set(report) != required:
        errors.append("Invalid report fields")
        return result

    if report["schema_version"] != "0.1.0":
        errors.append("Unsupported schema_version")

    if report["report_type"] != "ecg_auxiliary_analysis":
        errors.append("Invalid report_type")

    if not isinstance(report["summary"], str) or not report["summary"].strip():
        errors.append("summary must be a non-empty string")

    for name in ("observations", "explanations", "limitations"):
        if not isinstance(report[name], list):
            errors.append(f"{name} must be a list")

    if errors:
        return result

    # 校验依赖的 Context 元数据。缺失时不静默放行。
    try:
        policy = context["interpretation"]
        expected_limits = policy["limitations"]
        documents = context["retrieved_knowledge"]["documents"]

        if not isinstance(expected_limits, list) or not expected_limits:
            raise ValueError("Missing limitations")

        if not isinstance(documents, list):
            raise ValueError("Invalid documents")

        expected_by_code = {}
        for item in expected_limits:
            code, message = item["code"], item["message"]
            if (
                not isinstance(code, str)
                or not code.strip()
                or not isinstance(message, str)
                or not message.strip()
                or code in expected_by_code
            ):
                raise ValueError("Invalid limitation")
            expected_by_code[code] = message

        knowledge_ids = set()
        for item in documents:
            chunk_id = item["id"]
            if (
                not isinstance(chunk_id, str)
                or not chunk_id.strip()
                or chunk_id in knowledge_ids
            ):
                raise ValueError("Invalid knowledge id")
            knowledge_ids.add(chunk_id)

    except (KeyError, TypeError, ValueError):
        errors.append("Invalid interpretation or retrieval metadata")
        return result

    # 观察只能引用实际测量/证据，不可引用知识文本冒充测量。
    allowed_roots = {
        "input",
        "model",
        "evidence",
        "signal_features",
        "confidence",
    }

    observation_ids = set()

    if not report["observations"]:
        errors.append("At least one observation is required")

    for index, item in enumerate(report["observations"]):
        label = f"observations[{index}]"

        if not isinstance(item, dict) or set(item) != {
            "id", "evidence_path", "value"
        }:
            errors.append(f"{label}: invalid fields")
            continue

        observation_id = item["id"]

        if (
            not isinstance(observation_id, str)
            or not observation_id.strip()
            or observation_id in observation_ids
        ):
            errors.append(f"{label}: invalid or duplicate id")
            continue

        observation_ids.add(observation_id)
        path = item["evidence_path"]

        try:
            if (
                not isinstance(path, str)
                or not path.startswith("/")
                or path.split("/")[1] not in allowed_roots
            ):
                raise ValueError("Disallowed evidence root")

            expected = resolve_pointer(context, path)

            # 当前只允许引用叶子值，避免复制整段 Context。
            if isinstance(expected, (dict, list)):
                raise ValueError("Evidence must be a scalar")

            # 严格数值复制：不允许舍入，也避免 True 被当作 1。
            actual = item["value"]
            if type(actual) is not type(expected) or actual != expected:
                errors.append(f"{label}: value differs from context")

        except (KeyError, IndexError, TypeError, ValueError):
            errors.append(f"{label}: invalid evidence_path")

    explanation_ids = set()

    for index, item in enumerate(report["explanations"]):
        label = f"explanations[{index}]"

        if not isinstance(item, dict) or set(item) != {
            "id", "text", "observation_ids", "knowledge_ids"
        }:
            errors.append(f"{label}: invalid fields")
            continue

        explanation_id = item["id"]

        if (
            not isinstance(explanation_id, str)
            or not explanation_id.strip()
            or explanation_id in explanation_ids
        ):
            errors.append(f"{label}: invalid or duplicate id")
        else:
            explanation_ids.add(explanation_id)

        if not isinstance(item["text"], str) or not item["text"].strip():
            errors.append(f"{label}: empty text")

        for field_name, valid_ids in (
            ("observation_ids", observation_ids),
            ("knowledge_ids", knowledge_ids),
        ):
            references = item[field_name]

            if (
                not isinstance(references, list)
                or not references
                or any(
                    not isinstance(ref, str) or ref not in valid_ids
                    for ref in references
                )
            ):
                errors.append(f"{label}: invalid {field_name}")
            elif len(references) != len(set(references)):
                errors.append(f"{label}: duplicate {field_name}")

    # 限制集合必须完整一致，不能删减或替换原文。
    supplied_limits = {}

    for item in report["limitations"]:
        if not isinstance(item, dict) or set(item) != {"code", "message"}:
            errors.append("Invalid limitation fields")
            continue

        code = item["code"]

        if not isinstance(code, str) or code in supplied_limits:
            errors.append("Invalid or duplicate limitation code")
            continue

        supplied_limits[code] = item["message"]

    if supplied_limits != expected_by_code:
        errors.append("Limitations differ from required interpretation")

    return result