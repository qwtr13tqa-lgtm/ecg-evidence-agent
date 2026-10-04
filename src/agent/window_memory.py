"""Recover historical window objects, never historical measurement values."""
import json
import re
from copy import deepcopy

WINDOW_TOOLS = {"inspect_error_window", "inspect_recent_error"}


class WindowReferenceError(ValueError):
    pass


def _number(value):
    if value.isdigit():
        return int(value)
    digits = dict(zip("零一二三四五六七八九", range(10)))
    if value == "两":
        return 2
    if "十" in value:
        left, right = value.split("十", 1)
        return (digits.get(left, 1) * 10) + digits.get(right, 0)
    return digits.get(value)


def referenced_turn(question):
    if not re.search(r"窗口|window", question, re.I):
        return None
    matches = re.findall(
        r"第\s*([0-9零一二两三四五六七八九十]+)\s*轮",
        question,
    )
    numbers = [_number(value) for value in matches]
    numbers += [
        int(value)
        for value in re.findall(r"\bturn\s+(\d+)\b", question, re.I)
    ]
    if re.search(r"\bfirst\s+(?:turn|round)\b", question, re.I):
        numbers.append(1)
    if not numbers:
        return None
    unique = set(numbers)
    if len(unique) != 1 or None in unique or min(unique) < 1:
        return -1
    return numbers[0]


def _lead(value):
    if not isinstance(value, str):
        return None
    mapping = {
        name.upper(): name
        for name in ["I", "II", "III", "aVR", "aVL", "aVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"]
    }
    return mapping.get(value.strip().upper())


def windows_from_data(data, arguments=None):
    """Read explicit returned coordinates; never infer a window from prose."""
    arguments = arguments or {}
    fallback = _lead(arguments.get("lead"))
    found = set()

    def walk(node, inherited=None):
        if isinstance(node, list):
            for value in node:
                walk(value, inherited)
            return
        if not isinstance(node, dict):
            return
        lead = (
            _lead(node.get("lead"))
            or _lead(node.get("lead_name"))
            or inherited
            or fallback
        )
        start, end = node.get("start_sample"), node.get("end_sample")
        if (lead is not None and type(start) is int and type(end) is int
                and 0 <= start < end):
            found.add((lead, start, end))
        for value in node.values():
            if isinstance(value, (dict, list)):
                walk(value, lead)

    walk(data)
    return [
        {"lead": lead, "start_sample": start, "end_sample": end}
        for lead, start, end in sorted(found)
    ]


def resolve_reference(session, analysis_id, question):
    number = referenced_turn(question)
    if number is None:
        return {"status": "not_requested"}
    if session.get("analysis_id") != analysis_id:
        raise WindowReferenceError("MEMORY_ANALYSIS_MISMATCH")
    if number == -1:
        return {
            "status": "needs_clarification",
            "message": "请明确要查询第几轮的哪个窗口。",
        }

    turns = session.get("turns", [])
    # Validate all records, including old records outside the recent prose budget.
    for turn in turns:
        output = turn.get("output", {})
        if (turn.get("analysis_id") != analysis_id
                or output.get("analysis_id") != analysis_id):
            raise WindowReferenceError("MEMORY_ANALYSIS_MISMATCH")

    candidates = {}
    if number <= len(turns):
        output = turns[number - 1]["output"]
        evidence = output.get("evidence") or {}
        for event in output.get("trace") or []:
            if (event.get("stage") != "tool"
                    or event.get("tool") not in WINDOW_TOOLS
                    or event.get("ok") is not True):
                continue
            evidence_id = event.get("evidence_id")
            response = evidence.get(evidence_id)
            if not isinstance(response, dict):
                continue
            if (response.get("analysis_id") != analysis_id
                    or response.get("evidence_id") != evidence_id):
                raise WindowReferenceError("MEMORY_EVIDENCE_MISMATCH")
            if response.get("ok") is not True:
                continue
            for window in windows_from_data(
                    response.get("data"), event.get("arguments")):
                key = (window["lead"], window["start_sample"],
                       window["end_sample"])
                candidates.setdefault(key, {
                    **window, "source_evidence_ids": [],
                })["source_evidence_ids"].append(evidence_id)

    items = list(candidates.values())
    if len(items) != 1:
        labels = "、".join(
            f'{item["lead"]} [{item["start_sample"]},{item["end_sample"]})'
            for item in items
        )
        message = (
            f"第{number}轮记录了多个窗口：{labels}。请指定导联和窗口。"
            if items else
            f"无法从第{number}轮的工具证据确认窗口，请提供导联和起止采样点。"
        )
        return {
            "status": "needs_clarification",
            "turn_number": number,
            "message": message,
            "candidates": items,
        }
    return {
        "status": "resolved",
        "analysis_id": analysis_id,
        "turn_number": number,
        "window": items[0],
    }


def prepare_reference(gateway, analysis_id, question):
    number = referenced_turn(question)
    if number is None:
        return {"status": "not_requested"}
    session = getattr(gateway, "memory_session", None)
    if session is None:
        return {
            "status": "needs_clarification",
            "message": "当前没有可核对的会话工具历史，请提供导联和窗口坐标。",
        }
    return resolve_reference(session, analysis_id, question)


def query_arguments(reference):
    return {
        key: reference["window"][key]
        for key in ("lead", "start_sample", "end_sample")
    }


def check_query(reference, name, arguments):
    if reference.get("status") != "resolved" or name not in WINDOW_TOOLS:
        return
    # A recovered object uses exact sample coordinates, not a new relative window.
    expected = query_arguments(reference)
    if (name != "inspect_error_window"
            or set(arguments) != set(expected)
            or type(arguments.get("start_sample")) is not int
            or type(arguments.get("end_sample")) is not int
            or arguments != expected):
        raise WindowReferenceError("WINDOW_REFERENCE_QUERY_MISMATCH")


def check_response(reference, response):
    if (response.get("ok") is not True
            or response.get("analysis_id") != reference["analysis_id"]):
        raise WindowReferenceError("WINDOW_REFERENCE_REQUERY_FAILED")
    expected = query_arguments(reference)
    if windows_from_data(response.get("data"), expected) != [expected]:
        raise WindowReferenceError("WINDOW_REFERENCE_RESPONSE_MISMATCH")


def check_submission(reference, answer):
    if reference.get("status") != "resolved":
        return
    wanted = reference.get("current_evidence_id")
    # Require an actual structured citation, not an ID merely written in prose.
    structured = {key: value for key, value in answer.items() if key != "answer"}

    def contains(node):
        if isinstance(node, dict):
            return any(contains(value) for value in node.values())
        if isinstance(node, list):
            return any(contains(value) for value in node)
        return isinstance(node, str) and node == wanted

    if not wanted or not contains(structured):
        raise WindowReferenceError("WINDOW_REFERENCE_CITATION_REQUIRED")
