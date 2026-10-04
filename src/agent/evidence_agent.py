"""Bounded native tool-calling graph. Each run has isolated state and executor."""
import hashlib
import json
import time
from copy import deepcopy
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from src.analysis.interpretation import build_interpretation
from src.tools.executor import ECGToolExecutor
from .tool_protocol import SYSTEM, SYNTHETIC_SYSTEM, TOOLS, strict_json
from .answer_validator import validate_answer, scalar_paths, AnswerValidationError
from .answer_parser import parse_answer, AnswerParseError
from .observation_repair import repair_non_scalar
from .window_memory import (
    WindowReferenceError, prepare_reference, query_arguments,
    check_query, check_response, check_submission,
)


class AgentState(TypedDict, total=False):
    messages: list
    pending: list
    evidence: dict
    knowledge: dict
    trace: list
    seen: list
    call_ids: list
    model_calls: int
    tool_calls: int
    route: str
    status: str
    error: str
    draft: dict
    validation: dict
    observation_repairs: int
    truncation_repairs: int
    finalization_only: bool


class ECGEvidenceAgent:
    def __init__(self, store, retriever, gateway, *, max_model_calls=4, max_tool_calls=6):
        for value in (max_model_calls, max_tool_calls):
            if type(value) is not int or not 1 <= value <= 20:
                raise ValueError("budgets must be integers 1..20")
        self.store, self.retriever, self.gateway = store, retriever, gateway
        self.max_model_calls, self.max_tool_calls = max_model_calls, max_tool_calls

    def run(self, analysis_id, question, *, allow_external=False, data_kind="real_ecg"):
        started = time.perf_counter()
        def failed(code):
            return {"status": "failed", "error": code, "route": "stop", "draft": {}, "validation": {}}
        if allow_external is not True:
            return {"analysis_id": analysis_id, **failed("EXTERNAL_TRANSFER_NOT_ENABLED")}
        if (not isinstance(question, str) or not question.strip() or len(question) > 4000
                or data_kind not in ("real_ecg", "synthetic_software_test")):
            return {"analysis_id": analysis_id, **failed("INVALID_INPUT")}
        try:
            analysis = self.store.get_result(analysis_id)
            policy = build_interpretation(analysis)
            if data_kind == "synthetic_software_test":
                policy["limitations"].append({"code": "SYNTHETIC_DATA",
                    "message": "本次全部数值为软件预设，未运行 SGRF-Net，不代表实际模型测量。"})
            executor = ECGToolExecutor(self.store, analysis_id)
            bootstrap = executor.execute("get_analysis_summary", {})
            if not bootstrap["ok"]:
                raise ValueError("summary unavailable")
            # Do not transmit local filenames/checkpoint paths or arbitrary provenance.
            bootstrap["data"].pop("provenance", None)
            bootstrap["observation_paths"] = scalar_paths(bootstrap["data"])
            initial = {"analysis_id": analysis_id, "data_kind": data_kind,
                       "interpretation": policy, "bootstrap": bootstrap}
            initial_text = json.dumps(initial, ensure_ascii=False, allow_nan=False)
            if len(initial_text) > 24000:
                raise ValueError("initial context too large")
        except Exception:
            return {"analysis_id": analysis_id, **failed("ANALYSIS_UNAVAILABLE")}

        try:
            reference = prepare_reference(self.gateway, analysis_id, question)
        except WindowReferenceError:
            return {"analysis_id": analysis_id,
                    **failed("MEMORY_ANALYSIS_OR_EVIDENCE_MISMATCH")}

        if reference["status"] == "needs_clarification":
            return {
                "analysis_id": analysis_id, "data_kind": data_kind,
                "status": "needs_clarification", "error": "",
                "draft": {"answer": reference["message"]},
                "validation": {"passed": False},
                "reference_resolution": reference,
                "evidence": {}, "knowledge": {}, "trace": [],
                "model_calls": 0, "tool_calls": 0,
                "requires_review": True,
                "elapsed_seconds": time.perf_counter() - started,
            }

        memory_evidence, memory_trace, memory_seen = {}, [], []
        if reference["status"] == "resolved":
            args = query_arguments(reference)
            tick = time.perf_counter()
            try:
                response = executor.execute("inspect_error_window", args)
                check_response(reference, response)
                response["observation_paths"] = scalar_paths(response["data"])
                payload = json.dumps(response, ensure_ascii=False, allow_nan=False)
                if len(payload) > 24000:
                    raise ValueError("memory response too large")
                reference["current_evidence_id"] = response["evidence_id"]
                memory_evidence[response["evidence_id"]] = deepcopy(response)
                memory_trace.append({
                    "stage": "tool", "tool": "inspect_error_window",
                    "arguments": args, "ok": True,
                    "evidence_id": response["evidence_id"], "error": None,
                    "source": "historical_window_resolution",
                    "source_turn": reference["turn_number"],
                    "elapsed_seconds": time.perf_counter() - tick,
                })
                # No historical measurement values or historical IDs enter the prompt.
                initial["resolved_window_reference"] = {
                    "analysis_id": analysis_id,
                    "turn_number": reference["turn_number"],
                    "window": args,
                    "current_evidence": response,
                    "instruction": (
                        "这是指定历史轮次的唯一工具查询窗口，已经重新查询。"
                        "请使用这份本轮证据回答并提供结构化引用。"
                        "不得替换为模型关注区域；无需再次查询同一窗口。"
                    ),
                }
                initial_text = json.dumps(initial, ensure_ascii=False, allow_nan=False)
                if len(initial_text) > 48000:
                    raise ValueError("memory context too large")
            except Exception:
                return {"analysis_id": analysis_id,
                        **failed("WINDOW_REFERENCE_REQUERY_FAILED")}

        # efficient-queries-1.0
        from .efficient_queries import complete as efficient_complete, policy_trace
        memory_trace.append(policy_trace(self.retriever))

        def decide(state):
            if state["model_calls"] >= self.max_model_calls:
                return failed("MODEL_BUDGET_EXHAUSTED")
            if len(json.dumps(state["messages"], ensure_ascii=False)) > 90000:
                return failed("CONTEXT_BUDGET_EXHAUSTED")
            count = state["model_calls"] + 1
            tick = time.perf_counter()
            phase = "gateway"
            from .truncation_repair import diagnostics, repair_messages
            reply = None
            try:
                available_tools = TOOLS
                if state.get("finalization_only"):
                    available_tools = ([t for t in TOOLS if t.get("function", {}).get("name") == "submit_answer"]
                        if state["tool_calls"] < self.max_tool_calls else [])
                reply = efficient_complete(self.gateway, state["messages"], available_tools, self.retriever)
                phase = "response_protocol"
                calls = reply.get("tool_calls", [])

                print("RESPONSE_PROTOCOL:", {
                    "finish_reason": reply.get("finish_reason"),
                    "tool_calls_type": type(calls).__name__,
                    "tool_call_count": len(calls) if isinstance(calls, list) else None,
                    "tools": [
                        {
                            "id": c.get("id"),
                            "type": c.get("type"),
                            "name": c.get("function", {}).get("name"),
                            "arguments_type": type(
                                c.get("function", {}).get("arguments")
                            ).__name__,
                        }
                        for c in calls
                        if isinstance(c, dict)
                           and isinstance(c.get("function"), dict)
                    ] if isinstance(calls, list) else [],
                }, flush=True)
                if reply.get("finish_reason") == "length":
                    trace = state["trace"] + [{"stage": "model", "call": count,
                        "elapsed_seconds": time.perf_counter() - tick,
                        "failure_phase": "response_protocol", "error_code": "MODEL_OUTPUT_TRUNCATED",
                        **diagnostics(reply)}]
                    if not state.get("truncation_repairs") and count < self.max_model_calls:
                        return {"route": "decide", "model_calls": count,
                            "truncation_repairs": 1, "finalization_only": True,
                            "messages": repair_messages(state["messages"], state["tool_calls"] >= self.max_tool_calls),
                            "trace": trace + [{"stage": "truncation_repair", "repair_attempt": 1,
                                "finalization_only": True, "within_original_budget": True}]}
                    return {**failed("MODEL_OUTPUT_TRUNCATED"), "model_calls": count,
                        "trace": trace + [{"stage": "truncation_repair_skipped",
                            "reason": "already_attempted" if state.get("truncation_repairs") else "model_budget_exhausted"}]}
                if state.get("finalization_only") and isinstance(calls, list) and any(
                        not isinstance(c, dict) or not isinstance(c.get("function"), dict)
                        or c["function"].get("name") != "submit_answer" for c in calls):
                    return {**failed("FINALIZATION_QUERY_REJECTED"), "model_calls": count,
                        "trace": state["trace"] + [{"stage": "tool_rejected", "call": count,
                            "error_code": "FINALIZATION_QUERY_REJECTED", **diagnostics(reply)}]}
                if not isinstance(calls, list) or reply.get("finish_reason") not in ("stop", "tool_calls"):
                    raise ValueError("invalid or truncated reply")
                trace = state["trace"] + [{"stage": "model", "call": count,
                    "elapsed_seconds": time.perf_counter() - tick, "tool_call_count": len(calls)}]
                if calls:
                    if state["tool_calls"] + len(calls) > self.max_tool_calls:
                        return {**failed("TOOL_BUDGET_EXHAUSTED"), "model_calls": count, "trace": trace}
                    ids = list(state["call_ids"])
                    for call in calls:
                        if (not isinstance(call, dict) or call.get("type") != "function"
                                or not isinstance(call.get("id"), str) or not call["id"] or call["id"] in ids
                                or not isinstance(call.get("function"), dict)
                                or not isinstance(call["function"].get("name"), str)
                                or not isinstance(call["function"].get("arguments"), str)):
                            raise ValueError("invalid tool call")
                        ids.append(call["id"])
                    if any(c["function"]["name"] == "submit_answer" for c in calls):
                        if len(calls) != 1:
                            return {**failed("SUBMIT_MUST_BE_ALONE"), "model_calls": count, "trace": trace}
                        phase = "answer_json"
                        answer = strict_json(calls[0]["function"]["arguments"])
                        phase = "answer_validation"
                        check_submission(reference, answer)
                        validation = validate_answer(answer, state["evidence"], state["knowledge"])
                        trace.append({"stage": "submission", "tool": "submit_answer", "ok": True})
                        return {"draft": answer, "validation": validation, "route": "stop",
                                "status": "completed_draft", "model_calls": count,
                                "tool_calls": state["tool_calls"] + 1, "trace": trace}
                    message = {"role": "assistant", "content": None, "tool_calls": calls}
                    return {"pending": calls, "call_ids": ids, "route": "tools", "model_calls": count,
                            "trace": trace, "messages": state["messages"] + [message]}
                phase = "answer_json"
                answer = parse_answer(reply.get("content"))
                phase = "answer_validation"
                check_submission(reference, answer)
                validation = validate_answer(answer, state["evidence"], state["knowledge"])
                return {"draft": answer, "validation": validation, "route": "stop",
                        "status": "completed_draft", "model_calls": count, "trace": trace}
            except Exception as exc:
                if (not state.get("finalization_only") and isinstance(exc, AnswerValidationError)
                        and exc.code == "OBSERVATION_NON_SCALAR"):
                    correction = repair_non_scalar(
                        state, calls, answer, exc.details, count,
                        self.max_model_calls, self.max_tool_calls)
                    if correction is not None:
                        return {**correction, "trace": trace + [{
                            "stage": "submission_repair", "ok": False,
                            "error_code": exc.code, **exc.details,
                            "repair_attempt": 1}]}
                    return {**failed(exc.code), "model_calls": count,
                            "tool_calls": state["tool_calls"] + (1 if calls else 0),
                            "trace": trace + [{"stage": "submission", "ok": False,
                                "error_code": exc.code, **exc.details}]}
                # Do not expose raw provider messages, prompts, or credentials.
                code = (str(exc) if isinstance(exc, WindowReferenceError)
                        else exc.code) if isinstance(
                            exc, (WindowReferenceError, AnswerValidationError)) else {
                    "gateway": "GATEWAY_REQUEST_FAILED", "response_protocol": "MODEL_PROTOCOL_INVALID",
                    "answer_json": "ANSWER_JSON_INVALID", "answer_validation": "ANSWER_VALIDATION_FAILED"}[phase]
                details = exc.details if isinstance(exc, (AnswerValidationError, AnswerParseError)) else {}
                return {**failed(code), "model_calls": count,
                        "trace": state["trace"] + [{"stage": "model", "call": count,
                            "elapsed_seconds": time.perf_counter() - tick, "error_type": type(exc).__name__,
                            "failure_phase": phase, "error_code": code, **diagnostics(reply), **details}]}

        def execute(state):
            messages, trace = list(state["messages"]), list(state["trace"])
            evidence, knowledge = deepcopy(state["evidence"]), deepcopy(state["knowledge"])
            seen, count = list(state["seen"]), state["tool_calls"]
            for call in state["pending"]:
                count += 1
                name = call["function"]["name"]
                args = None
                tick = time.perf_counter()
                try:
                    args = strict_json(call["function"]["arguments"])
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be object")
                    check_query(reference, name, args)
                    if name == "inspect_rr_intervals":
                        args = {"offset": 0, "limit": 50, **args}
                    if name == "search_knowledge":
                        args = {"top_k": 3, **args}
                    identity = json.dumps([name, args], sort_keys=True, allow_nan=False)
                    if identity in seen:
                        trace.append({"stage": "tool", "tool": name, "arguments": args,
                            "ok": False, "error": "REPEATED_TOOL_CALL",
                            "elapsed_seconds": time.perf_counter() - tick})
                        return {**failed("REPEATED_TOOL_CALL"), "tool_calls": count, "trace": trace}
                    seen.append(identity)
                    if name == "search_knowledge":
                        if (set(args) != {"query", "top_k"} or not isinstance(args["query"], str)
                                or not 1 <= len(args["query"].strip()) <= 1000
                                or type(args["top_k"]) is not int or not 1 <= args["top_k"] <= 3):
                            raise ValueError("knowledge arguments")
                        docs = [hit.to_dict() for hit in self.retriever.search(**args)]
                        response = {"analysis_id": analysis_id, "ok": True,
                            "evidence_id": analysis_id + ":knowledge:" + hashlib.sha256(identity.encode()).hexdigest()[:16],
                            "data": {"documents": docs, "status": "candidates_returned" if docs else "no_match"}}
                    else:
                        response = executor.execute(name, args)
                        if name == "get_analysis_summary" and response.get("ok"):
                            response["data"].pop("provenance", None)
                    if response.get("ok"):
                        response["observation_paths"] = scalar_paths(response["data"])
                    payload = json.dumps(response, ensure_ascii=False, allow_nan=False)
                    if len(payload) > 24000:
                        raise ValueError("tool response too large")
                    if response.get("ok"):
                        evidence[response["evidence_id"]] = deepcopy(response)
                        if name == "search_knowledge":
                            knowledge.update({doc["id"]: doc for doc in docs})
                except WindowReferenceError as exc:
                    trace.append({"stage": "tool", "tool": name,
                        "arguments": args, "ok": False, "error": str(exc)})
                    return {**failed(str(exc)), "tool_calls": count, "trace": trace}
                except Exception:
                    response = {"analysis_id": analysis_id, "ok": False, "error": "TOOL_ARGUMENTS_OR_EXECUTION_FAILED"}
                    payload = json.dumps(response)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": payload})
                trace.append({"stage": "tool", "tool": name, "arguments": args,
                    "ok": response.get("ok", False), "evidence_id": response.get("evidence_id"),
                    "error": response.get("error"), "elapsed_seconds": time.perf_counter() - tick})
            return {"messages": messages, "trace": trace, "evidence": evidence, "knowledge": knowledge,
                    "seen": seen, "tool_calls": count, "route": "decide"}

        graph = StateGraph(AgentState)
        graph.add_node("decide", decide)
        graph.add_node("tools", execute)
        graph.add_edge(START, "decide")
        graph.add_conditional_edges("decide", lambda s: s["route"], {"tools": "tools", "decide": "decide", "stop": END})
        graph.add_conditional_edges("tools", lambda s: s["route"], {"decide": "decide", "stop": END})
        state = graph.compile().invoke({"messages": [
            {"role": "system", "content": SYSTEM + "\nobservations 仅引用工具 observation_paths 中具体标量字段。不得引用 temporal_regions 等列表本身；逐项引用子字段。结构化 value 保留工具原值和类型，正文可合理四舍五入。" + (SYNTHETIC_SYSTEM if data_kind == "synthetic_software_test" else "")},
            {"role": "user", "content": initial_text},
            {"role": "user", "content": question}],
            "evidence": {bootstrap["evidence_id"]: bootstrap, **memory_evidence}, "knowledge": {},
            "trace": memory_trace, "seen": memory_seen, "call_ids": [], "pending": [],
            "model_calls": 0, "tool_calls": sum(t.get("stage") == "tool" for t in memory_trace), "status": "running", "error": "",
            "draft": {}, "validation": {}}, config={"recursion_limit": 2 * self.max_model_calls + 5})
        return {"analysis_id": analysis_id, "data_kind": data_kind,
                "reference_resolution": reference,
                "status": state["status"], "error": state["error"],
                "observation_repairs": state.get("observation_repairs", 0),
                "truncation_repairs": state.get("truncation_repairs", 0),
                "draft": state["draft"], "validation": state["validation"],
                "limitations": policy["limitations"], "requires_review": True,
                "evidence": state["evidence"], "knowledge": state["knowledge"],
                "trace": state["trace"], "model_calls": state["model_calls"],
                "tool_calls": state["tool_calls"], "elapsed_seconds": time.perf_counter() - started}
