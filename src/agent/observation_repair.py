"""One bounded correction for non-scalar observation pointers."""
import json
from copy import deepcopy


def repair_non_scalar(state, calls, answer, details, count, max_models, max_tools):
    spent = state["tool_calls"] + (1 if calls else 0)
    if state.get("observation_repairs", 0) or count >= max_models or spent >= max_tools:
        return None
    index = details.get("observation_index")
    observations = answer.get("observations", [])
    if type(index) is not int or not 0 <= index < len(observations):
        return None
    obs = observations[index]
    eid, prefix = obs.get("evidence_id"), obs.get("path")
    if not isinstance(prefix, str):
        return None
    response = state["evidence"].get(eid, {})
    paths = [p for p in response.get("observation_paths", [])
             if isinstance(p, str) and p.startswith(prefix + "/")][:40]
    feedback = {
        "error": "OBSERVATION_NON_SCALAR", "observation_index": index,
        "evidence_id": eid, "requested_path": prefix,
        "candidate_scalar_paths": paths,
        "instruction": "该路径指向列表或对象，不能作为单条 observation。请根据已有工具证据分别引用具体标量字段，复制原始值并保持数值类型。空列表不要虚构子字段。重新提交完整回答；不要改变查询对象。仅允许此次格式纠正。",
    }
    messages = deepcopy(state["messages"])
    if calls:
        messages.append({"role": "assistant", "content": None, "tool_calls": deepcopy(calls)})
        messages.append({"role": "tool", "tool_call_id": calls[0]["id"],
                         "content": json.dumps(feedback, ensure_ascii=False)})
    else:
        messages.append({"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)})
        messages.append({"role": "user", "content": json.dumps(feedback, ensure_ascii=False)})
    return {"messages": messages, "observation_repairs": 1,
            "tool_calls": spent, "model_calls": count,
            "call_ids": list(state["call_ids"]) + [c["id"] for c in calls],
            "route": "decide", "status": "running", "error": "",
            "draft": {}, "validation": {}}
