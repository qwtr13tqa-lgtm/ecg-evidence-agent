"""Checks copied scalar observations and reference membership, not prose semantics."""
import json
import re


class AnswerValidationError(ValueError):
    def __init__(self, code, index, path):
        super().__init__(code)
        self.code = code
        self.details = {"observation_index": index}
        if isinstance(path, str) and len(path) <= 160 and re.fullmatch(r"/[A-Za-z0-9_/~.-]*", path):
            self.details["requested_path"] = path


def scalar_paths(data, limit=200):
    """Bounded actual data-relative pointers, without duplicating values."""
    paths = []
    def walk(value, prefix):
        if len(paths) >= limit:
            return
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, prefix + "/" + str(key).replace("~", "~0").replace("/", "~1"))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, prefix + "/" + str(index))
        elif prefix:
            paths.append(prefix)
    walk(data, "")
    return paths


def validate_answer(answer, evidence, knowledge):
    json.dumps(answer, allow_nan=False)
    if not isinstance(answer, dict) or set(answer) != {"answer", "evidence_ids", "knowledge_ids", "observations"}:
        raise ValueError("answer schema")
    if not isinstance(answer["answer"], str) or not answer["answer"].strip():
        raise ValueError("empty answer")
    for field, pool in (("evidence_ids", evidence), ("knowledge_ids", knowledge)):
        ids = answer[field]
        if (not isinstance(ids, list) or any(not isinstance(x, str) or x not in pool for x in ids)
                or len(ids) != len(set(ids))):
            raise ValueError("invalid references")
    observations = answer["observations"]
    if not isinstance(observations, list) or not observations:
        raise ValueError("missing observations")
    seen = set()
    for index, obs in enumerate(observations):
        if not isinstance(obs, dict) or set(obs) != {"evidence_id", "path", "value"}:
            raise ValueError("observation schema")
        eid, path = obs["evidence_id"], obs["path"]
        def reject(code):
            raise AnswerValidationError(code, index, path)
        if not isinstance(eid, str) or eid not in answer["evidence_ids"]:
            raise ValueError("uncited observation")
        if (not isinstance(path, str) or not path.startswith("/")
                or re.search(r"~(?![01])", path) or (eid, path) in seen):
            reject("OBSERVATION_POINTER_INVALID")
        seen.add((eid, path))
        value = evidence[eid]["data"]
        for token in path[1:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                if (not token.isascii() or not token.isdecimal() or len(token) > 10
                        or str(int(token)) != token or int(token) >= len(value)):
                    reject("OBSERVATION_INDEX_INVALID")
                value = value[int(token)]
            elif isinstance(value, dict):
                if token not in value:
                    reject("OBSERVATION_PATH_NOT_FOUND")
                value = value[token]
            else:
                reject("OBSERVATION_PATH_THROUGH_SCALAR")
        if isinstance(value, (dict, list)):
            reject("OBSERVATION_NON_SCALAR")
        if type(value) is not type(obs["value"]) or value != obs["value"]:
            reject("OBSERVATION_VALUE_MISMATCH")
    return {"passed": True, "scope": "structure_values_references_only",
            "semantic_support_checked": False, "medical_correctness_checked": False}
