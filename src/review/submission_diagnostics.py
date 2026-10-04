"""Bounded diagnostics from authoritative evidence; never correct answers silently."""
import re
from copy import deepcopy

def bounded(value):
    if isinstance(value, (dict, list)):
        return {'type': type(value).__name__, 'size': len(value)}
    if isinstance(value, str):
        return value[:500]
    return value

def describe_failure(exc, answer, evidence):
    details = dict(getattr(exc, 'details', {}))
    result = {'error_code': exc.code, **details}
    index = details.get('observation_index')
    observations = answer.get('observations', []) if isinstance(answer, dict) else []
    if type(index) is not int or not 0 <= index < len(observations):
        return result
    obs = observations[index]
    eid, path = obs.get('evidence_id'), obs.get('path')
    result.update(evidence_id=eid, submitted_value=bounded(obs.get('value')),
                  submitted_type=type(obs.get('value')).__name__, path_exists=False)
    if eid not in evidence or not isinstance(path, str) or not path.startswith('/') or re.search(r'~(?![01])', path):
        return result
    value = evidence[eid]['data']
    try:
        for token in path[1:].split('/'):
            token = token.replace('~1', '/').replace('~0', '~')
            if isinstance(value, list):
                if not token.isascii() or not token.isdecimal() or len(token)>10 or str(int(token)) != token:
                    return result
                value = value[int(token)]
            elif isinstance(value, dict):
                value = value[token]
            else:
                return result
    except (KeyError, IndexError):
        return result
    result.update(path_exists=True, expected_value=bounded(value), expected_type=type(value).__name__)
    return deepcopy(result)
