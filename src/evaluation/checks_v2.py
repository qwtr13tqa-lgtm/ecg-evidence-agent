"""Versioned RR alias scoring; original evidence copies must remain exact."""
import math
from src.evaluation.checks import score_output as score_v1


def score_output(case, output, reference):
    scored = score_v1(case, output, reference)
    scored['version'] = 'development-checks-2.0'
    if case['expectation']['kind'] != 'rr' or reference.get('rr') is None:
        return scored
    evidence = output.get('evidence', {})
    draft = output.get('draft', {})
    cited = draft.get('evidence_ids', [])
    obs = draft.get('observations', [])
    aid = reference.get('analysis_id')
    rr = reference['rr']
    eligible = {t.get('evidence_id') for t in output.get('trace', [])
                if t.get('stage') == 'tool' and t.get('tool') == 'inspect_rr_intervals'
                and t.get('ok') is True}
    def exact_copy(eid, path, expected):
        return any(o.get('evidence_id') == eid and o.get('path') == path
                   and type(o.get('value')) is type(expected) and o.get('value') == expected
                   for o in obs) and eid in cited
    def numeric(a,b):
        return (type(a) in (int,float) and type(b) in (int,float)
                and math.isfinite(a) and math.isfinite(b)
                and math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-7))
    def accepted(key):
        for eid,item in evidence.items():
            data=item.get('data',{})
            if item.get('analysis_id') != aid or item.get('ok') is not True:
                continue
            old='/signal_features/rhythm/'+key
            if 'input' in data and exact_copy(eid,old,rr[key]):
                return True
            if eid not in eligible:
                continue
            facts=data.get('calculation_facts',{})
            if (facts.get('status') != 'verified' or facts.get('analysis_id') != aid
                    or facts.get('scope') != 'stored_rr_arithmetic_only'):
                continue
            field='candidate_peak_count' if key=='candidate_beat_count' else key
            actual=facts.get(field)
            agrees=(type(actual) is int and actual==rr[key]) if key=='candidate_beat_count' else (
                actual is None and rr[key] is None or numeric(actual,rr[key]))
            if agrees and exact_copy(eid,'/calculation_facts/'+field,actual):
                return True
        return False
    fields={k:accepted(k) for k in ('candidate_beat_count','mean_rr_seconds','heart_rate_bpm')}
    counts={k:any(eid in evidence and exact_copy(eid,'/'+k,rr[k]) for eid in eligible)
            for k in ('total_intervals','retained_count','excluded_count')}
    m=scored['metrics']
    m['task_observations']=bool(m['structure_values_references'] and m['analysis_binding']
                                and all(fields.values()) and all(counts.values()))
    scored['rr_field_checks']={**counts,**fields}
    scored['numeric_policy']='Exact copy from cited evidence; reference equivalence rtol=1e-6 atol=1e-7 for measurements only.'
    return scored
