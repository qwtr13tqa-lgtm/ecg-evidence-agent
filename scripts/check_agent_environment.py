"""Read-only metadata/asset check. No gateway, inference, or environment secret reads."""
import json
import platform
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError
ROOT=Path(__file__).resolve().parents[1]

def main():
    expected={}
    for line in (ROOT/'requirements-agent-observed.txt').read_text().splitlines():
        if '==' in line and not line.lstrip().startswith('#'):
            k,v=line.split('==',1);expected[k]=v
    rows=[]
    for name,want in expected.items():
        try:actual=version(name)
        except PackageNotFoundError:actual=None
        rows.append({'package':name,'observed_version':want,'current_version':actual,'matches':actual==want})
    assets=['app_ecg.py','src/analysis/rr_facts.py','src/agent/evidence_agent.py',
            'src/reporting/readable_view.py','src/evaluation/checks_v2.py',
            'data/Processed_PTBXL/test.npy','data/knowledge/ecg_knowledge.jsonl',
            'ckpt_shape_guided_shapex_gate/best_TSRNet_SHAPEXGate-epoch23-auc0.8871.pt']
    report={'python':platform.python_version(),'expected_python':'3.10.12','packages':rows,
            'assets':{p:(ROOT/p).is_file() for p in assets},
            'scope':'Metadata and existence only; not dependency resolution, GPU, inference or gateway validation.'}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 1 if any(r['current_version'] is None for r in rows) or not all(report['assets'].values()) else 0
if __name__=='__main__':raise SystemExit(main())
