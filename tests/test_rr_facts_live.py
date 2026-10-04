"""Local arithmetic test, no external request."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.analysis.pipeline import ECGAnalysisPipeline
from src.analysis.rr_facts import build_rr_facts, render_rr_facts

def main():
    p=argparse.ArgumentParser(); p.add_argument('--sample',type=int,default=0); a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    data=np.load(root/'data/Processed_PTBXL/test.npy',mmap_mode='r')
    if not 0<=a.sample<len(data): p.error('sample out of range')
    result=ECGAnalysisPipeline().analyze(data[a.sample,100:4900,:],source_id='test.npy',sample_index=a.sample,crop_start_sample=100)
    facts=build_rr_facts(result)
    print(json.dumps(facts,ensure_ascii=False,indent=2,allow_nan=False)); print(render_rr_facts(facts))
    if facts['status']!='verified': raise SystemExit('RR FACTS: FAIL')
    print('RR FACTS: PASS (stored arithmetic only; no external request)')
if __name__=='__main__': main()
