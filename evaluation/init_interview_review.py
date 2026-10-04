"""Create optional claim/call annotations, never overwrite existing human reviews."""
import argparse,hashlib,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',default='evaluation/interview_runs');p.add_argument('--batch');a=p.parse_args()
 for f in Path(a.runs).glob('*/result.json'):
  r=json.loads(f.read_text(encoding='utf-8'));o=r['output']
  if a.batch and r.get('config',{}).get('batch_id')!=a.batch:continue
  target=f.with_name('interview_review.json')
  if target.exists():continue
  v={'run_id':r['run_id'],'result_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'reviewer':'',
     'instructions':'将答案拆成可核对陈述，每条填写text、evidence_path、verdict(supported/unsupported/contradicted)。所有补查逐条分类并说明理由。不要只挑正确陈述。task_correct等三项仍填写原review.json。',
     'claims':[], 'calls':[{'trace_index':i,'tool':t.get('tool'),'arguments':t.get('arguments'),'classification':None,'reason':''} for i,t in enumerate(o.get('trace',[])) if t.get('stage')=='tool' and t.get('source')!='bootstrap']}
  target.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
