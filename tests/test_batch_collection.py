import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from src.review.collection_ops import base_rows,execute,evidence,needs_history
from src.review.collection_runtime import run_collection,checked_plan,deterministic_answer
from src.review.cross_record_agent import run,query_record
from src.review.cross_context import model_evidence
from src.agent.answer_validator import validate_answer


def report():
    return {'threshold':0,'checkpoint_sha256':'ck','adapter_sha256':'ad','rows':[
        {'index':i,'score':s,'label':l,'prediction':int(s>=0),'input_sha256':str(i).zfill(64)}
        for i,s,l in [(5,-.8,1),(6,-.9,1),(7,-.9,1),(8,.3,1),(9,.4,1),(10,.5,0),(11,-.1,0)]]}

def select(**kw):
    op={'kind':'select','filters':[{'field':'category','op':'eq','value':'fn'}],
        'order_by':'score','direction':'asc','limit':5,'offset':0,'columns':['score','label','prediction'],
        'group_by':None,'metrics':[]};op.update(kw);return op

def aggregate(**kw):
    op=select(kind='aggregate',filters=[{'field':'category','op':'in','value':['fn','tp']}],order_by=None,columns=[],group_by='category',metrics=['reconstruction_error','region_count']);op.update(kw);return op

def call(name,args,cid='call'):
    return {'finish_reason':'tool_calls','tool_calls':[{'id':cid,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}

def plan(mode='query',operations=None,clarification='',assumptions=None):
    return call('plan_batch_task',{'mode':mode,'operations':operations if operations is not None else [select()], 'clarification':clarification,'assumptions':assumptions or []})

class Gateway:
    def __init__(self,*replies):self.replies=list(replies);self.calls=[]
    def complete(self,messages,tools):
        self.calls.append((copy.deepcopy(messages),copy.deepcopy(tools)))
        r=self.replies.pop(0)
        if isinstance(r,Exception):raise r
        return r(messages,tools) if callable(r) else r

class History:
    def __init__(self,wrong=False):self.wrong=wrong;self.loads=[]
    def list_analyses(self,limit=100,offset=0):return [{'sample_index':i,'analysis_id':'a'+str(i)} for i in (5,6,8,9)] if offset==0 else []
    def load_analysis(self,aid):
        self.loads.append(aid);i=int(aid[1:]);r=next(r for r in report()['rows'] if r['index']==i)
        obj=SimpleNamespace(analysis_id=aid,model=SimpleNamespace(anomaly_score=r['score']),provenance={
            'input_sha256':r['input_sha256'],'checkpoint_sha256':'wrong' if self.wrong else 'ck','review_adapter_sha256':'ad',
            'sample_index':i,'crop_start_sample':100,'model_decision':{'status':'configured','threshold':0,'prediction':'model_anomaly' if r['prediction'] else 'model_normal'}})
        obj.to_llm_context=lambda:{'model':{'reconstruction_error':float(i),'shape_error':i/10},'evidence':{'temporal_regions':[] if i==5 else [{'start':100,'end':200}], 'lead_evidence':[{'lead':'V1' if i%2 else 'V2','rank':1}]},'signal_features':{'rhythm':{'rr_cv':.03,'measurement_status':'unvalidated'}}}
        return obj,None,None

def submit_from_evidence(messages,tools):
    pool={}
    for m in messages:
        if m['role'] not in ('tool','user'):continue
        try:e=json.loads(m['content'])
        except (ValueError,TypeError):continue
        if isinstance(e,dict) and e.get('evidence_id'):pool[e['evidence_id']]=e
    root=next(e for e in pool.values() if e.get('scope')=='explicit_batch_report')
    col=next(e for e in pool.values() if e.get('scope')=='batch_collection')
    return call('submit_answer',{'answer':'按实际有效样本数比较，缺失值未计为0；仅为描述性结果。','evidence_ids':[root['evidence_id'],col['evidence_id']],'knowledge_ids':[],
        'observations':[{'evidence_id':root['evidence_id'],'path':'/record_count','value':root['data']['record_count']}, {'evidence_id':col['evidence_id'],'path':'/matched_n','value':col['data']['matched_n']}]},'submit')

class CollectionTests(unittest.TestCase):
    def test_sort_filter_then_topk_with_ties(self):
        d=execute(base_rows(report()),select(limit=2));self.assertEqual([r['index'] for r in d['rows']],[6,7]);self.assertEqual(d['matched_n'],3);self.assertTrue(d['has_more'])
    def test_pagination(self):self.assertEqual(execute(base_rows(report()),select(offset=2))['rows'][0]['index'],5)
    def test_all_rows(self):self.assertEqual(execute(base_rows(report()),select(filters=[],limit=500))['returned_n'],7)
    def test_empty(self):self.assertEqual(execute(base_rows(report()),select(filters=[{'field':'score','op':'gt','value':50}]))['returned_n'],0)
    def test_missing_not_zero_or_negative_match(self):
        rows=base_rows(report());self.assertEqual(execute(rows,select(filters=[{'field':'region_count','op':'ne','value':0}]))['matched_n'],0)
        self.assertEqual(execute(rows,select(filters=[{'field':'region_count','op':'is_missing','value':None}]))['matched_n'],7)
    def test_aggregate_validity_quantiles(self):
        rows=base_rows(report());rows[0]['reconstruction_error']=2;rows[1]['reconstruction_error']=4
        g=execute(rows,aggregate())['groups'][0];self.assertEqual(g['group_value'],'fn');m=g['metrics']['reconstruction_error']
        self.assertEqual((m['valid_n'],m['missing_n'],m['median'],m['q1'],m['q3']),(2,1,3,2.5,3.5))
    def test_categorical_missing_group(self):
        d=execute(base_rows(report()),aggregate(group_by='top_lead',metrics=[]));self.assertEqual(d['groups'][0]['group_value'],None);self.assertEqual(d['groups'][0]['count'],5)
    def test_no_mutation(self):
        rows=base_rows(report());old=copy.deepcopy(rows);execute(rows,select());self.assertEqual(rows,old)
    def test_reject_unsupported_and_nonfinite(self):
        for op in [select(order_by='probability'),select(limit=0),select(limit=True),select(filters=[{'field':'score','op':'gt','value':float('nan')}]),select(filters=[{'field':'category','op':'eq','value':'all'}]),select(group_by='category')]:
            with self.subTest(op=op),self.assertRaises(ValueError):execute(base_rows(report()),op)
    def test_history_needed_only_for_feature_fields(self):self.assertFalse(needs_history(select()));self.assertTrue(needs_history(aggregate()))
    def test_program_generated_citations(self):
        e=evidence('aid','batch',execute(base_rows(report()),select()));o={'evidence':{e['evidence_id']:e}}
        deterministic_answer(o,'本地结果');self.assertTrue(validate_answer(o['draft'],o['evidence'],{})['passed'])
    def test_table_bound_to_report(self):
        with self.assertRaisesRegex(ValueError,'SAMPLE_OUTSIDE'):
            query_record(None,report(),{r['index']:r for r in base_rows(report())},'aid','get_analysis_summary',{'sample_index':999})
    def test_projection_preserves_pointers_and_full_result(self):
        rows=[{**base_rows(report())[0],'index':i} for i in range(100)]
        e=evidence('aid','batch',execute(rows,select(limit=100)));projected=model_evidence(e)
        self.assertEqual(len(projected['data']['rows']),40);self.assertEqual(len(e['data']['rows']),100)
        self.assertIn('transport_notice',projected);self.assertEqual(projected['data']['rows'][20],e['data']['rows'][20])

class RuntimeTests(unittest.TestCase):
    def test_topk_one_model_zero_record_queries(self):
        gw=Gateway(plan(operations=[select(limit=2)]));out=run(History(),gw,'aid','找出所有漏报中异常分数最低的两条',report(),'batch')
        self.assertEqual(out['status'],'completed_draft',out);self.assertEqual(len(gw.calls),1);self.assertEqual(out['tool_calls'],0)
        e=next(iter(out['evidence'].values()));self.assertEqual([r['index'] for r in e['data']['rows']],[6,7]);self.assertTrue(out['validation']['passed'])
    def test_clarification_does_not_scan_records(self):
        h=History();gw=Gateway(plan('clarify',[], '“最值得关注”按最低分数还是局部误差排序？'))
        out=run(h,gw,'aid','找出最值得关注的几条',report(),'batch');self.assertEqual(h.loads,[])
        self.assertEqual(out['status'],'completed_draft');self.assertTrue(any(t.get('stage')=='clarification' and t['task_completed'] is False for t in out['trace']))
    def test_default_five_is_visible(self):
        out=run(None,Gateway(plan(assumptions=['未指定数量，默认前5条'])),'aid','最低几条',report(),'batch');self.assertIn('默认前5条',out['draft']['answer'])
    def test_whole_plan_checked_before_execution(self):
        h=History();out=run(h,Gateway(plan(operations=[select(),select(order_by='unknown')])),'aid','问题',report(),'batch')
        self.assertEqual(out['status'],'failed');self.assertEqual(out['evidence'],{});self.assertEqual(h.loads,[])
    def test_group_stats_actual_matched_history(self):
        out=run(History(),Gateway(plan(operations=[aggregate()])),'aid','统计FN TP',report(),'batch')
        d=next(iter(out['evidence'].values()))['data'];fn=next(g for g in d['groups'] if g['group_value']=='fn')
        self.assertEqual(fn['metrics']['region_count']['valid_n'],2);self.assertEqual(fn['metrics']['region_count']['missing_n'],1)
    def test_incompatible_history_excluded(self):
        out=run(History(wrong=True),Gateway(plan(operations=[aggregate()])),'aid','统计',report(),'batch')
        groups=next(iter(out['evidence'].values()))['data']['groups'];self.assertTrue(all(g['metrics']['region_count']['valid_n']==0 for g in groups))
    def test_analysis_uses_seed_and_preserves_budget(self):
        gw=Gateway(plan('analyze',[aggregate()]),submit_from_evidence)
        out=run(History(),gw,'aid','比較共性与反例',report(),'batch',max_model_calls=3)
        self.assertEqual(out['status'],'completed_draft',out);self.assertEqual(out['model_calls'],2)
        self.assertIn('query_batch_collection',[t['function']['name'] for t in gw.calls[1][1]])
    def test_explanation_timeout_keeps_local_results(self):
        gw=Gateway(plan('analyze',[aggregate()]),TimeoutError())
        out=run(History(),gw,'aid','比較共性',report(),'batch');self.assertEqual(out['error'],'GATEWAY_REQUEST_FAILED')
        self.assertTrue(any(e['scope']=='batch_collection' for e in out['evidence'].values()));self.assertEqual(out['model_calls'],2)
    def test_plan_timeout_has_no_results_claim(self):
        out=run(None,Gateway(TimeoutError()),'aid','筛选',report(),'batch');self.assertEqual(out['status'],'failed');self.assertEqual(out['evidence'],{})
    def test_snapshot_reuses_collection_without_replanning(self):
        out=run(History(),Gateway(plan('analyze',[aggregate()]),TimeoutError()),'aid','比較共性',report(),'batch')
        gw=Gateway(submit_from_evidence);retried=run(None,gw,'aid','比較共性',None,'batch',snapshot=out,max_model_calls=1)
        self.assertEqual(retried['status'],'completed_draft',retried);self.assertEqual(len(gw.calls),1)
    def test_cross_batch_snapshot_rejected(self):
        out=run(History(),Gateway(plan('analyze',[aggregate()]),TimeoutError()),'aid','比较',report(),'batch')
        r=run(None,Gateway(),'aid','比较',None,'other',snapshot=out,max_model_calls=1);self.assertEqual(r['error'],'SNAPSHOT_BATCH_MISMATCH')
    def test_snapshot_persists_under_existing_history_whitelist(self):
        out=run(History(),Gateway(plan(operations=[select()])),'aid','最低几条',report(),'batch')
        stored={k:out[k] for k in ('analysis_id','status','draft','validation','evidence','trace','model_calls','tool_calls','elapsed_seconds')}
        stored=json.loads(json.dumps(stored));self.assertTrue(any(e['scope']=='batch_collection' for e in stored['evidence'].values()))
    def test_collection_tool_after_plan(self):
        gw=Gateway(plan('analyze',[aggregate()]),call('query_batch_collection',select(limit=1),'select'),submit_from_evidence)
        out=run(History(),gw,'aid','比较并找代表',report(),'batch')
        self.assertEqual(out['status'],'completed_draft',out);self.assertEqual(out['tool_calls'],1)
        self.assertTrue(any(t.get('tool')=='query_batch_collection' and t.get('stage')=='tool' and t.get('ok') for t in out['trace']))
    def test_duplicate_seed_query_not_recomputed(self):
        gw=Gateway(plan('analyze',[aggregate()]),call('query_batch_collection',aggregate(),'repeat'),submit_from_evidence)
        out=run(History(),gw,'aid','比较',report(),'batch');self.assertEqual(out['status'],'completed_draft',out)
        self.assertTrue(any(t.get('error')=='REPEATED_TOOL_CALL' for t in out['trace']))


class RenderTests(unittest.TestCase):
    def test_failed_explanation_still_renders_saved_table(self):
        from src.ui.collection_results import render
        class UI:
            def __init__(self):self.frames=[];self.warnings=[]
            def write(self,*a,**k):pass
            def caption(self,*a,**k):pass
            def info(self,*a,**k):pass
            def json(self,*a,**k):pass
            def warning(self,text):self.warnings.append(text)
            def dataframe(self,data,**k):self.frames.append(data)
            def expander(self,*a,**k):return self
            def __enter__(self):return self
            def __exit__(self,*a):pass
        out=run(History(),Gateway(plan('analyze',[aggregate()]),TimeoutError()),'aid','比较',report(),'batch')
        out=json.loads(json.dumps(out));ui=UI();render(ui,out,'k')
        self.assertTrue(ui.frames);self.assertTrue(any('未完成' in w for w in ui.warnings))
    def test_scope_metadata_retained_on_plan_failure(self):
        out=run(None,Gateway(TimeoutError()),'aid','问题',report(),'batch')
        self.assertEqual(out['trace'][0]['stage'],'batch_scope')

if __name__=='__main__':unittest.main()
