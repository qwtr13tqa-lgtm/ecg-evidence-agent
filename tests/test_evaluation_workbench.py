import copy
import unittest
from src.evaluation.workbench import paired, issues, timing, METRICS


def row():
    config={k:'same' for k in ('model','endpoint_sha256','data_sha256','suite_sha256','temperature','max_tokens','timeout_seconds','input_sha256')}
    return dict(case='case',scheme='agent',repetition=1,turn=1,run_id='run',timeout=False,outcome='pending',review={},latency=10,
        record=dict(config=config,case={'question':'q'},reference={'analysis_id':'a','window':{'start':1}},
            output={'status':'completed_draft','trace':[]},automatic={'metrics':{},'field_checks':{}}))


def fine():
    return dict(redundant_calls=0,pending_calls=0,failed_calls=0,unsupported_claims=0)


class WorkbenchTests(unittest.TestCase):
    def test_pending_is_not_failure(self):
        self.assertEqual([x['category'] for x in issues(row(),fine())],['任务待复核'])
    def test_failed_field_has_path(self):
        r=row();r['record']['automatic']['field_checks']={'peak_sample':False}
        self.assertTrue(any(x['detail']=='peak_sample' for x in issues(r,fine())))
    def test_redundant_and_pending_separate(self):
        f=fine();f.update(redundant_calls=2,pending_calls=1)
        labels={x['category'] for x in issues(row(),f)}
        self.assertTrue({'已确认冗余查询','调用待核查'}<=labels)
    def test_timeout_trace(self):
        r=row();r['timeout']=True;r['record']['output']['status']='failed'
        self.assertTrue({'超时','运行未完成'}<={x['category'] for x in issues(r,fine())})
    def test_code_change_allowed(self):
        a=row();b=copy.deepcopy(a);b['record']['config']['source_sha256']='changed'
        self.assertEqual(len(paired([a],[b])[0]),1)
    def test_model_input_reference_changes_rejected(self):
        for key in ('model','input_sha256','data_sha256','suite_sha256'):
            a=row();b=copy.deepcopy(a);b['record']['config'][key]='different'
            self.assertEqual(len(paired([a],[b])[1]),1)
        a=row();b=copy.deepcopy(a);b['record']['reference']['window']['start']=2
        self.assertEqual(len(paired([a],[b])[1]),1)
    def test_analysis_id_change_allowed(self):
        a=row();b=copy.deepcopy(a);b['record']['reference']['analysis_id']='b'
        self.assertEqual(len(paired([a],[b])[0]),1)
    def test_missing_config_rejected(self):
        a=row();b=copy.deepcopy(a);del b['record']['config']['model']
        self.assertEqual(len(paired([a],[b])[1]),1)
    def test_unmatched_count(self):
        a=row();b=copy.deepcopy(a);b['case']='other'
        self.assertEqual(paired([a],[b])[2:],(1,1))
    def test_timing_preserves_failed_waits(self):
        r=row();r['record']['automatic']['request_measurements']=[{'elapsed_seconds':8,'status':'failed'}]
        r['record']['output']['trace']=[{'stage':'tool','elapsed_seconds':1}]
        self.assertEqual(timing(r),{'总耗时':10,'模型请求耗时合计':8,'已记录工具耗时合计':1,'其余本地开销（非网络时间）':1})
    def test_every_metric_has_diagnosis(self):
        self.assertEqual(len(METRICS),10)
        for v in METRICS.values():self.assertTrue(len(v)==5 and all(v))

if __name__=='__main__':unittest.main()
