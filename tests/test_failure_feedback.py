import copy
import unittest
from src.ui.failure_feedback import describe,render

def failed(code='GATEWAY_REQUEST_FAILED',local=True):
    out={'analysis_id':'a','status':'failed','error':code,'evidence':{},'trace':[{'stage':'model','error_type':'APITimeoutError'},{'stage':'failure','error_type':'ValueError','error':code}],'model_calls':2,'tool_calls':0}
    if local:
        for i,kind in enumerate(('aggregate','aggregate','select','select')):
            key='a:'+str(i);out['evidence'][key]={'analysis_id':'a','evidence_id':key,'ok':True,'scope':'batch_collection','data':{'query_status':'completed','operation':{'kind':kind}}}
        out['evidence']['a:overview']={'analysis_id':'a','evidence_id':'a:overview','ok':True,'scope':'explicit_batch_report','data':{}}
    return out
class FeedbackTests(unittest.TestCase):
    def test_timeout_preserves_completed_work(self):
        x=describe(failed(),True);self.assertIn('4 项',x['progress']);self.assertIn('2 项分组统计',x['progress']);self.assertIn('等待时间',x['reason']);self.assertIn('仅重试',x['next_step'])
    def test_wrapper_value_error_does_not_mask_timeout(self):self.assertEqual(describe(failed())['gateway_error_type'],'APITimeoutError')
    def test_plan_failure_is_not_blame_on_user(self):
        x=describe(failed('PLAN_CLARIFICATION_CONFLICT',False));self.assertIn('字段冲突',x['reason']);self.assertIn('不必反复',x['next_step']);self.assertNotIn('已保留',x['progress'])
    def test_unknown_not_invented_timeout(self):
        o=failed(local=False);o['trace']=[];x=describe(o);self.assertIn('未说明具体原因',x['reason'])
    def test_no_fake_success_or_mutation(self):
        o=failed();old=copy.deepcopy(o);x=describe(o);self.assertEqual(o,old);self.assertEqual(x['original_status'],'failed');self.assertFalse(x['automatic_retry'])
    def test_wrong_analysis_evidence_not_counted(self):
        o=failed();o['analysis_id']='other';self.assertNotIn('已保留 4',describe(o)['progress'])
    def test_no_retry_control_not_promised(self):self.assertNotIn('点击',describe(failed(),False)['next_step'])
    def test_validation_failure_explained(self):self.assertIn('校验',describe(failed('OBSERVATION_POINTER_INVALID'))['reason'])
    def test_no_raw_code_in_primary_ui(self):
        class UI:
            def __init__(self):self.primary=[];self.depth=0
            def warning(self,s):self.primary.append(s)
            def write(self,s):self.primary.append(s)
            def caption(self,s):self.primary.append(s)
            def expander(self,s):return self
            def __enter__(self):self.depth+=1
            def __exit__(self,*a):self.depth-=1
            def json(self,d):assert self.depth==1
        ui=UI();render(ui,failed(),True);self.assertNotIn('GATEWAY_REQUEST_FAILED',''.join(ui.primary));self.assertIn('未完成',''.join(ui.primary))
if __name__=='__main__':unittest.main()
