import unittest
from copy import deepcopy
from src.ui.answer_waveform import targets


def fixture():
    return dict(analysis_id='a',status='completed_draft',validation={'passed':True},
        draft={'evidence_ids':['a:e'],'answer':'错误的文字 V1 [4730,4800)'},
        trace=[dict(stage='tool',tool='inspect_recent_error',ok=True,evidence_id='a:e')],
        evidence={'a:e':dict(analysis_id='a',ok=True,data=dict(start_sample=4200,end_sample=4800,leads=[{'lead':'V2'}]))})


class WindowTests(unittest.TestCase):
    def test_exact_object_not_prose(self):
        w=targets(fixture(),'a',4800)[0]
        self.assertEqual((w['lead'],w['start_sample'],w['end_sample']),('V2',4200,4800))
    def test_cross_analysis(self):self.assertEqual(targets(fixture(),'b',4800),[])
    def test_cross_analysis_evidence(self):
        x=fixture();x['evidence']['a:e']['analysis_id']='b';self.assertEqual(targets(x,'a',4800),[])
    def test_unvalidated(self):
        x=fixture();x['validation']={};self.assertEqual(targets(x,'a',4800),[])
    def test_uncited(self):
        x=fixture();x['draft']['evidence_ids']=[];self.assertEqual(targets(x,'a',4800),[])
    def test_failed_tool(self):
        x=fixture();x['trace'][0]['ok']=False;self.assertEqual(targets(x,'a',4800),[])
    def test_invalid_boundary(self):self.assertEqual(targets(fixture(),'a',4799),[])
    def test_multiple_leads(self):
        x=fixture();x['evidence']['a:e']['data']['leads'].append({'lead':'V1'})
        self.assertEqual(len(targets(x,'a',4800)),2)
    def test_no_mutation(self):
        x=fixture();old=deepcopy(x);targets(x,'a',4800);self.assertEqual(x,old)
    def test_sixth_turn_old_reference_current_evidence(self):
        turns=[fixture()]+[{'status':'failed'}]*4+[fixture()]
        self.assertEqual(targets(turns[5],'a',4800)[0]['start_sample'],4200)

if __name__=='__main__':unittest.main()
