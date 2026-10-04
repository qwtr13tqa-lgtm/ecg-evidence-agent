import copy
import unittest
from src.reporting.readable_view import partition,evidence_rows,trace_rows,render_answer_details
class FakeUI:
    def __getattr__(self,k):return lambda *a,**kw:self
    def __enter__(self):return self
    def __exit__(self,*a):pass
class Tests(unittest.TestCase):
    def test_partition(self):
        o={'draft':{'evidence_ids':['b']},'evidence':{'a':{},'b':{}},'knowledge':{}}
        self.assertEqual(partition(o)[0],[('b',{})]);self.assertEqual(partition(o)[1],[('a',{})])
    def test_missing_measurement(self):
        r=evidence_rows({'input':{},'model':{},'signal_features':{}})
        self.assertEqual(r[-1]['数值'],'未提供')
    def test_zero_value(self):
        r=evidence_rows({'intervals':[],'excluded_count':0});self.assertEqual(r[0]['数值'],'0')
    def test_timeout(self):
        r=trace_rows({'trace':[{'stage':'model','error_type':'APITimeoutError','elapsed_seconds':180}]})
        self.assertEqual(r[0]['状态'],'失败')
    def test_no_mutation(self):
        o={'draft':{'evidence_ids':['A']},'evidence':{'A':{'data':{'intervals':[],'total_intervals':9}}},'validation':{'passed':True},'status':'completed_draft'}
        before=copy.deepcopy(o);render_answer_details(FakeUI(),o);self.assertEqual(o,before)
    def test_empty(self):render_answer_details(FakeUI(),{})
if __name__=='__main__':unittest.main()
