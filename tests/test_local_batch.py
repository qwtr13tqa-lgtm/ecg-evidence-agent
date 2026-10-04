import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.review.local_batch import plan,advance,LOCK
from src.evaluation.records import atomic_json

REPORT={'threshold':.5,'rows':[{'index':i,'label':1,'prediction':int(i>0),'score':float(i),'input_sha256':str(i)*64} for i in range(3)]}
class History:
    def load_analysis(self,aid):return object(),None,{}

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.open=patch('src.review.local_batch.open_analysis',side_effect=lambda report,i,*a:('aid'+str(i),i==0)).start()
        self.match=patch('src.review.local_batch.compatible',return_value=True).start()
    def tearDown(self):patch.stopall();self.temp.cleanup()
    def go(self,limit=5,retry=False):return advance(REPORT,['fn','tp'],'data',self.root,History(),lambda:None,limit,retry)
    def test_resume_skips_completed(self):
        _,j=self.go(1);self.assertEqual(j['rows'][0]['status'],'reused')
        _,j=self.go();self.assertEqual(self.open.call_count,3);self.assertEqual(j['rows'][2]['status'],'created')
        self.go();self.assertEqual(self.open.call_count,3)
    def test_failure_continues_and_explicit_retry(self):
        self.open.side_effect=[ValueError('bad'),('b',False),('c',False)]
        _,j=self.go();self.assertEqual(j['rows'][0]['status'],'failed');self.assertEqual(j['rows'][2]['status'],'created')
        self.go();self.assertEqual(self.open.call_count,3)
        self.open.side_effect=None;self.open.return_value=('fixed',False)
        _,j=self.go(retry=True);self.assertEqual(j['rows'][0]['attempts'],2);self.assertEqual(len(j['rows'][0]['events']),2)
    def test_interrupted_resumed(self):
        p,j=plan(REPORT,['fn','tp'],self.root);j['rows'][0]['status']='running';atomic_json(p,j)
        _,j=self.go(1);self.assertEqual(j['rows'][0]['status'],'reused')
    def test_stale_completion_rechecked(self):
        self.go();self.match.return_value=False;self.go(1);self.assertEqual(self.open.call_count,4)
    def test_scope_jobs_separate(self):
        a,j=plan(REPORT,['fn'],self.root);b,k=plan(REPORT,['tp'],self.root)
        self.assertNotEqual(a,b);self.assertEqual([r['index'] for r in j['rows']],[0])
    def test_limit_and_selection(self):
        with self.assertRaises(ValueError):self.go(0)
        with self.assertRaises(ValueError):plan(REPORT,[],self.root)
    def test_lock_rejects_overlap(self):
        LOCK.acquire()
        try:
            with self.assertRaises(ValueError):self.go()
        finally:LOCK.release()
    def test_terminal_progress_persisted(self):
        def progress(j):
            if j['rows'][0]['status']=='reused':raise RuntimeError('UI interrupted')
        with self.assertRaises(RuntimeError):advance(REPORT,['fn','tp'],'data',self.root,History(),lambda:None,progress=progress)
        _,j=plan(REPORT,['fn','tp'],self.root);self.assertEqual(j['rows'][0]['status'],'reused');self.go();self.assertEqual(self.open.call_count,3)
if __name__=='__main__':unittest.main()
