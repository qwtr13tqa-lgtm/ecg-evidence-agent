import unittest
from copy import deepcopy
from src.review.batch import query,validate,ask,signal
class Tests(unittest.TestCase):
 def setUp(self):
  self.r={'threshold':0,'rows':[{'index':0,'label':0,'prediction':1,'score':2,'input_sha256':'0'*64},{'index':1,'label':0,'prediction':1,'score':1,'input_sha256':'1'*64},{'index':2,'label':1,'prediction':0,'score':-1,'input_sha256':'2'*64}]}
 def test_fp(self):self.assertEqual([r['index'] for r in query(self.r)],[0,1])
 def test_fn(self):self.assertEqual(query(self.r,'fn')[0]['index'],2)
 def test_sort(self):self.assertEqual(query(self.r,'fp','asc')[0]['index'],1)
 def test_empty(self):self.assertEqual(query(self.r,'tn'),[])
 def test_no_mutation(self):
  old=deepcopy(self.r);query(self.r);self.assertEqual(old,self.r)
 def test_mismatch(self):
  self.r['rows'][0]['prediction']=0
  with self.assertRaises(ValueError):validate(self.r)
 def test_duplicate(self):
  self.r['rows'][1]['index']=0
  with self.assertRaises(ValueError):validate(self.r)
 def test_nan(self):
  self.r['rows'][0]['score']=float('nan')
  with self.assertRaises(ValueError):validate(self.r)
 def test_unallowed_query(self):
  with self.assertRaises(ValueError):query(self.r,'another_batch')
 def test_model_same_results(self):
  class Gateway:
   def complete(s,*args):return {'finish_reason':'tool_calls','tool_calls':[{'type':'function','function':{'name':'query_batch','arguments':'{"category":"fp","order":"desc"}'}}]}
  self.assertEqual(ask(self.r,'误报',Gateway())['rows'],query(self.r))
 def test_no_tool(self):
  class Gateway:
   def complete(s,*args):return {'tool_calls':[]}
  self.assertEqual(ask(self.r,'区域',Gateway())['status'],'unsupported')
 def test_extra_arguments(self):
  class Gateway:
   def complete(s,*args):return {'finish_reason':'tool_calls','tool_calls':[{'type':'function','function':{'name':'query_batch','arguments':'{"category":"fp","order":"desc","batch":"other"}'}}]}
  with self.assertRaises(ValueError):ask(self.r,'误报',Gateway())
 def test_wrong_file_hash(self):
  import tempfile
  from pathlib import Path
  with tempfile.TemporaryDirectory() as folder:
   path = Path(folder) / 'test.npy'
   path.write_bytes(b'test')
   with self.assertRaises(ValueError):
    signal(
     {'data_sha256': 'bad'},
     self.r['rows'][0],
     str(path)
    )
if __name__=='__main__':unittest.main()
