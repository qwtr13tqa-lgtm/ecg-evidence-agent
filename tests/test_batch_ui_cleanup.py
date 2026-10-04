import unittest
from copy import deepcopy
from src.ui.batch_display import comparison,display_rows,assumptions

def entry(cat,median=0.123456):
    return {'data':{'batch_sha256':'same','operation':{'kind':'aggregate','group_by':'category','filters':[{'field':'category','op':'eq','value':cat}],'metrics':['shape_error']},'groups':[{'group_value':cat,'count':10,'metrics':{'shape_error':{'median':median,'valid_n':8,'missing_n':2}}}],'assumptions':['描述性统计']}}
class DisplayTests(unittest.TestCase):
    def test_merge_and_missing(self):
        r=comparison([entry('fn'),entry('tp')]);self.assertEqual(r[0]['FN 有效 / 总数'],'8 / 10');self.assertEqual(r[0]['FN 中位数'],0.1235)
    def test_non_mutation(self):
        e=[entry('fn'),entry('tp')];before=deepcopy(e);comparison(e);display_rows([{'score':0.123456}]);self.assertEqual(e,before)
    def test_different_batch(self):
        e=entry('tp');e['data']['batch_sha256']='other';self.assertEqual(comparison([entry('fn'),e]),[])
    def test_extra_filter(self):
        e=entry('tp');e['data']['operation']['filters'].append({'field':'score','op':'gt','value':0});self.assertEqual(comparison([entry('fn'),e]),[])
    def test_duplicate_group(self):self.assertEqual(comparison([entry('fn'),entry('fn'),entry('tp')]),[])
    def test_pagination(self):
        e=entry('tp');e['data']['has_more']=True;self.assertEqual(comparison([entry('fn'),e]),[])
    def test_deduplicate(self):self.assertEqual(assumptions([entry('fn'),entry('tp')]),['描述性统计'])
    def test_hide_id_keep_values(self):self.assertEqual(display_rows([{'analysis_id':'abc','score':None,'index':0}]),[{'模型分数':None,'样本':0}])
if __name__=='__main__':unittest.main()
