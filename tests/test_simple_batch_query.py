import unittest
from unittest.mock import patch
from src.review.simple_batch_query import parse_simple_query,run_simple

def report():
    rows=[{'index':i,'score':s,'label':l,'prediction':int(s>=0),'input_sha256':'a'*64} for i,s,l in [(1,.3,0),(10,.8,0),(37,.3,0),(5,-.1,1),(2,.2,1),(3,-.2,0)]]
    return {'threshold':0,'rows':rows}
class QueryTests(unittest.TestCase):
    def test_reported_question(self):self.assertEqual(parse_simple_query('找出所有误报，按分数降序'),{'category':'fp','order':'desc'})
    def test_categories(self):
        for q,c in [('找出所有漏报，按分数升序','fn'),('找出判对的正常记录，按分数降序','tn'),('列出全部记录，按分数升序','all'),('查询TP，按模型分数从高到低排序','tp')]:self.assertEqual(parse_simple_query(q)['category'],c)
    def test_extra_conditions_not_dropped(self):
        for q in ['找出所有误报，按分数降序，只要RR稳定的','找出前3个误报，按分数降序','找出所有误报，按分数降序，并解释原因','比较误报和漏报','不要找出所有误报，按分数降序']:
            self.assertIsNone(parse_simple_query(q))
    def test_stable_complete_sort_no_model(self):
        out=run_simple(None,'a','找出所有误报，按分数降序',report(),'hash')
        self.assertEqual([r['index'] for r in out['local_query']['rows']],[10,1,37]);self.assertEqual(out['model_calls'],0);self.assertEqual(out['tool_calls'],0)
        self.assertEqual(out['explanation_status'],'not_requested')
    def test_empty_selection(self):
        r=report();r['rows']=[r['rows'][-1]]
        self.assertEqual(run_simple(None,'a','找出所有误报，按分数降序',r,'hash')['local_query']['record_count'],0)
    def test_invalid_report_rejected(self):
        r=report();r['rows'][0]['prediction']=0
        with self.assertRaises(ValueError):run_simple(None,'a','找出所有误报，按分数降序',r,'hash')
    def test_history_failure_keeps_all_results(self):
        import types,sys
        module=types.ModuleType('src.review.conversation_batch')
        def fail(*args):raise OSError('unavailable')
        module.review=fail
        with patch.dict(sys.modules,{'src.review.conversation_batch':module}):
            out=run_simple(object(),'a','找出所有误报，按分数降序',report(),'hash')
        self.assertEqual(len(out['local_query']['rows']),3)
        self.assertTrue(out['local_query']['warnings'])
        self.assertEqual(out['status'],'completed_draft')
    def test_matching_history_navigation_preserved(self):
        import types,sys
        module=types.ModuleType('src.review.conversation_batch')
        module.review=lambda *a:[{'index':10,'state':'matched','source_analysis_id':'matched-id'}]
        with patch.dict(sys.modules,{'src.review.conversation_batch':module}):
            out=run_simple(object(),'a','找出所有误报，按分数降序',report(),'hash')
        self.assertEqual(out['local_query']['rows'][0]['source_analysis_id'],'matched-id')
        self.assertEqual(out['local_query']['rows'][1]['state'],'needs_local_analysis')
    def test_no_mutation(self):
        import copy
        r=report();old=copy.deepcopy(r);run_simple(None,'a','列出全部记录，按分数升序',r,'hash');self.assertEqual(r,old)
if __name__=='__main__':unittest.main()
