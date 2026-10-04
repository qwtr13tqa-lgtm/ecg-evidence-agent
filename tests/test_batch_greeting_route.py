import unittest
from src.review.simple_batch_query import parse_simple_query,run_simple
class GreetingTests(unittest.TestCase):
    def test_original_question(self):
        self.assertEqual(parse_simple_query('你好 找出所有误报，按分数降序'),{'category':'fp','order':'desc'})
    def test_greeting_variants(self):
        for prefix in ('','你好，','您好！','你好。请帮我','请','麻烦你'):
            with self.subTest(prefix=prefix):self.assertEqual(parse_simple_query(prefix+'找出所有漏报，按分数升序'),{'category':'fn','order':'asc'})
    def test_extra_conditions_preserved(self):
        for question in ('你好 找出所有误报，按分数降序，只要RR稳定的','你好 不要找出所有误报，按分数降序','你好 找出所有误报，按分数降序，再解释原因','你好 找出前3条误报，按分数降序','你好'):
            with self.subTest(question=question):self.assertIsNone(parse_simple_query(question))
    def test_execution_without_gateway(self):
        report={'threshold':0,'rows':[{'index':10,'label':0,'prediction':1,'score':.5,'input_sha256':'a'*64}]}
        out=run_simple(None,'analysis','你好 找出所有误报，按分数降序',report,'batch')
        self.assertEqual(out['model_calls'],0)
        self.assertEqual(out['local_query']['record_count'],1)
        self.assertEqual(out['local_query']['rows'][0]['index'],10)
if __name__=='__main__':unittest.main()
