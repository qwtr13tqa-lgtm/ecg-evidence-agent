import unittest
from types import SimpleNamespace
from src.agent.conversation import new_conversation, begin_turn, finish_turn
from src.reporting.conversation_export import build_conversation_export, conversation_markdown
from src.reporting.analysis_export import json_bytes


class TestConversationExport(unittest.TestCase):
    def fixture(self):
        result=SimpleNamespace(analysis_id='A',rhythm=None,to_llm_context=lambda:{'provenance':{'analysis_id':'A'}})
        s=new_conversation('A')
        for i in range(2):
            r=begin_turn(s,'A','问题'+str(i))
            finish_turn(s,'A',r,{'analysis_id':'A','status':'failed','error':'TIMEOUT','draft':{},'evidence':{}},{'API_KEY':'secret'})
        return result,s

    def test_all_turns_in_order_and_no_key(self):
        r,s=self.fixture();b=build_conversation_export(r,s)
        self.assertEqual([t['request']['question'] for t in b['turns']],['问题0','问题1'])
        self.assertNotIn('secret',json_bytes(b).decode())
        self.assertIn('第 2 轮',conversation_markdown(b).decode())

    def test_cross_analysis_rejected(self):
        r,s=self.fixture();s['turns'][1]['analysis_id']='B'
        with self.assertRaises(ValueError):build_conversation_export(r,s)

    def test_export_empty(self):
        r,s=self.fixture();s['turns']=[]
        self.assertEqual(build_conversation_export(r,s)['turns'],[])

if __name__=='__main__':unittest.main()
