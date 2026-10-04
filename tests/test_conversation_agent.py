import unittest
from src.agent.conversation import new_conversation, begin_turn, finish_turn, ConversationGateway
from tests.test_stage2_agent import FakeGateway, answer, call
from src.agent.demo_analysis import synthetic_analysis
from src.agent.evidence_agent import ECGEvidenceAgent
from src.knowledge.retriever import BM25Retriever


class TestConversationAgent(unittest.TestCase):
    def test_two_turns_through_graph(self):
        store,result=synthetic_analysis();aid=result.analysis_id;s=new_conversation(aid)
        for q in ('输入有多长？','那最后0.6秒呢？'):
            rid=begin_turn(s,aid,q)
            g=FakeGateway([call(),answer])
            out=ECGEvidenceAgent(store,BM25Retriever([]),ConversationGateway(g,s,aid)).run(
                aid,q,allow_external=True,data_kind='synthetic_software_test')
            self.assertEqual(out['status'],'completed_draft')
            finish_turn(s,aid,rid,out)
        self.assertIn('untrusted_conversation_history',g.requests[0][2]['content'])
        self.assertEqual(g.requests[0][-1]['content'],'那最后0.6秒呢？')
        self.assertEqual(len(s['turns']),2)
        self.assertEqual(out['trace'][1]['tool'],'inspect_error_window')

if __name__=='__main__':unittest.main()
