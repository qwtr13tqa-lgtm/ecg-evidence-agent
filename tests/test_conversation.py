import unittest
from copy import deepcopy
from src.agent.conversation import new_conversation, begin_turn, finish_turn, history_context, ConversationGateway


def output(aid='a', answer='9个RR', status='completed_draft'):
    return {'analysis_id':aid,'status':status,'draft':{'answer':answer},'validation':{'passed':True}}


def add(s,question='有哪些RR？',out=None):
    rid=begin_turn(s,s['analysis_id'],question)
    finish_turn(s,s['analysis_id'],rid,out or output())


class TestConversation(unittest.TestCase):
    def test_followup_receives_prior_turn(self):
        s=new_conversation('a');add(s)
        class Gateway:
            def complete(self,messages,tools):self.messages=messages;return {'ok':True}
        g=Gateway();original=[{'role':'system','content':'rules'},{'role':'user','content':'current analysis'},{'role':'user','content':'其中最长的是哪个？'}]
        snapshot=deepcopy(original)
        ConversationGateway(g,s,'a').complete(original,[])
        self.assertIn('9个RR',g.messages[2]['content'])
        self.assertEqual(g.messages[-1]['content'],'其中最长的是哪个？')
        self.assertEqual(original,snapshot)
        self.assertIn('本轮',g.messages[0]['content'])

    def test_first_turn_no_extra_context(self):
        self.assertEqual(history_context(new_conversation('a'),'a'),[])

    def test_cross_analysis_rejected(self):
        s=new_conversation('a')
        with self.assertRaises(ValueError):history_context(s,'b')
        with self.assertRaises(ValueError):begin_turn(s,'b','x')

    def test_output_mismatch(self):
        s=new_conversation('a');r=begin_turn(s,'a','x')
        with self.assertRaises(ValueError):finish_turn(s,'a',r,output('b'))

    def test_stale_result_ignored(self):
        s=new_conversation('a');r=begin_turn(s,'a','x')
        fresh=new_conversation('a');begin_turn(fresh,'a','new')
        self.assertFalse(finish_turn(fresh,'a',r,output()))
        self.assertEqual(fresh['turns'],[])

    def test_failure_kept_not_used_as_memory(self):
        s=new_conversation('a');add(s);add(s,out=output(status='failed'))
        self.assertEqual(len(s['turns']),2)
        self.assertEqual(len(history_context(s,'a')),1)

    def test_bounded_last_three(self):
        s=new_conversation('a')
        for i in range(5):add(s,str(i))
        self.assertEqual([t['question'] for t in history_context(s,'a')],['2','3','4'])

    def test_oversize_history_not_truncated_into_fact(self):
        s=new_conversation('a');add(s,out=output(answer='x'*10000))
        self.assertEqual(history_context(s,'a'),[])

    def test_pending_blocks_duplicate(self):
        s=new_conversation('a');begin_turn(s,'a','x')
        with self.assertRaises(ValueError):begin_turn(s,'a','y')

    def test_snapshot_not_mutated(self):
        s=new_conversation('a');o=output();add(s,out=o);o['draft']['answer']='bad'
        self.assertEqual(s['turns'][0]['output']['draft']['answer'],'9个RR')

    def test_turn_limit(self):
        s=new_conversation('a')
        for i in range(20):add(s)
        with self.assertRaises(ValueError):begin_turn(s,'a','x')

if __name__=='__main__':unittest.main()
