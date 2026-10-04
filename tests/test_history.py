import hashlib
import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from src.analysis.history import HistoryRepository
from src.analysis.result import ECGAnalysisResult,ECGInputInfo,ECGModelOutput,ECGEvidenceSummary
from src.features.rhythm import RhythmFeatureExtractor
from src.tools.executor import ECGToolExecutor
from src.agent.conversation import history_context
from src.reporting.conversation_export import build_conversation_export


def fixture(aid='A',sample=0):
    signal=np.zeros((4800,12),dtype=np.float32)
    result=ECGAnalysisResult(ECGInputInfo(4800,12),ECGModelOutput(-.9,-.95,.05,
        reconstruction=signal+1,sigma=np.full((4800,1),-.2,dtype=np.float32),
        error_map=np.arange(57600,dtype=np.float32).reshape(4800,12)),ECGEvidenceSummary(),
        provenance={'analysis_id':aid,'sample_index':sample,'crop_start_sample':100,
        'input_sha256':hashlib.sha256(signal.tobytes()).hexdigest(),
        'model_decision':{'score':-.9,'status':'unconfigured','threshold':None,'prediction':None,'reason':'THRESHOLD_NOT_CONFIGURED'}},
        rhythm=RhythmFeatureExtractor().extract(signal),analysis_id=aid)
    return result,signal


def output(aid='A',status='completed_draft'):
    return {'analysis_id':aid,'status':status,'draft':{'answer':'没有可用心率'},
            'validation':{'passed':True},'evidence':{},'knowledge':{},'trace':[]}


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'history.sqlite3'
        self.repo=HistoryRepository(self.path,runtime_id='process1')
        self.result,self.signal=fixture();self.repo.save_analysis(self.result,self.signal,1.)
        self.cid=self.repo.new_conversation('A')
    def tearDown(self):self.repo.close();self.tmp.cleanup()
    def test_restart_restores_arrays_without_model(self):
        fresh=HistoryRepository(self.path,runtime_id='process2')
        r,s,a=fresh.load_analysis('A')
        np.testing.assert_array_equal(r.model.error_map,self.result.model.error_map)
        np.testing.assert_array_equal(r.model.sigma,self.result.model.sigma)
        np.testing.assert_array_equal(s,self.signal)
        self.assertEqual(r.rhythm.rr_details,self.result.rhythm.rr_details)
        self.assertEqual(a['sample'],0)
    def test_read_returns_independent_result(self):
        r=self.repo.get_result('A');r.model.error_map[:]=0
        self.assertNotEqual(self.repo.get_result('A').model.error_map[1,1],0)
    def test_tools_after_restart(self):
        e=ECGToolExecutor(HistoryRepository(self.path),'A')
        self.assertTrue(e.execute('inspect_recent_error',{'duration_seconds':.6,'lead':'V1'})['ok'])
        self.assertTrue(e.execute('inspect_rr_intervals',{})['ok'])
    def test_second_analysis_preserves_first(self):
        r,s=fixture('B',1);self.repo.save_analysis(r,s,2)
        self.assertEqual(len(self.repo.list_analyses()),2)
        self.assertEqual(self.repo.get_result('A').analysis_id,'A')
    def test_analysis_id_never_overwritten(self):
        with self.assertRaises(sqlite3.IntegrityError):self.repo.save_analysis(self.result,self.signal,5)
        self.assertEqual(self.repo.load_analysis('A')[2]['elapsed'],1.)
    def test_bad_signal_hash_no_insert(self):
        r,s=fixture('B');s[0,0]=1
        with self.assertRaises(ValueError):self.repo.save_analysis(r,s,1)
        self.assertEqual(len(self.repo.list_analyses()),1)
    def test_nonfinite_no_insert(self):
        r,s=fixture('B');r.model.error_map[0,0]=np.nan
        with self.assertRaises(ValueError):self.repo.save_analysis(r,s,1)
    def test_corrupt_blob_detected(self):
        with self.repo.connect() as db:db.execute("UPDATE analyses SET artifacts=? WHERE analysis_id='A'",(b'bad',))
        with self.assertRaises(ValueError):self.repo.load_analysis('A')
    def test_turn_durable_before_gateway(self):
        rid=self.repo.begin_turn('A',self.cid,'问题')
        self.assertEqual(HistoryRepository(self.path).conversation('A',self.cid)['pending']['request_id'],rid)
    def test_duplicate_submission_blocked(self):
        self.repo.begin_turn('A',self.cid,'问题')
        with self.assertRaises(sqlite3.IntegrityError):self.repo.begin_turn('A',self.cid,'另一个')
    def test_concurrent_submission_one_wins(self):
        def send(_):
            try:return self.repo.begin_turn('A',self.cid,'q')
            except sqlite3.IntegrityError:return None
        with ThreadPoolExecutor(2) as pool:results=list(pool.map(send,range(2)))
        self.assertEqual(sum(x is not None for x in results),1)
    def test_saved_history_used_in_next_turn(self):
        rid=self.repo.begin_turn('A',self.cid,'问心率')
        self.repo.finish_turn('A',self.cid,rid,output())
        s=HistoryRepository(self.path).conversation('A',self.cid)
        self.assertEqual(history_context(s,'A')[0]['question'],'问心率')
    def test_failure_preserved_not_memory(self):
        rid=self.repo.begin_turn('A',self.cid,'q');self.repo.finish_turn('A',self.cid,rid,output(status='failed'))
        s=self.repo.conversation('A',self.cid)
        self.assertEqual(len(s['turns']),1);self.assertEqual(history_context(s,'A'),[])
    def test_cross_analysis_write_rejected(self):
        with self.assertRaises(ValueError):self.repo.begin_turn('B',self.cid,'q')
        rid=self.repo.begin_turn('A',self.cid,'q')
        with self.assertRaises(ValueError):self.repo.finish_turn('A',self.cid,rid,output('B'))
    def test_cross_evidence_rejected(self):
        rid=self.repo.begin_turn('A',self.cid,'q');out=output();out['evidence']={'B:e':{'analysis_id':'B'}}
        with self.assertRaises(ValueError):self.repo.finish_turn('A',self.cid,rid,out)
    def test_interrupted_on_process_restart(self):
        rid=self.repo.begin_turn('A',self.cid,'q')
        new=HistoryRepository(self.path,runtime_id='process2',recover=True)
        s=new.conversation('A',self.cid)
        self.assertIsNone(s['pending']);self.assertEqual(s['turns'][0]['output']['error'],'REQUEST_INTERRUPTED')
        self.assertFalse(self.repo.finish_turn('A',self.cid,rid,output()))
    def test_same_process_does_not_interrupt(self):
        self.repo.begin_turn('A',self.cid,'q')
        new=HistoryRepository(self.path,runtime_id='process1',recover=True)
        self.assertIsNotNone(new.conversation('A',self.cid)['pending'])
    def test_explicit_abandon_blocks_late_response(self):
        rid=self.repo.begin_turn('A',self.cid,'q');self.assertTrue(self.repo.abandon_turn('A',self.cid,rid))
        self.assertFalse(self.repo.finish_turn('A',self.cid,rid,output()))
    def test_new_conversation_preserves_old(self):
        rid=self.repo.begin_turn('A',self.cid,'q');self.repo.finish_turn('A',self.cid,rid,output())
        new=self.repo.new_conversation('A')
        self.assertEqual(len(self.repo.conversations('A')),2)
        self.assertEqual(len(self.repo.conversation('A',self.cid)['turns']),1)
        self.assertEqual(self.repo.conversation('A',new)['turns'],[])
    def test_config_allowlist(self):
        rid=self.repo.begin_turn('A',self.cid,'q');self.repo.finish_turn('A',self.cid,rid,output(),{'API_KEY':'secret','model':'m'})
        self.assertNotIn('secret',json.dumps(self.repo.conversation('A',self.cid)))
    def test_threshold_snapshot_unchanged(self):
        self.result.provenance['model_decision']['threshold']=999
        self.assertIsNone(self.repo.get_result('A').provenance['model_decision']['threshold'])
    def test_backup_complete(self):
        path=Path(self.tmp.name)/'backup.db';self.repo.backup(path)
        self.assertEqual(HistoryRepository(path).get_result('A').analysis_id,'A')
    def test_limit_does_not_delete_older_turns(self):
        for i in range(20):
            rid=self.repo.begin_turn('A',self.cid,str(i));self.repo.finish_turn('A',self.cid,rid,output())
        with self.assertRaises(ValueError):self.repo.begin_turn('A',self.cid,'21')
        self.assertEqual(len(self.repo.conversation('A',self.cid)['turns']),20)
    def test_export_after_recovery(self):
        self.repo.begin_turn('A',self.cid,'q')
        new=HistoryRepository(self.path,runtime_id='process2',recover=True)
        b=build_conversation_export(new.get_result('A'),new.conversation('A',self.cid))
        self.assertEqual(b['turns'][0]['agent_output']['error'],'REQUEST_INTERRUPTED')
    def test_exclusive_app_lock(self):
        a=HistoryRepository(self.path,exclusive=True)
        try:
            with self.assertRaises(RuntimeError):HistoryRepository(self.path,exclusive=True)
        finally:a.close()

if __name__=='__main__':unittest.main()
