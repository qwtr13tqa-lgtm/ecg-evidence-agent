import importlib.util
from pathlib import Path
import sqlite3
import tempfile
from contextlib import contextmanager
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'src/ui'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
w=load('ecg_workspace');d=load('region_diagnostics')

class Repo:
    def __init__(self,path):self.path=path
    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path);db.row_factory=sqlite3.Row;db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:yield db
        finally:db.close()

class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.repo=Repo(Path(self.tmp.name)/'test.sqlite3')
        with self.repo.connect() as db:
            db.executescript('CREATE TABLE analyses(analysis_id TEXT PRIMARY KEY,created_at TEXT,sample_index INTEGER,elapsed REAL,snapshot TEXT);CREATE TABLE conversations(conversation_id TEXT PRIMARY KEY,analysis_id TEXT);CREATE TABLE turns(conversation_id TEXT);')
            db.executemany('INSERT INTO analyses VALUES(?,?,?,?,?)',[(f'id{i:03}',f'2026-09-{i%28+1:02}T00:00:00',i%4,0.2,'original') for i in range(25)])
            db.execute("INSERT INTO conversations VALUES('chat','id000')")
            db.execute("INSERT INTO turns VALUES('chat')")
        w.initialize(self.repo)
    def tearDown(self):self.tmp.cleanup()
    def test_pagination(self):
        one=w.list_page(self.repo);two=w.list_page(self.repo,page=2)
        self.assertEqual(one['total'],25);self.assertEqual(len(one['rows']),10)
        self.assertFalse(set(r['analysis_id'] for r in one['rows']) & set(r['analysis_id'] for r in two['rows']))
    def test_sample_filter(self):
        result=w.list_page(self.repo,sample=3)
        self.assertEqual(result['total'],6)
        self.assertTrue(all(r['sample_index']==3 for r in result['rows']))
    def test_archive_restore_preserves_records(self):
        w.update_record(self.repo,'id000',archived=True)
        self.assertEqual(w.list_page(self.repo)['total'],24)
        self.assertEqual(w.list_page(self.repo,visibility='archived')['rows'][0]['turn_count'],1)
        w.update_record(self.repo,'id000',archived=False)
        self.assertEqual(w.list_page(self.repo)['total'],25)
        with self.repo.connect() as db:
            self.assertEqual(db.execute("SELECT snapshot FROM analyses WHERE analysis_id='id000'").fetchone()[0],'original')
            self.assertEqual(db.execute('SELECT count(*) FROM turns').fetchone()[0],1)
    def test_literal_search_and_rename(self):
        w.update_record(self.repo,'id000',title="样本复核_%'")
        self.assertEqual(w.list_page(self.repo,query="_%'")['total'],1)
        self.assertEqual(w.list_page(self.repo,query="' OR 1=1 --")['total'],0)
    def test_last_page_clamped(self):
        result=w.list_page(self.repo,page=99)
        self.assertEqual(result['page'],3);self.assertEqual(len(result['rows']),5)
    def test_persistent_guide(self):
        self.assertFalse(w.guide_seen(self.repo));w.mark_guide_seen(self.repo)
        second=Repo(self.repo.path);w.initialize(second)
        self.assertTrue(w.guide_seen(second))
    def test_unknown_record(self):
        with self.assertRaises(KeyError):w.update_record(self.repo,'missing',archived=True)

class RegionTests(unittest.TestCase):
    config={'version':'evidence-1.0','region_threshold':.8,'min_region_length':48,'merge_gap':24,'top_k_regions':3,'normalization_percentiles':[5,95]}
    def signal(self,spans):
        x=np.zeros((1000,12),dtype=np.float32)
        for s,e in spans:x[s:e]=1
        return x
    def test_degenerate_empty(self):
        r=d.diagnose(self.signal([]),self.config,[],500)
        self.assertEqual(r['status'],'verified');self.assertTrue(r['normalization_degenerate'])
    def test_short_regions_empty(self):
        r=d.diagnose(self.signal([(100,120),(300,320),(500,520)]),self.config,[],500)
        self.assertEqual(r['raw_regions'],3);self.assertEqual(r['excluded_short_regions'],3)
        self.assertEqual(r['status'],'verified')
    def test_merge_then_length(self):
        r=d.diagnose(self.signal([(100,130),(140,170)]),self.config,[{'start':100,'end':170}],500)
        self.assertEqual(r['status'],'verified');self.assertEqual(r['raw_regions'],2);self.assertEqual(r['merged_regions'],1)
        self.assertEqual(r['min_region_length_seconds'],.096)
    def test_no_silent_replacement_of_saved_regions(self):
        r=d.diagnose(self.signal([(100,200)]),self.config,[],500)
        self.assertEqual(r['status'],'inconsistent')
    def test_missing_configuration(self):
        self.assertEqual(d.diagnose(self.signal([]),{},[],500)['status'],'unavailable')
    def test_unknown_version(self):
        self.assertEqual(d.diagnose(self.signal([]),dict(self.config,version='future'),[],500)['status'],'unavailable')
    def test_bad_error_map(self):
        x=self.signal([]);x[0,0]=np.nan
        self.assertEqual(d.diagnose(x,self.config,[],500)['status'],'unavailable')
    def test_top_k_counts(self):
        cfg=dict(self.config,top_k_regions=1)
        r=d.diagnose(self.signal([(100,200),(300,400)]),cfg,[{'start':100,'end':200}],500)
        self.assertEqual(r['status'],'verified');self.assertEqual(r['top_k_omitted'],1)

if __name__=='__main__':unittest.main()
