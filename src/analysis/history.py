"""Single-user SQLite history. Atomic snapshots, numeric NPZ blobs; no pickle.

One Streamlit process per history database. This is not user authentication.
"""
import hashlib
import io
import json
import os
import sqlite3
import uuid
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from src.analysis.result import ECGAnalysisResult, ECGInputInfo, ECGModelOutput, ECGEvidenceSummary
from src.features.rhythm import RhythmFeatures

RUNTIME_ID=str(uuid.uuid4())
ARRAY_FIELDS=('reconstruction','sigma','error_map')
CONFIG_FIELDS=('model','timeout_seconds','max_tokens','max_model_calls','max_tool_calls','sdk_retries','endpoint_sha256')
OUTPUT_FIELDS=('analysis_id','data_kind','status','error','draft','validation','limitations','requires_review',
               'evidence','knowledge','trace','model_calls','tool_calls','elapsed_seconds','gateway_failure','local_support')


def now():return datetime.now(timezone.utc).isoformat()

def encode(value):return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))


def pack(result,raw_signal):
    aid=result.analysis_id
    if not isinstance(aid,str) or not aid or result.provenance.get('analysis_id')!=aid:
        raise ValueError('Analysis identity missing or mismatched')
    shape=(result.input.num_samples,result.input.num_leads)
    signal=np.asarray(raw_signal,dtype=np.float32)
    if signal.shape!=shape or not np.isfinite(signal).all():raise ValueError('Invalid input snapshot')
    signal=np.ascontiguousarray(signal)
    actual=hashlib.sha256(signal.tobytes()).hexdigest()
    if result.provenance.get('input_sha256')!=actual:raise ValueError('Input hash mismatch')
    arrays={'input_signal':signal}
    for name in ARRAY_FIELDS:
        value=getattr(result.model,name)
        if value is None:continue
        array=np.asarray(value)
        if array.dtype.kind not in 'fiu' or not np.isfinite(array).all():raise ValueError('Invalid numeric artifact')
        if name in ('error_map','reconstruction') and array.shape!=shape:raise ValueError('Artifact shape mismatch')
        arrays[name]=array
    if 'error_map' not in arrays:raise ValueError('Error map required for later tool queries')
    context=result.to_llm_context()
    snapshot={'schema_version':'history-snapshot-1.0','analysis_id':aid,'input':asdict(result.input),
        'model':{k:context['model'][k] for k in context['model']},
        'evidence':context['evidence'],'confidence':result.confidence,'provenance':deepcopy(result.provenance),
        'rhythm':asdict(result.rhythm) if result.rhythm is not None else None}
    meta=encode(snapshot)
    bio=io.BytesIO();np.savez_compressed(bio,**arrays);blob=bio.getvalue()
    return meta,blob,hashlib.sha256(blob).hexdigest()


def unpack(meta,blob,expected_hash):
    if hashlib.sha256(blob).hexdigest()!=expected_hash:raise ValueError('Artifact checksum mismatch')
    d=json.loads(meta)
    if d.get('schema_version')!='history-snapshot-1.0':raise ValueError('Unsupported history version')
    with np.load(io.BytesIO(blob),allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files}
    model=ECGModelOutput(**d['model'],**{k:arrays.get(k) for k in ARRAY_FIELDS})
    result=ECGAnalysisResult(input=ECGInputInfo(**d['input']),model=model,
        evidence=ECGEvidenceSummary(**d['evidence']),confidence=d['confidence'],
        provenance=d['provenance'],rhythm=RhythmFeatures(**d['rhythm']) if d['rhythm'] else None,
        analysis_id=d['analysis_id'])
    signal=arrays['input_signal']
    if signal.shape!=(result.input.num_samples,result.input.num_leads):raise ValueError('Stored input shape mismatch')
    if hashlib.sha256(np.ascontiguousarray(signal).tobytes()).hexdigest()!=result.provenance.get('input_sha256'):
        raise ValueError('Stored input hash mismatch')
    return result,signal


class HistoryRepository:
    def __init__(self,path,runtime_id=None,*,recover=False,exclusive=False):
        self.path=Path(path).resolve();self.path.parent.mkdir(parents=True,exist_ok=True)
        self.runtime_id=runtime_id or RUNTIME_ID
        self._lock_file=None
        if exclusive:
            from src.analysis.history_lock import acquire_history_lock
            self._lock_file=acquire_history_lock(str(self.path)+'.lock')
        try:
            with self.connect() as db:
                db.executescript('''
                CREATE TABLE IF NOT EXISTS history_meta(version INTEGER NOT NULL);
                INSERT INTO history_meta SELECT 1 WHERE NOT EXISTS(SELECT 1 FROM history_meta);
                CREATE TABLE IF NOT EXISTS analyses(
                    analysis_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, sample_index INTEGER,
                    elapsed REAL NOT NULL, snapshot TEXT NOT NULL, artifacts BLOB NOT NULL, artifact_sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS conversations(
                    conversation_id TEXT PRIMARY KEY, analysis_id TEXT NOT NULL REFERENCES analyses(analysis_id),
                    created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS turns(
                    request_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
                    ordinal INTEGER NOT NULL, question TEXT NOT NULL, state TEXT NOT NULL,
                    owner TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT, output TEXT, configuration TEXT,
                    UNIQUE(conversation_id,ordinal));
                CREATE UNIQUE INDEX IF NOT EXISTS one_pending_turn ON turns(conversation_id) WHERE state='pending';
                ''')
                if db.execute('SELECT version FROM history_meta').fetchone()[0]!=1:raise ValueError('Unsupported database version')
                # Recover incomplete requests from a previous process, never replay remote calls.
                rows=db.execute("SELECT t.request_id,c.analysis_id FROM turns t JOIN conversations c USING(conversation_id) WHERE t.state='pending' AND owner!=?",(self.runtime_id,)).fetchall()
                for row in rows if recover else []:
                    out={'analysis_id':row['analysis_id'],'status':'failed','error':'REQUEST_INTERRUPTED',
                         'draft':{},'validation':{},'evidence':{},'knowledge':{},'trace':[]}
                    db.execute("UPDATE turns SET state='interrupted',output=?,finished_at=? WHERE request_id=? AND state='pending'",
                               (encode(out),now(),row['request_id']))
        except BaseException:
            self.close()
            raise
        # Local DB may contain ECG artifacts and questions.
        try:os.chmod(self.path,0o600)
        except OSError:pass

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=15)
        db.row_factory=sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:yield db
        finally:db.close()

    def close(self):
        if self._lock_file is not None:
            self._lock_file.close();self._lock_file=None

    def save_analysis(self,result,signal,elapsed):
        meta,blob,digest=pack(result,signal)
        if type(elapsed) not in (int,float) or not np.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid duration')
        with self.connect() as db:
            db.execute('INSERT INTO analyses VALUES(?,?,?,?,?,?,?)',
                       (result.analysis_id,now(),result.provenance.get('sample_index'),elapsed,meta,blob,digest))
        return result.analysis_id

    def load_analysis(self,aid):
        with self.connect() as db:
            row=db.execute('SELECT * FROM analyses WHERE analysis_id=?',(aid,)).fetchone()
        if row is None:raise KeyError('Analysis not found')
        result,signal=unpack(row['snapshot'],row['artifacts'],row['artifact_sha256'])
        if result.analysis_id!=aid:raise ValueError('Snapshot identity mismatch')
        return result,signal,{'analysis_id':aid,'sample':row['sample_index'],'elapsed':row['elapsed'],'created_at':row['created_at']}

    def get_result(self,aid):return self.load_analysis(aid)[0]

    def list_analyses(self,limit=50,offset=0):
        if type(limit) is not int or not 1<=limit<=100 or type(offset) is not int or offset<0:raise ValueError('Invalid page')
        with self.connect() as db:
            rows=db.execute('''SELECT a.analysis_id,a.created_at,a.sample_index,a.elapsed,
                (SELECT count(*) FROM conversations c JOIN turns t USING(conversation_id) WHERE c.analysis_id=a.analysis_id) AS turn_count
                FROM analyses a ORDER BY a.created_at DESC,a.analysis_id DESC LIMIT ? OFFSET ?''',(limit,offset)).fetchall()
        return [dict(r) for r in rows]

    def new_conversation(self,aid):
        cid=str(uuid.uuid4())
        with self.connect() as db:db.execute('INSERT INTO conversations VALUES(?,?,?)',(cid,aid,now()))
        return cid

    def conversations(self,aid):
        with self.connect() as db:
            rows=db.execute('SELECT * FROM conversations WHERE analysis_id=? ORDER BY created_at DESC',(aid,)).fetchall()
        return [dict(r) for r in rows]

    def conversation(self,aid,cid):
        with self.connect() as db:
            c=db.execute('SELECT analysis_id FROM conversations WHERE conversation_id=?',(cid,)).fetchone()
            if c is None or c['analysis_id']!=aid:raise ValueError('Conversation analysis mismatch')
            rows=db.execute('SELECT * FROM turns WHERE conversation_id=? ORDER BY ordinal',(cid,)).fetchall()
        turns=[];pending=None
        for row in rows:
            if row['state']=='pending':pending={'request_id':row['request_id'],'question':row['question'],'started_at':row['started_at']};continue
            turns.append({'analysis_id':aid,'request_id':row['request_id'],'question':row['question'],
                'output':json.loads(row['output']),'configuration':json.loads(row['configuration'] or '{}')})
        return {'analysis_id':aid,'conversation_id':cid,'turns':turns,'pending':pending}

    def begin_turn(self,aid,cid,question):
        if not isinstance(question,str) or not question.strip() or len(question)>4000:raise ValueError('Invalid question')
        rid=str(uuid.uuid4())
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            c=db.execute('SELECT analysis_id FROM conversations WHERE conversation_id=?',(cid,)).fetchone()
            if c is None or c['analysis_id']!=aid:raise ValueError('Conversation analysis mismatch')
            count=db.execute('SELECT count(*) FROM turns WHERE conversation_id=?',(cid,)).fetchone()[0]
            if count>=20:raise ValueError('Conversation limit reached; start a new conversation')
            db.execute('INSERT INTO turns VALUES(?,?,?,?,?,?,?,?,?,?)',
                       (rid,cid,count+1,question.strip(),'pending',self.runtime_id,now(),None,None,'{}'))
        return rid

    def finish_turn(self,aid,cid,rid,output,configuration=None):
        if output.get('analysis_id')!=aid:raise ValueError('Output analysis mismatch')
        for eid,item in (output.get('evidence') or {}).items():
            if item.get('analysis_id')!=aid or not eid.startswith(aid+':') or item.get('evidence_id')!=eid:
                raise ValueError('Evidence analysis mismatch')
        out={k:deepcopy(v) for k,v in output.items() if k in OUTPUT_FIELDS}
        if out.get('status') not in ('completed_draft','failed'):raise ValueError('Invalid terminal state')
        conf={k:v for k,v in (configuration or {}).items() if k in CONFIG_FIELDS}
        with self.connect() as db:
            c=db.execute('SELECT analysis_id FROM conversations WHERE conversation_id=?',(cid,)).fetchone()
            if c is None or c['analysis_id']!=aid:raise ValueError('Conversation analysis mismatch')
            updated=db.execute("UPDATE turns SET state=?,finished_at=?,output=?,configuration=? WHERE request_id=? AND conversation_id=? AND state='pending' AND owner=?",
                ('completed' if out['status']=='completed_draft' else 'failed',now(),encode(out),encode(conf),rid,cid,self.runtime_id)).rowcount
        return updated==1

    def abandon_turn(self,aid,cid,rid):
        # User can mark a disconnected wait as interrupted; this does not cancel the provider request.
        output={'analysis_id':aid,'status':'failed','error':'WAIT_ABANDONED',
                'draft':{},'validation':{},'evidence':{},'knowledge':{},'trace':[]}
        return self.finish_turn(aid,cid,rid,output)

    def backup(self,destination):
        dest=Path(destination)
        if dest.exists():raise FileExistsError(dest)
        dest.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as source:
            target=sqlite3.connect(dest)
            try:source.backup(target)
            finally:target.close()
        try:os.chmod(dest,0o600)
        except OSError:pass
        return dest
