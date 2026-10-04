"""Explicit synthetic software fixtures. NOT ECG measurements or clinical performance."""
from types import SimpleNamespace
from copy import deepcopy

DATA=[
 (0,1,-.9,.1,.9,1), (1,1,-.9,None,.4,0), (2,1,-.8,None,None,2),
 (3,1,.7,.2,.2,0), (4,1,.8,.3,.6,1), (5,0,.9,.4,.5,0),(6,0,-.4,.1,.2,0),
 (7,1,-.7,.2,.3,0),(8,1,-.7,.2,.1,1),(9,1,-.7,.2,.2,0)]

def fixture():
    records=[dict(index=i,label=l,score=s,reconstruction_error=r,shape_error=h,region_count=n) for i,l,s,r,h,n in DATA]
    report={'threshold':0.0,'checkpoint_sha256':'synthetic-checkpoint','adapter_sha256':'synthetic-adapter','rows':[{'index':r['index'],'label':r['label'],'score':r['score'],'prediction':int(r['score']>=0),'input_sha256':str(r['index'])*64} for r in records]}
    return {'kind':'synthetic_software_test','records':records,'report':report,'notice':'Hand constructed numbers for positive/empty/missing/tie branches; no real ECG inference.'}

class SyntheticResult:
    def __init__(self,row,report):
        self.row=deepcopy(row);self.analysis_id='synthetic-'+str(row['index']);self.input=SimpleNamespace(num_samples=4800,sampling_rate=500)
        self.model=SimpleNamespace(anomaly_score=row['score'],error_map=None)
        self.provenance={'input_sha256':str(row['index'])*64,'checkpoint_sha256':report['checkpoint_sha256'],'review_adapter_sha256':report['adapter_sha256'],'sample_index':row['index'],'crop_start_sample':100,'model_decision':{'status':'configured','threshold':0.0,'prediction':'model_anomaly' if row['score']>=0 else 'model_normal'}}
    def to_llm_context(self):
        return {'input':{'num_samples':4800,'sampling_rate':500},'model':{'anomaly_score':self.row['score'],'reconstruction_error':self.row['reconstruction_error'],'shape_error':self.row['shape_error']},'evidence':{'lead_evidence':[],'temporal_regions':[{'start':i*50,'end':i*50+48} for i in range(self.row['region_count'])]},'signal_features':{},'synthetic_software_test':True}

class SyntheticHistory:
    def __init__(self,data,no_tp_values=False):
        self.results={}
        for original in data['records']:
            row=deepcopy(original)
            if no_tp_values and row['label']==1 and row['score']>=0:row['shape_error']=None
            obj=SyntheticResult(row,data['report']);self.results[obj.analysis_id]=obj
    def list_analyses(self,limit=100,offset=0):return [{'analysis_id':aid,'sample_index':r.row['index']} for aid,r in self.results.items()][offset:offset+limit]
    def load_analysis(self,aid):return self.results[aid],None,{}
