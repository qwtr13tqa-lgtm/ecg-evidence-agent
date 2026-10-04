"""Run-bound references compiled from the visible authoritative evidence projection."""
import hashlib
import json
import re
from copy import deepcopy
from src.agent.answer_validator import AnswerValidationError

SCHEMA={'type':'object','additionalProperties':False,'properties':{
    'answer':{'type':'string','description':'简洁回答；数字可用{{引用编号}}占位，由程序填入原值。'},
    'citation_ids':{'type':'array','minItems':1,'maxItems':80,'uniqueItems':True,
                    'items':{'type':'string'},'description':'仅选择本次reference_catalog中的编号，覆盖实际陈述；至少一项初始批量数值。'}},
    'required':['answer','citation_ids']}
SKIP={'operation','assumptions','batch_sha256','source_analysis_id','analysis_id','source_evidence_id',
      'config_sha256','dataset_sha256','labels_sha256','scores_file_sha256','formula'}

def catalog_messages(messages,evidence):
    messages=deepcopy(messages);body=json.loads(messages[1]['content']);registry={};views=[]
    def esc(key):return str(key).replace('~','~0').replace('/','~1')
    for view in body['working_evidence']:
        eid=view['evidence_id'];data=view['data']
        def walk(value,path):
            if isinstance(value,dict):
                return {k:walk(v,path+'/'+esc(k)) for k,v in value.items() if k not in SKIP}
            if isinstance(value,list):
                return [walk(v,path+'/'+str(i)) for i,v in enumerate(value)]
            if isinstance(value,str) and path.rsplit('/',1)[-1] not in ('lead','top_lead','prediction','measurement_status','status','calibration_status'):
                return value
            ref='C'+hashlib.sha256(json.dumps([eid,path],ensure_ascii=False).encode()).hexdigest()[:12]
            observation={'evidence_id':eid,'path':path,'value':deepcopy(value)}
            if ref in registry and registry[ref]!=observation:raise ValueError('CITATION_ID_COLLISION')
            registry[ref]=observation
            return [ref,value]
        refs=walk(data,'')
        excerpts=[{'original_path':r['original_path'],'data':walk(r['data'],r['original_path'])} for r in view.get('row_excerpts',[])]
        views.append({'evidence_id':eid,'scope':view.get('scope'),'row_selection':view.get('row_selection'),
                      'operation':data.get('operation'),'reference_catalog':refs,'row_excerpts':excerpts})
    body['working_evidence']=views
    messages[1]['content']=json.dumps(body,ensure_ascii=False,separators=(',',':'))
    messages[0]['content']+='''\n本批次提交协议：submit_answer只填写answer和citation_ids。引用编号ref由程序绑定原始证据、路径、值，不能自行编写observations/evidence_ids或猜数组下标。reference_catalog及row_excerpts.data可引用数值及状态用[编号,原值]表示，其他字符串为上下文说明；结合所属对象的category/group_value、field/metrics字段、样本index、导联及窗口核对含义。数字可写{{C编号}}，程序只替换为该编号原始值；编号必须在citation_ids中。不要把不同对象的编号混用。至少引用一项scope=explicit_batch_report的数值。引用正确不代表因果成立。'''
    return messages,registry

def expand_answer(answer,registry):
    # Backwards-compatible legacy submissions are validated unchanged, never silently repaired.
    if isinstance(answer,dict) and 'citation_ids' not in answer:return deepcopy(answer),[]
    def reject(code,index=0,ref=None):
        exc=AnswerValidationError(code,index,None)
        if isinstance(ref,str):exc.details['citation_id']=ref[:80]
        raise exc
    if not isinstance(answer,dict) or set(answer)!={'answer','citation_ids'}:reject('CITATION_SCHEMA_INVALID')
    ids=answer['citation_ids'];text=answer['answer']
    if not isinstance(text,str) or not text.strip():reject('CITATION_SCHEMA_INVALID')
    if not isinstance(ids,list) or not 1<=len(ids)<=80:reject('CITATION_SCHEMA_INVALID')
    seen=set();observations=[]
    for i,ref in enumerate(ids):
        if not isinstance(ref,str) or ref not in registry:reject('CITATION_UNKNOWN',i,ref)
        if ref in seen:reject('CITATION_DUPLICATE',i,ref)
        seen.add(ref);observations.append(deepcopy(registry[ref]))
    def replace(match):
        ref=match.group(1)
        if ref not in seen:reject('CITATION_PLACEHOLDER_UNBOUND',ref=ref)
        value=registry[ref]['value']
        return value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,allow_nan=False)
    text=re.sub(r'\{\{([^{}]+)\}\}',replace,text)
    return {'answer':text,'evidence_ids':list(dict.fromkeys(o['evidence_id'] for o in observations)),
            'knowledge_ids':[],'observations':observations},list(ids)
