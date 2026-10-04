"""Deterministic, narrow product-help routing. Numeric requests stay on tools."""
import hashlib,re
from pathlib import Path
ALIASES={
'batch':('批量错分复核','批量复核'),
'signal':('信号分析','查看信号','查看波形'),
'chat':('当前样本问答','向本记录提问','查询范围'),
'details':('分析详情','查看分析依据'),
'export':('导出记录','导出'),
'history':('历史记录','分析ID','样本编号','样本索引','删除记录','归档'),
'upload':('上传','12导联上传'),
'guide':('新手指引','步骤指引','使用指引')}


def topic(question):
    q=question.strip()
    if len(q)>150:return None
    # Deliberately conservative: data requests and mixed intents must use tools.
    if re.search(r'多少|哪些|列出|找出|比较|计算|均值|最大值|最小值|本批|这批|这个样本|该样本|\d',q):return None
    if not any(x in q for x in ('是什么','什么用','干什么','怎么用','如何使用','怎么操作','怎么填','如何填写','怎么上传','如何上传','怎么删除','如何删除','怎么导出','如何导出')):return None
    for key,aliases in ALIASES.items():
        if any(x in q for x in aliases):return key
    return None


def is_product_help(question):return topic(question) is not None


def section(root,key):
    text=(Path(root)/'docs/user_guide.md').read_text(encoding='utf-8')
    marker='<!-- topic:'+key+' -->'
    if marker not in text:raise ValueError('Missing help topic')
    return text.split(marker,1)[1].split('<!-- topic:',1)[0].strip()


def answer(root,aid,question):
    key=topic(question)
    if key is None:raise ValueError('Not a product-help question')
    content=section(root,key);eid=aid+':product_help:'+hashlib.sha256(content.encode()).hexdigest()[:16]
    return dict(analysis_id=aid,status='completed_draft',error='',data_kind='real_ecg',
        draft={'answer':content,'evidence_ids':[eid],'knowledge_ids':[],'observations':[]},
        validation={'passed':True,'scope':'local_product_document_not_ecg_measurement'},
        evidence={eid:{'analysis_id':aid,'evidence_id':eid,'ok':True,'data':{'topic':key,'source':'docs/user_guide.md','text':content},'scope':'product_help'}},
        knowledge={},trace=[{'stage':'local_help','topic':key,'source':'docs/user_guide.md'}],
        model_calls=0,tool_calls=0,requires_review=False)


def render_page_help(st,root,key):
    with st.expander('这页怎么用'):
        st.write(section(root,key))
