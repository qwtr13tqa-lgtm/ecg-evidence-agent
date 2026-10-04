"""Configuration check by default; --send sends one synthetic tool request."""
import argparse
import json
import os
from src.agent.failure_help import failure_help


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--send',action='store_true')
    a=p.parse_args();key=os.getenv('ECG_API_KEY','')
    check={'key_present':bool(key),'key_contains_whitespace':any(c.isspace() for c in key),
           'key_has_bearer_prefix':key.lower().startswith('bearer ')}
    print(json.dumps(check,ensure_ascii=False))
    if not key or check['key_contains_whitespace'] or check['key_has_bearer_prefix']:
        raise SystemExit('Fix ECG_API_KEY in the terminal that starts Streamlit; never paste it into chat.')
    if not a.send:return
    from src.agent.gateway import ToolGateway
    from src.agent.tool_protocol import spec
    gateway=ToolGateway(timeout=30,max_tokens=128)
    try:
        out=gateway.complete([{'role':'user','content':'Synthetic connectivity test. Call ping with value=1. No other text.'}],
                             [spec('ping','Connectivity test',{'value':{'type':'integer'}},('value',))])
        print(json.dumps({'status':'response_received','finish_reason':out.get('finish_reason'),
                          'tool_call_count':len(out.get('tool_calls',[])),
                          'scope':'connectivity_only_not_full_agent_task'},ensure_ascii=False))
    except Exception as exc:
        report=failure_help({'status':'failed','trace':[{'error_type':type(exc).__name__,'failure_phase':'gateway'}]})
        report['http_status']=getattr(exc,'status_code',None)
        print(json.dumps(report,ensure_ascii=False));raise SystemExit(1)
    finally:gateway.close()

if __name__=='__main__':main()
