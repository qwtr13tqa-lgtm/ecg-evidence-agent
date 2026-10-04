"""Failure explanation from trace metadata; never infer a missing HTTP cause."""
def failure_help(output):
    # CROSS_BUDGET_HELP_V1
    if output.get('error')=='CONTEXT_BUDGET_EXHAUSTED':
        return {'error_type':'ContextBudget','failure_phase':'context_budget','message':'本轮上下文超过本地预算，不是API密钥配置错误。请查看trace中context_budget字符数；保留失败记录，不会自动重试。','original_status':output.get('status'),'automatic_retry':False}
    trace=output.get('trace') or []
    last=next((t for t in reversed(trace) if t.get('error_type')), {})
    kind=last.get('error_type','unknown')
    help_text={
        'APITimeoutError':'请求超过等待时间。可稍后手动重试；重复超时需检查网关负载和请求诊断，延长超时不保证解决。',
        'AuthenticationError':'认证失败。请在启动页面的终端重新设置有效 ECG_API_KEY，并重启服务。',
        'APIConnectionError':'连接失败。检查本机到已配置网关的网络连接。',
        'RateLimitError':'网关限流或额度受限。检查配额，稍后手动重试。',
        'InternalServerError':'网关或上游服务返回错误。查看诊断中的HTTP状态，检查可用模型通道。',
        'ValueError':'配置或协议错误。检查环境变量和本轮调用阶段。',
    }.get(kind,'现有记录不足以确定原因。请查看本轮调用轨迹和 evaluation.inspect_gateway 的诊断。')
    return {'error_type':kind,'failure_phase':last.get('failure_phase') or last.get('stage'),
            'message':help_text,'original_status':output.get('status'),'automatic_retry':False}


class DiagnosticGateway:
    """Preserve safe exception metadata when the existing graph catches the exception."""
    def __init__(self, gateway):
        self.gateway=gateway
        self.last_error=None

    def complete(self,messages,tools):
        try:return self.gateway.complete(messages,tools)
        except Exception as exc:
            status=getattr(exc,'status_code',None)
            self.last_error={'error_type':type(exc).__name__,
                             'http_status':status if type(status) is int else None}
            raise
