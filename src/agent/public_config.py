"""Explicit gateway configuration; importing never requires credentials."""
import os
from urllib.parse import urlsplit

def gateway_config():
    values = {k: os.environ.get(k, "").strip() for k in
              ("ECG_API_KEY", "ECG_BASE_URL", "ECG_MODEL")}
    missing = [k for k, v in values.items() if not v]
    if missing:
        raise ValueError("缺少网关配置：" + ", ".join(missing))
    try:
        url = urlsplit(values["ECG_BASE_URL"])
        valid = url.scheme in ("https", "http") and bool(url.hostname)
        valid = valid and not (url.username or url.password or url.query or url.fragment)
        url.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("ECG_BASE_URL 应为 http(s) 服务地址，不能含凭据、查询参数或片段")
    return values
