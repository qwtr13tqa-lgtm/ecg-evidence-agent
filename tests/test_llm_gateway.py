"""学校网关连接测试：仅发送虚构数据。

通过显式运行本文件才会调用接口。
不读取 ECG 数据，不保存或打印密钥。
"""

import json
import os

from openai import OpenAI


from src.agent.public_config import gateway_config


def main():
    config = gateway_config()
    api_key = config["ECG_API_KEY"]
    model = config["ECG_MODEL"]

    # 完全虚构的软件测试数据，不代表患者测量。
    payload = {
        "case_id": "SYNTHETIC_GATEWAY_TEST",
        "heart_rate_bpm": 75.0,
        "measurement_status": "unvalidated",
    }

    client = OpenAI(
        api_key=api_key,
        base_url=config["ECG_BASE_URL"],
        timeout=60.0,
        max_retries=0,
    )

    print(f"Model: {model}")
    print("Sending synthetic data only...")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a JSON interface test assistant. "
                        "The input is fictional software test data. "
                        "Return only one JSON object with exactly these keys: "
                        "case_id, heart_rate_bpm, measurement_status. "
                        "Copy all values exactly from the input. "
                        "Do not provide medical interpretation. "
                        "Do not include Markdown or extra text."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(payload),
                },
            ],
            temperature=0.0,
            max_tokens=256,
        )
    except Exception as exc:
        # 不打印完整异常响应，避免泄漏请求细节。
        status = getattr(exc, "status_code", None)
        print(f"Request failed: {type(exc).__name__}")
        if status is not None:
            print(f"HTTP status: {status}")

        response = getattr(exc, "response", None)
        if response is not None:
            # 只读取服务端错误消息，不打印请求头或密钥。
            try:
                body = response.json()
                error = body.get("error", {})
                if isinstance(error, dict):
                    message = str(error.get("message", ""))
                    message = message.replace(api_key, "[REDACTED]")
                    print("Gateway message:", message[:1000])
            except ValueError:
                print("Gateway returned a non-JSON error response.")
        raise SystemExit(1)
    finally:
        client.close()

    if not response.choices:
        raise SystemExit("模型未返回 choices。")

    choice = response.choices[0]
    text = choice.message.content

    print(f"Finish reason: {choice.finish_reason}")

    if choice.finish_reason != "stop":
        raise SystemExit("响应未正常结束，本次测试不通过。")

    if not isinstance(text, str) or not text.strip():
        raise SystemExit("模型返回内容为空。")

    print("\nModel output:")
    print(text)

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        raise SystemExit(
            "输出不是严格 JSON；先保留错误，不自动剥离或修补。"
        )

    if not isinstance(result, dict) or set(result) != set(payload):
        raise SystemExit("返回字段不符合预期。")

    for key, expected in payload.items():
        actual = result[key]

        if key == "heart_rate_bpm":
            valid = (
                type(actual) in (int, float)
                and actual == expected
            )
        else:
            valid = (
                type(actual) is type(expected)
                and actual == expected
            )

        if not valid:
            raise SystemExit(f"字段值不符合预期：{key}")

    print("\nLLM GATEWAY SMOKE TEST: PASS")
    print("仅确认虚构数据调用与 JSON 返回正常。")


if __name__ == "__main__":
    main()