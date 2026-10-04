import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("ECG_API_KEY"),
    base_url=os.getenv("ECG_BASE_URL"),
)

resp = client.chat.completions.create(
    model=os.getenv("ECG_MODEL"),
    messages=[{"role":"user","content":"1+1等于几"}],
    timeout=180
)
print(resp.choices[0].message.content)
