import ssl
import zipfile
import io
import urllib3
from pathlib import Path
import requests
import time
from dotenv import load_dotenv
load_dotenv()
import os

token = os.environ["MINERU_API_TOKEN"]
urllib3.disable_warnings()
ssl._create_default_https_context = ssl._create_unverified_context

API = "https://mineru.net/api/v4"

r = requests.post(f"{API}/file-urls/batch", headers={"Authorization": f"Bearer {token}"},
                  json={"files": [{"name": "test.pdf"}], "model_version": "vlm"}, verify=False)
d = r.json()["data"]
upload_url = d["file_urls"][0]
batch_id = d["batch_id"]
print(f"1. batch_id={batch_id[:8]}...")

requests.put(upload_url, data=Path("doc/test.pdf").read_bytes())
print("2. 上传成功")

print("3. 轮询中...", end=" ")
zip_url = None
for i in range(20):
    r = requests.get(f"{API}/extract-results/batch/{batch_id}",
                     headers={"Authorization": f"Bearer {token}"}, verify=False)
    data = r.json()["data"]
    state = data["extract_result"][0]["state"]
    print(state, end=" ")
    if state == "done":
        zip_url = data["extract_result"][0]["full_zip_url"]
        break
    elif state == "failed":
        err = data["extract_result"][0].get("err_msg", "")
        print(f"\n    failed: {err}")
        break
    time.sleep(3)

if not zip_url:
    print("\n超时")
    exit(1)

out_dir = Path("output") / "test"
out_dir.mkdir(parents=True, exist_ok=True)
r = requests.get(zip_url, verify=False)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(out_dir)

md_src = out_dir / "full.md"
if md_src.exists():
    final = out_dir / "test.md"
    md_src.rename(final)
    content = final.read_text(encoding="utf-8")
    print(f"\n4. 输出: {final.resolve()}")
    print(f"   大小: {final.stat().st_size} bytes / {len(content)} 字符")
    print("   前 5 行:")
    for line in content.splitlines()[:5]:
        print(f"     {line}")
else:
    print(f"\n4. 无 full.md, 目录: {list(out_dir.iterdir())}")
