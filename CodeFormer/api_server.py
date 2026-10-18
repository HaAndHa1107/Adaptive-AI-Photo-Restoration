import asyncio
import os
import secrets
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / 'uploads'
RESULTS = BASE / 'results' / 'app_jobs'
UPLOADS.mkdir(exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
app = FastAPI(title='AI Photo Restoration')
security = HTTPBasic()
job_lock = asyncio.Lock()


def authorize(credentials: HTTPBasicCredentials = Depends(security)):
    username = os.getenv('PHOTO_APP_USER')
    password = os.getenv('PHOTO_APP_PASSWORD')
    if not username or not password:
        raise HTTPException(status_code=503, detail='請先設定 PHOTO_APP_USER 和 PHOTO_APP_PASSWORD')
    if not (secrets.compare_digest(credentials.username, username) and
            secrets.compare_digest(credentials.password, password)):
        raise HTTPException(status_code=401, detail='帳號或密碼錯誤', headers={'WWW-Authenticate': 'Basic'})


@app.get('/health')
def health():
    return {'status': 'online'}


@app.get('/', response_class=HTMLResponse)
def home(_: None = Depends(authorize)):
    return '''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Photo Restoration</title><style>
body{font-family:system-ui,sans-serif;background:#111827;color:white;max-width:520px;margin:30px auto;padding:20px}
main{background:#1f2937;border-radius:18px;padding:24px}button{background:#6366f1;color:white;border:0;border-radius:10px;padding:14px;width:100%;font-size:16px;margin-top:15px}
input{margin:20px 0;width:100%}img{width:100%;border-radius:12px;margin-top:14px}a{color:#a5b4fc}#status{line-height:1.6}
</style></head><body><main><h2>AI Photo Restoration</h2><p>CodeFormer + Real-ESRGAN 自適應修復</p>
<input id="photo" type="file" accept="image/jpeg,image/png"><button id="btn" onclick="restore()">開始 AI 修復</button>
<p id="status">請選擇照片</p><img id="before" hidden><img id="after" hidden><p><a id="save" hidden download="restored.png">下載修復照片</a></p></main>
<script>let lastUrl=null;async function restore(){const f=document.getElementById('photo').files[0],st=document.getElementById('status'),btn=document.getElementById('btn');
if(!f){st.textContent='請先選照片';return}btn.disabled=true;st.textContent='正在上傳並使用 GPU 修復，請稍候…';
const form=new FormData();form.append('file',f);document.getElementById('before').src=URL.createObjectURL(f);document.getElementById('before').hidden=false;
try{const r=await fetch('/restore',{method:'POST',body:form});if(!r.ok){throw new Error(await r.text())}const blob=await r.blob();if(lastUrl)URL.revokeObjectURL(lastUrl);lastUrl=URL.createObjectURL(blob);
const out=document.getElementById('after'),save=document.getElementById('save');out.src=lastUrl;out.hidden=false;save.href=lastUrl;save.hidden=false;st.textContent='修復完成！上方原圖，下方修復圖。';
}catch(e){st.textContent='修復失敗：'+e.message}finally{btn.disabled=false}}
</script></body></html>'''


@app.post('/restore')
async def restore(file: UploadFile = File(...), _: None = Depends(authorize)):
    if file.content_type not in {'image/jpeg', 'image/png'}:
        raise HTTPException(400, '僅支援 JPG / PNG')
    data = await file.read(20 * 1024 * 1024 + 1)
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(413, '圖片超過 20 MB')
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, '無效圖片')
    h, w = image.shape[:2]
    if h * w > 4_000_000:
        raise HTTPException(413, '測試版限制原圖最多 400 萬像素')
    job_id = uuid.uuid4().hex
    upload_path = UPLOADS / f'{job_id}.png'
    output_dir = RESULTS / job_id
    output_dir.mkdir(parents=True)
    cv2.imwrite(str(upload_path), image)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(BASE), str(BASE.parent), env.get('PYTHONPATH', '')]))
    cmd = [sys.executable, str(BASE / 'run.py'), '--input', str(upload_path), '--output-dir', str(output_dir), '--app-mode']
    async with job_lock:
        try:
            process = await asyncio.create_subprocess_exec(*cmd, cwd=str(BASE), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            try:
                stdout, _ = await asyncio.wait_for(process.communicate(), timeout=600)
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                raise HTTPException(504, '修復超時（10 分鐘）')
            log = stdout.decode('utf-8', errors='replace')
            print(log[-6000:])
            result_path = output_dir / 'output_adaptive.png'
            if process.returncode != 0 or not result_path.is_file():
                raise HTTPException(500, 'AI 修復失敗，請查看 PyCharm Terminal 日誌')
            return FileResponse(result_path, media_type='image/png', filename='restored.png')
        finally:
            upload_path.unlink(missing_ok=True)
