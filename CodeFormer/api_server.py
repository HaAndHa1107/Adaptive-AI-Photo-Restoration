
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

app = FastAPI(title="AI Photo Restoration")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_SIZE = 20 * 1024 * 1024


@app.get("/health")
def health():
    return {
        "status": "online",
        "message": "AI Restoration Server is running"
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport"
              content="width=device-width, initial-scale=1">
        <title>AI Photo Restoration</title>
        <style>
            body {
                font-family: sans-serif;
                background: #111827;
                color: white;
                padding: 30px 18px;
                max-width: 450px;
                margin: auto;
            }
            h1 { text-align: center; }
            .panel {
                background: #1f2937;
                padding: 22px;
                border-radius: 16px;
            }
            input { width: 100%; margin: 18px 0; }
            button {
                width: 100%;
                padding: 15px;
                background: #6366f1;
                border: 0;
                border-radius: 10px;
                color: white;
                font-size: 16px;
            }
            #status { margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>AI Photo Restoration</h1>
        <div class="panel">
            <p>選擇照片並上傳至你的電腦</p>
            <input type="file" id="photo" accept="image/*">
            <button onclick="upload()">上傳照片</button>
            <p id="status">等待選擇照片...</p>
        </div>

        <script>
            async function upload() {
                const file = document.getElementById("photo").files[0];
                const status = document.getElementById("status");

                if (!file) {
                    status.textContent = "請先選擇照片";
                    return;
                }

                const form = new FormData();
                form.append("file", file);

                status.textContent = "正在上傳...";

                try {
                    const response = await fetch("/upload", {
                        method: "POST",
                        body: form
                    });

                    const result = await response.json();

                    if (!response.ok) {
                        throw new Error(result.detail || "上傳失敗");
                    }

                    status.textContent =
                        "上傳成功！檔案：" + result.filename;
                } catch (error) {
                    status.textContent = "錯誤：" + error.message;
                }
            }
        </script>
    </body>
    </html>
    """


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    allowed_types = {
        "image/jpeg": ".jpg",
        "image/png": ".png"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="目前只支援 JPG 和 PNG"
        )

    content = await file.read(MAX_UPLOAD_SIZE + 1)

    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="圖片不可超過 20 MB"
        )

    import cv2
    import numpy as np

    image_array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="無法辨識圖片內容"
        )

    filename = uuid4().hex + allowed_types[file.content_type]
    save_path = UPLOAD_DIR / filename
    save_path.write_bytes(content)

    print(f"[+] 手機上傳成功：{save_path}")

    return {
        "success": True,
        "filename": filename,
        "width": image.shape[1],
        "height": image.shape[0]
    }
