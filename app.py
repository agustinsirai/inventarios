from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os

from backend.config import STORAGE_PREVIEWS, STORAGE_JSON, ENABLE_DOWNLOAD
from backend.render_engine import generate_preview_and_json

app = FastAPI()

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/previews", StaticFiles(directory=STORAGE_PREVIEWS), name="previews")
app.mount("/json", StaticFiles(directory=STORAGE_JSON), name="json")


@app.get("/", response_class=HTMLResponse)
def home():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/config")
def api_config():
    return {"enable_download": bool(ENABLE_DOWNLOAD)}


@app.post("/api/generate")
def api_generate(payload: dict):
    count = int(payload.get("count", 12))
    orientation = payload.get("orientation", "portrait")
    if orientation not in ("portrait", "landscape"):
        orientation = "portrait"
    if count < 1:
        count = 1
    if count > 120:
        count = 120

    out = generate_preview_and_json(count=count, orientation=orientation)
    return JSONResponse(
        {
            "print_id": out["print_id"],
            "preview_url": f"/previews/{out['preview_file']}",
            "json_url": f"/json/{out['json_file']}",
        }
    )
