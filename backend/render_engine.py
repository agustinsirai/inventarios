import os
import uuid
import shutil
from datetime import datetime
from typing import Literal, Dict, Any

from backend.config import TIFF_DIR, DB_PATH, STORAGE_PREVIEWS, STORAGE_JSON, STORAGE_TMP

# Motor: copiamos tu v3_assoc acá para no tocar nada del root
from backend import engine_v3_assoc as eng


def _ensure_dirs():
    os.makedirs(STORAGE_PREVIEWS, exist_ok=True)
    os.makedirs(STORAGE_JSON, exist_ok=True)
    os.makedirs(STORAGE_TMP, exist_ok=True)


def generate_preview_and_json(
    count: int,
    orientation: Literal["portrait", "landscape"],
) -> Dict[str, Any]:
    """
    Genera:
      - preview.jpg (1600px lado max, JPG)
      - print.json  (certificado/receta completa)
    NO genera PDF ni PNG alta en esta etapa.
    """
    _ensure_dirs()
    t0 = datetime.now()
    print("[timing] start", t0.isoformat())

    print_id = uuid.uuid4().hex[:12]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(STORAGE_TMP, f"{stamp}_{print_id}")
    os.makedirs(out_dir, exist_ok=True)

    cfg = eng.Config()
    cfg.orientation = "portrait" if orientation == "portrait" else "landscape"
    cfg.target_count = int(count)

    # Solo preview + json (fase 1)
    cfg.export_pdf = False
    cfg.export_png = False
    cfg.export_preview = True
    cfg.export_json = True

    # (preview settings ya vienen en el script; podés tocar acá si querés)
    # cfg.preview_max_side_px = 1600
    # cfg.preview_quality = 85
    t1 = datetime.now()
    print("[timing] before eng.generate_print", (t1 - t0).total_seconds(), "s")
    payload = eng.generate_print(
        input_folder=TIFF_DIR,
        output_folder=out_dir,
        cfg=cfg,
        selection_mode="random",   # random puro, sin tags
        db_path=DB_PATH,           # no se usa en random
        db_tags_text="",
        assoc_enabled=False,
    )
    t2 = datetime.now()
    print("[timing] after eng.generate_print", (t2 - t1).total_seconds(), "s")
    print("[timing] total so far", (t2 - t0).total_seconds(), "s")
    prev_src = payload["outputs"]["preview"]
    json_src = payload["outputs"]["json"]

    if not prev_src or not os.path.exists(prev_src):
        raise RuntimeError("No se generó preview.jpg")
    if not json_src or not os.path.exists(json_src):
        raise RuntimeError("No se generó print.json")

    preview_name = f"{payload['print_id']}.jpg"
    json_name = f"{payload['print_id']}.json"

    prev_dst = os.path.join(STORAGE_PREVIEWS, preview_name)
    json_dst = os.path.join(STORAGE_JSON, json_name)

    shutil.copy2(prev_src, prev_dst)
    shutil.copy2(json_src, json_dst)

    # limpieza tmp (deja solo lo guardado)
    try:
        shutil.rmtree(out_dir, ignore_errors=True)
    except Exception:
        pass

    return {
        "print_id": payload["print_id"],
        "preview_file": preview_name,
        "json_file": json_name,
    }
