# inventarios_print_v3_assoc.py
# Inventarios Print (A3 300dpi)
# - TIFF con transparencia
# - sin superposición usando máscara de ocupación (alpha)
# - Exporta PNG + PDF + Preview JPG
# - Guarda JSON (serialización del print: seed + receta completa)
# - UI: orientación, cantidad, modo selección (Random / DB Tags), tags, db file, Associations
#
# DB Tags (v3):
# - Usa tags.name + tag_aliases.alias
# - Scoring por asset (flexible; NO exige todos)
# - Asociaciones por co-ocurrencia (opcional) para agrandar pool cuando queda chico
# - Cuando un objeto se repite: rota fuerte + chance de mirror (espejo) para que NO parezca clon
#
# Requisitos:
#   pip install pillow numpy
#
# Uso:
#   python inventarios_print_v3_assoc.py

import os
import re
import time
import json
import random
import sqlite3
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image, ImageOps
import tkinter as tk
from tkinter import filedialog, messagebox


# =========================
# Config
# =========================

@dataclass
class Config:
    dpi: int = 300
    orientation: str = "portrait"
    margin_mm: float = 12.0
    background: Tuple[int, int, int, int] = (255, 255, 255, 255)

    target_count: int = 45

    rotate_min_deg: float = -10.0
    rotate_max_deg: float = 10.0

    # Tope de agrandado para evitar pixelado (1.0 = nunca agranda sobre tamaño original)
    max_upscale: float = 1.0

    # Perfil de escala (se ajusta según cantidad)
    small_scale_range: Tuple[float, float] = (0.03, 0.08)
    medium_scale_range: Tuple[float, float] = (0.08, 0.14)
    medium_probability: float = 0.18

    padding_px_lowres: int = 2
    occupancy_downscale: int = 4
    attempts_per_object: int = 250
    max_seconds: int = 25

    export_png: bool = True
    export_pdf: bool = True

    # Preview (baja)
    export_preview: bool = True
    preview_max_side_px: int = 1600
    preview_quality: int = 85  # JPG

    # JSON
    export_json: bool = True

    # Asociaciones (co-ocurrencia)
    assoc_enabled_default: bool = True
    assoc_top_n: int = 12             # cuántos tags asociados agregar
    assoc_generic_top_n: int = 60     # tags “genéricos” por frecuencia global a ignorar
    assoc_expand_if_pool_lt_mult: int = 3  # si pool < target_count * mult, expandir


CFG = Config()


# =========================
# Helpers
# =========================

def a3_pixels(dpi: int, orientation: str) -> Tuple[int, int]:
    w_mm, h_mm = 297.0, 420.0
    if orientation.lower().startswith("land"):
        w_mm, h_mm = h_mm, w_mm
    w_px = int(round((w_mm / 25.4) * dpi))
    h_px = int(round((h_mm / 25.4) * dpi))
    return w_px, h_px


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round((mm / 25.4) * dpi))


def list_tiffs(folder: str) -> List[str]:
    exts = {".tif", ".tiff"}
    files = []
    for fn in os.listdir(folder):
        p = os.path.join(folder, fn)
        if os.path.isfile(p) and os.path.splitext(fn.lower())[1] in exts:
            files.append(p)
    files.sort()
    return files


def load_rgba(path: str) -> Image.Image:
    im = Image.open(path)
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    return im


def alpha_mask(im_rgba: Image.Image) -> Image.Image:
    return im_rgba.split()[-1]


def tokenize(text: str) -> List[str]:
    text = text.strip().lower()
    if not text:
        return []
    parts = re.split(r"[^a-z0-9áéíóúñü]+", text, flags=re.IGNORECASE)
    return [p for p in parts if p]


def get_scale_profile_by_count(n: int) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    if n <= 1:
        return (0.22, 0.38), (0.38, 0.55), 0.55
    if n == 2:
        return (0.18, 0.30), (0.30, 0.42), 0.50
    if n == 3:
        return (0.14, 0.24), (0.24, 0.34), 0.45
    if n <= 5:
        return (0.10, 0.18), (0.18, 0.28), 0.38
    if n <= 10:
        return (0.08, 0.14), (0.14, 0.22), 0.30
    if n <= 25:
        return (0.06, 0.12), (0.12, 0.20), 0.22
    return (0.03, 0.08), (0.08, 0.14), 0.18


def choose_scale(small_range: Tuple[float, float], medium_range: Tuple[float, float], medium_prob: float) -> float:
    if random.random() < medium_prob:
        return random.uniform(*medium_range)
    return random.uniform(*small_range)


def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    h, w = mask.shape
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        y0 = max(0, -dy)
        y1 = min(h, h - dy)
        for dx in range(-radius, radius + 1):
            x0 = max(0, -dx)
            x1 = min(w, w - dx)
            out[y0:y1, x0:x1] |= mask[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return out


def build_lowres_mask_from_alpha(img_rgba: Image.Image, downscale: int, padding: int) -> np.ndarray:
    a = np.array(alpha_mask(img_rgba), dtype=np.uint8)
    occ = a > 0
    if downscale > 1:
        H, W = occ.shape
        h2 = max(1, H // downscale)
        w2 = max(1, W // downscale)
        occ = occ[: h2 * downscale, : w2 * downscale]
        occ = occ.reshape(h2, downscale, w2, downscale).any(axis=(1, 3))
    if padding > 0:
        occ = dilate_bool(occ, radius=padding)
    return occ


def try_place(canvas_size: Tuple[int, int],
              canvas_occ: np.ndarray,
              obj_occ: np.ndarray,
              cfg: Config,
              margin_px: int) -> Tuple[Optional[Tuple[int, int]], int]:
    W, H = canvas_size
    ds = cfg.occupancy_downscale
    oh, ow = obj_occ.shape

    x_min = margin_px // ds
    y_min = margin_px // ds
    x_max = (W - margin_px) // ds - ow
    y_max = (H - margin_px) // ds - oh
    if x_max <= x_min or y_max <= y_min:
        return None, 0

    for i in range(cfg.attempts_per_object):
        x_lr = random.randint(x_min, x_max)
        y_lr = random.randint(y_min, y_max)
        region = canvas_occ[y_lr:y_lr + oh, x_lr:x_lr + ow]
        if region.shape != obj_occ.shape:
            continue
        if np.any(region & obj_occ):
            continue
        return (x_lr * ds, y_lr * ds), (i + 1)

    return None, cfg.attempts_per_object


def stamp_occ(canvas_occ: np.ndarray, obj_occ: np.ndarray, x: int, y: int, ds: int) -> None:
    x_lr = x // ds
    y_lr = y // ds
    oh, ow = obj_occ.shape
    canvas_occ[y_lr:y_lr + oh, x_lr:x_lr + ow] |= obj_occ


def make_preview_jpg(rgb_img: Image.Image, max_side: int) -> Image.Image:
    w, h = rgb_img.size
    m = max(w, h)
    if m <= max_side:
        return rgb_img
    scale = max_side / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return rgb_img.resize((nw, nh), Image.LANCZOS)


# =========================
# DB: helpers (tags + aliases + asociaciones)
# =========================

def _db_connect(db_path: str) -> sqlite3.Connection:
    if not os.path.isfile(db_path):
        raise RuntimeError("DB no encontrada.")
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def _db_generic_tags(con: sqlite3.Connection, top_n: int) -> set:
    # tags más frecuentes globalmente (genéricos) para NO usar como asociaciones
    q = """
    SELECT t.name, COUNT(*) AS n
    FROM asset_tags at
    JOIN tags t ON t.id = at.tag_id
    GROUP BY t.id
    ORDER BY n DESC
    LIMIT ?
    """
    rows = con.execute(q, (top_n,)).fetchall()
    return {r["name"] for r in rows}


def db_fetch_assets_by_tags(db_path: str,
                           tags: List[str],
                           assoc_enabled: bool,
                           target_count: int,
                           cfg: Config) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Devuelve:
      - paths ordenados por score desc
      - expanded_tags (asociaciones agregadas)
      - meta: pool_size_before/after
    """
    wanted = [t.strip().lower() for t in tags if t.strip()]
    if not wanted:
        raise RuntimeError("Modo DB Tags requiere al menos 1 tag.")

    con = _db_connect(db_path)
    try:
        # Query base (solo wanted, peso 2)
        def scored_paths_for(tag_weight_pairs: List[Tuple[str, int]]) -> List[sqlite3.Row]:
            values_sql = ",".join(["(?,?)"] * len(tag_weight_pairs))
            params: List[Any] = []
            for t, w in tag_weight_pairs:
                params.extend([t, w])

            q = f"""
            WITH wanted(tag, w) AS (
                VALUES {values_sql}
            ),
            resolved AS (
                SELECT t.id AS tag_id, w.w AS w
                FROM tags t
                JOIN wanted w ON w.tag = t.name
                UNION
                SELECT a.tag_id AS tag_id, w.w AS w
                FROM tag_aliases a
                JOIN wanted w ON w.tag = a.alias
            ),
            scored AS (
                SELECT at.asset_id, SUM(r.w) AS score
                FROM asset_tags at
                JOIN resolved r ON r.tag_id = at.tag_id
                GROUP BY at.asset_id
            )
            SELECT a.path, s.score
            FROM scored s
            JOIN assets a ON a.id = s.asset_id
            ORDER BY s.score DESC
            """
            return con.execute(q, params).fetchall()

        base_rows = scored_paths_for([(t, 2) for t in wanted])
        base_paths = [r["path"] for r in base_rows if r["path"] and os.path.isfile(r["path"])]
        pool_before = len(base_paths)

        expanded: List[str] = []

        # ¿expandimos?
        min_pool = max(1, int(target_count) * int(cfg.assoc_expand_if_pool_lt_mult))
        if assoc_enabled and pool_before < min_pool and base_rows:
            generic = _db_generic_tags(con, cfg.assoc_generic_top_n)
            wanted_set = set(wanted)

            # Tomamos los assets más cercanos al query base y vemos tags co-ocurrentes.
            # (No hace falta meter placeholders masivos: usamos subquery con LIMIT)
            # Usamos los top M assets base para co-ocurrencia.
            top_assets_n = min(200, len(base_rows))
            top_asset_ids_q = f"""
            WITH wanted(tag, w) AS (
                VALUES {",".join(["(?,?)"] * len(wanted))}
            ),
            resolved AS (
                SELECT t.id AS tag_id
                FROM tags t JOIN wanted w ON w.tag = t.name
                UNION
                SELECT a.tag_id
                FROM tag_aliases a JOIN wanted w ON w.tag = a.alias
            ),
            scored AS (
                SELECT at.asset_id, COUNT(*) AS score
                FROM asset_tags at
                JOIN resolved r ON r.tag_id = at.tag_id
                GROUP BY at.asset_id
                ORDER BY score DESC
                LIMIT ?
            )
            SELECT asset_id FROM scored
            """
            params = []
            for t in wanted:
                params.extend([t, 1])
            params.append(top_assets_n)

            asset_ids = [r["asset_id"] for r in con.execute(top_asset_ids_q, params).fetchall()]
            if asset_ids:
                ph = ",".join(["?"] * len(asset_ids))
                coq = f"""
                SELECT t.name, COUNT(*) AS n
                FROM asset_tags at
                JOIN tags t ON t.id = at.tag_id
                WHERE at.asset_id IN ({ph})
                GROUP BY t.id
                ORDER BY n DESC
                LIMIT 300
                """
                co_rows = con.execute(coq, asset_ids).fetchall()

                # Elegimos asociaciones:
                # - no en wanted
                # - no genéricas
                # - largo mínimo para evitar basura
                for r in co_rows:
                    name = (r["name"] or "").strip().lower()
                    if not name:
                        continue
                    if name in wanted_set:
                        continue
                    if name in generic:
                        continue
                    if len(name) < 3:
                        continue
                    expanded.append(name)
                    if len(expanded) >= cfg.assoc_top_n:
                        break

        # Si hay asociaciones: re-score con pesos (wanted=2, assoc=1)
        if expanded:
            pairs = [(t, 2) for t in wanted] + [(t, 1) for t in expanded]
            rows2 = scored_paths_for(pairs)
            paths2 = [r["path"] for r in rows2 if r["path"] and os.path.isfile(r["path"])]
            pool_after = len(paths2)
            return paths2, expanded, {"pool_before": pool_before, "pool_after": pool_after}

        return base_paths, [], {"pool_before": pool_before, "pool_after": pool_before}

    finally:
        con.close()


def build_pool_random(files: List[str]) -> List[str]:
    pool = files.copy()
    random.shuffle(pool)
    return pool


# =========================
# Generación
# =========================

def generate_print(input_folder: str,
                   output_folder: str,
                   cfg: Config,
                   selection_mode: str,
                   db_path: str,
                   db_tags_text: str,
                   assoc_enabled: bool) -> Dict[str, Any]:
    files = list_tiffs(input_folder)
    if not files:
        raise RuntimeError("No encontré TIFFs en la carpeta seleccionada.")

    # Seed (reproducible)
    seed = int(time.time() * 1000)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    W, H = a3_pixels(cfg.dpi, cfg.orientation)
    margin_px = mm_to_px(cfg.margin_mm, cfg.dpi)
    usable_h = H - 2 * margin_px

    small_range, medium_range, medium_prob = get_scale_profile_by_count(int(cfg.target_count))

    canvas = Image.new("RGBA", (W, H), cfg.background)

    ds = cfg.occupancy_downscale
    canvas_occ = np.zeros((max(1, H // ds), max(1, W // ds)), dtype=bool)

    mode = selection_mode.strip().lower()
    expanded_tags: List[str] = []
    pool_meta: Dict[str, int] = {"pool_before": 0, "pool_after": 0}

    if mode == "db":
        tags = tokenize(db_tags_text)
        pool, expanded_tags, pool_meta = db_fetch_assets_by_tags(
            db_path=db_path,
            tags=tags,
            assoc_enabled=assoc_enabled,
            target_count=int(cfg.target_count),
            cfg=cfg,
        )
        if not pool:
            raise RuntimeError("DB Tags: no encontré assets con esos tags (ni aliases).")
        random.shuffle(pool)
    else:
        pool = build_pool_random(files)

    pool_idx = 0
    start = time.time()
    count_target = int(cfg.target_count)
    placed_count = 0
    tries_without_success = 0

    items: List[Dict[str, Any]] = []
    use_count: Dict[str, int] = {}

    while placed_count < count_target and (time.time() - start) < cfg.max_seconds:
        if pool_idx >= len(pool):
            pool_idx = 0
            random.shuffle(pool)

        path = pool[pool_idx]
        pool_idx += 1

        try:
            im = load_rgba(path)
        except Exception:
            continue

        # Tamaño relativo (siempre varía)
        rel = choose_scale(small_range, medium_range, medium_prob)
        target_h = max(80, int(rel * usable_h))

        w0, h0 = im.size
        max_h = int(h0 * cfg.max_upscale)
        target_h = min(target_h, max_h)

        scale = target_h / max(1, h0)
        new_w = max(1, int(round(w0 * scale)))
        new_h = max(1, int(round(h0 * scale)))

        im2 = im.resize((new_w, new_h), resample=Image.LANCZOS)

        # Variación extra si el asset se repite
        use_count[path] = use_count.get(path, 0) + 1
        rep = use_count[path]

        mirrored = False
        if rep >= 2:
            deg = random.uniform(-35.0, 35.0)
            if random.random() < 0.45:
                im2 = ImageOps.mirror(im2)
                mirrored = True
        else:
            deg = random.uniform(cfg.rotate_min_deg, cfg.rotate_max_deg)

        im3 = im2.rotate(deg, resample=Image.BICUBIC, expand=True)

        # Crop por alpha real
        a = alpha_mask(im3)
        bbox = a.getbbox()
        if not bbox:
            continue
        im3 = im3.crop(bbox)

        # Máscara de ocupación
        obj_occ = build_lowres_mask_from_alpha(im3, cfg.occupancy_downscale, cfg.padding_px_lowres)
        pos, attempts_used = try_place((W, H), canvas_occ, obj_occ, cfg, margin_px)

        if pos is None:
            tries_without_success += 1
            if tries_without_success > 12:
                count_target = max(placed_count, count_target - 1)
                tries_without_success = 0
            continue

        x, y = pos
        canvas.alpha_composite(im3, (x, y))
        stamp_occ(canvas_occ, obj_occ, x, y, ds)

        placed_count += 1
        tries_without_success = 0

        items.append({
            "source_path": os.path.abspath(path),
            "file_name": os.path.basename(path),
            "pos_x": int(x),
            "pos_y": int(y),
            "w": int(im3.size[0]),
            "h": int(im3.size[1]),
            "rotation_deg": float(deg),
            "target_h": int(target_h),
            "scale": float(scale),
            "attempts_used": int(attempts_used),
            "bbox_crop": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            "repeat_index": int(rep),
            "mirrored": bool(mirrored),
        })

    os.makedirs(output_folder, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = f"inventarios_A3_{cfg.orientation}_{cfg.dpi}dpi_{stamp}_n{placed_count}"

    png_path = os.path.join(output_folder, base + ".png")
    pdf_path = os.path.join(output_folder, base + ".pdf")
    preview_path = os.path.join(output_folder, base + "_preview.jpg")
    json_path = os.path.join(output_folder, base + ".json")

    out_rgb = canvas.convert("RGB")

    if cfg.export_png:
        out_rgb.save(png_path, "PNG", optimize=True)
    if cfg.export_pdf:
        out_rgb.save(pdf_path, "PDF", resolution=cfg.dpi)

    if cfg.export_preview:
        prev = make_preview_jpg(out_rgb, cfg.preview_max_side_px)
        prev.save(preview_path, "JPEG", quality=cfg.preview_quality, optimize=True, progressive=True)

    payload = {
        "print_id": base,
        "timestamp": stamp,
        "seed": seed,
        "config": asdict(cfg),
        "selection": {
            "mode": "db" if mode == "db" else "random",
            "db_path": os.path.abspath(db_path) if mode == "db" else None,
            "db_tags": tokenize(db_tags_text) if mode == "db" else [],
            "associations_enabled": bool(assoc_enabled) if mode == "db" else False,
            "expanded_tags": expanded_tags if mode == "db" else [],
            "pool_meta": pool_meta if mode == "db" else {},
        },
        "canvas": {"w": W, "h": H, "margin_px": margin_px},
        "placed_count": placed_count,
        "target_count": int(cfg.target_count),
        "scale_profile": {
            "small_range": list(small_range),
            "medium_range": list(medium_range),
            "medium_probability": medium_prob,
        },
        "outputs": {
            "png": os.path.abspath(png_path) if cfg.export_png else None,
            "pdf": os.path.abspath(pdf_path) if cfg.export_pdf else None,
            "preview": os.path.abspath(preview_path) if cfg.export_preview else None,
            "json": os.path.abspath(json_path) if cfg.export_json else None,
        },
        "items": items,
    }

    if cfg.export_json:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


# =========================
# UI
# =========================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Inventarios Print")

        self.orientation_var = tk.StringVar(value="Portrait")
        self.count_var = tk.StringVar(value=str(CFG.target_count))

        self.mode_var = tk.StringVar(value="Random")  # Random / DB Tags
        self.db_path_var = tk.StringVar(value="")
        self.db_tags_var = tk.StringVar(value="")

        self.assoc_var = tk.BooleanVar(value=CFG.assoc_enabled_default)

        frm = tk.Frame(self, padx=12, pady=12)
        frm.pack(fill="both", expand=True)

        tk.Label(frm, text="Orientation").grid(row=0, column=0, sticky="w")
        tk.OptionMenu(frm, self.orientation_var, "Portrait", "Landscape").grid(row=0, column=1, sticky="ew")

        tk.Label(frm, text="Objects (count)").grid(row=1, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.count_var).grid(row=1, column=1, sticky="ew")

        tk.Label(frm, text="Selection mode").grid(row=2, column=0, sticky="w")
        tk.OptionMenu(frm, self.mode_var, "Random", "DB Tags").grid(row=2, column=1, sticky="ew")

        tk.Label(frm, text="DB file (sqlite)").grid(row=3, column=0, sticky="w")
        db_row = tk.Frame(frm)
        db_row.grid(row=3, column=1, sticky="ew")
        tk.Entry(db_row, textvariable=self.db_path_var).pack(side="left", fill="x", expand=True)
        tk.Button(db_row, text="Browse", command=self.browse_db).pack(side="left", padx=(6, 0))

        tk.Label(frm, text="DB tags (comma/space)").grid(row=4, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.db_tags_var).grid(row=4, column=1, sticky="ew")

        # Associations
        tk.Checkbutton(frm, text="Associations (expand pool)", variable=self.assoc_var).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

        tk.Button(frm, text="Generate", command=self.on_generate).grid(
            row=6, column=0, columnspan=2, pady=(10, 0), sticky="ew"
        )

        frm.grid_columnconfigure(1, weight=1)

    def browse_db(self):
        p = filedialog.askopenfilename(
            title="Elegí la DB SQLite",
            filetypes=[("SQLite DB", "*.sqlite *.db"), ("All files", "*.*")]
        )
        if p:
            self.db_path_var.set(p)

    def pick_folder(self, title: str) -> Optional[str]:
        return filedialog.askdirectory(title=title) or None

    def on_generate(self):
        try:
            n = int(self.count_var.get().strip())
            if n <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Error", "Objects (count) debe ser un número entero mayor que 0.")
            return

        ori = self.orientation_var.get().strip().lower()
        CFG.target_count = n
        CFG.orientation = "landscape" if ori.startswith("land") else "portrait"

        mode_ui = self.mode_var.get().strip().lower()
        mode = "db" if "db" in mode_ui else "random"

        if mode == "db":
            if not self.db_path_var.get().strip():
                messagebox.showerror("Error", "Modo DB Tags: elegí una DB sqlite.")
                return
            if not self.db_tags_var.get().strip():
                messagebox.showerror("Error", "Modo DB Tags: escribí al menos 1 tag.")
                return

        in_dir = self.pick_folder("Elegí la carpeta con los TIFF (transparencia)")
        if not in_dir:
            return
        out_dir = self.pick_folder("Elegí la carpeta de salida (PNG + PDF + Preview + JSON)")
        if not out_dir:
            return

        try:
            payload = generate_print(
                in_dir,
                out_dir,
                CFG,
                selection_mode=mode,
                db_path=self.db_path_var.get().strip(),
                db_tags_text=self.db_tags_var.get().strip(),
                assoc_enabled=bool(self.assoc_var.get()),
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        outs = payload.get("outputs", {})
        sel = payload.get("selection", {})
        expanded = sel.get("expanded_tags", []) or []
        pool_meta = sel.get("pool_meta", {}) or {}

        msg = "Listo.\n"
        msg += f"\nPreview: {outs.get('preview')}"
        msg += f"\nJSON: {outs.get('json')}"
        msg += f"\nPNG: {outs.get('png')}"
        msg += f"\nPDF: {outs.get('pdf')}"
        msg += f"\n\nPlaced: {payload.get('placed_count')} / {payload.get('target_count')}"

        if sel.get("mode") == "db":
            msg += f"\n\nPool: {pool_meta.get('pool_before')} → {pool_meta.get('pool_after')}"
            if expanded:
                msg += f"\nExpanded tags: {', '.join(expanded[:20])}"

        messagebox.showinfo("Inventarios Print", msg)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
