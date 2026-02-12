import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # ...\web_version
TIFF_DIR = os.path.join(BASE_DIR, "assets_tiff")
DB_PATH  = r"D:\Users\Agustin\Desktop\TODAS LAS COSAS\GENERADOR DE PRINTS\inventarios.sqlite"

STORAGE_PREVIEWS = r"D:\Users\Agustin\Desktop\TODAS LAS COSAS\GENERADOR DE PRINTS\web_version\storage\previews"
STORAGE_JSON     = r"D:\Users\Agustin\Desktop\TODAS LAS COSAS\GENERADOR DE PRINTS\web_version\storage\json"
STORAGE_TMP      = r"D:\Users\Agustin\Desktop\TODAS LAS COSAS\GENERADOR DE PRINTS\web_version\storage\_tmp"

# Fase 1: permitir descarga (solo desde fullscreen overlay)
ENABLE_DOWNLOAD = True

# Fondo video (opcional)
VIDEO_HORZ = "static/video/01NM_1.mp4"
VIDEO_VERT = "static/video/home-vertical.mp4"
