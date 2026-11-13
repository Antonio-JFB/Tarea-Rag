"""
Descarga las imágenes principales de cada artículo de data/ desde
Wikipedia en español a imagenes/ y escribe manifest.json.

Por artículo toma la imagen principal (pageimages) + las más grandes de las que
aparecen en la página, filtrando íconos, banderas, logos y SVG. Solo red + disco:
no usa GPU ni Ollama. Es re-ejecutable (salta lo ya descargado).

Uso:
    python descargar_imagenes.py
    python descargar_imagenes.py --por-articulo 4 --ancho 800
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import requests

API = "https://es.wikipedia.org/w/api.php"
UA = {"User-Agent": "RAG-Baseline-Tesis/1.0 (proyecto academico)"}

# Nombres de archivo que casi nunca son contenido útil (íconos, plantillas, logos).
EXCLUIR = re.compile(
    r"(icon|logo|flag|bandera|symbol|question_book|commons-logo|wiki|ambox|"
    r"edit-clear|increase|decrease|portal|folder|padlock|speaker|red_pog|"
    r"star|crystal|nuvola|disambig|stub|lock)",
    re.IGNORECASE,
)


def api(params: dict) -> dict:
    p = {"format": "json", "formatversion": "2", **params}
    for intento in range(6):
        time.sleep(0.5)  # evita el 429 por rafagas de peticiones
        r = requests.get(API, params=p, headers=UA, timeout=30)
        if r.status_code == 429:
            time.sleep(5 * (intento + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Wikipedia API: demasiados 429")


def titulo_articulo(path: str) -> str:
    return os.path.basename(path).replace(".txt", "").replace("_", " ")


def imagen_principal(titulo: str, ancho: int) -> str | None:
    data = api({"action": "query", "titles": titulo, "prop": "pageimages",
                "piprop": "name", "pilicense": "any"})
    pages = data.get("query", {}).get("pages", [])
    if pages and pages[0].get("pageimage"):
        return "File:" + pages[0]["pageimage"]
    return None


def archivos_de_pagina(titulo: str) -> list[str]:
    data = api({"action": "query", "titles": titulo, "prop": "images", "imlimit": 100})
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return []
    return [i["title"] for i in pages[0].get("images", [])]


def info_archivos(files: list[str], ancho: int) -> list[dict]:
    """imageinfo por lotes de 40: url del thumbnail, tamaño original, mime y descripción."""
    salida = []
    for i in range(0, len(files), 40):
        lote = files[i:i + 40]
        data = api({"action": "query", "titles": "|".join(lote), "prop": "imageinfo",
                    "iiprop": "url|size|mime|extmetadata", "iiurlwidth": ancho})
        for p in data.get("query", {}).get("pages", []):
            ii = (p.get("imageinfo") or [None])[0]
            if not ii:
                continue
            meta = ii.get("extmetadata", {})
            desc = re.sub(r"<[^>]+>", "", meta.get("ImageDescription", {}).get("value", "")).strip()
            salida.append({
                "file": p["title"], "mime": ii.get("mime", ""),
                "w": ii.get("width", 0), "h": ii.get("height", 0),
                "thumb": ii.get("thumburl") or ii.get("url"),
                "descripcion": desc[:300],
                "licencia": meta.get("LicenseShortName", {}).get("value", ""),
            })
    return salida


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:80]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data")
    ap.add_argument("--out", default="imagenes")
    ap.add_argument("--por-articulo", type=int, default=3)
    ap.add_argument("--ancho", type=int, default=640, help="ancho del thumbnail a descargar")
    ap.add_argument("--min-lado", type=int, default=300, help="descarta imágenes más chicas")
    ap.add_argument("--pausa", type=float, default=0.4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")
    manifest = []
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    ya = {m["file"] for m in manifest}

    nuevos = 0
    for path in sorted(glob.glob(os.path.join(args.corpus, "*.txt"))):
        art = titulo_articulo(path)
        try:
            principal = imagen_principal(art, args.ancho)
            todos = archivos_de_pagina(art)
            candidatos = [f for f in todos if not EXCLUIR.search(f)
                          and f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if principal and principal not in candidatos and not EXCLUIR.search(principal):
                candidatos.insert(0, principal)
            infos = [i for i in info_archivos(candidatos, args.ancho)
                     if i["mime"] in ("image/jpeg", "image/png")
                     and min(i["w"], i["h"]) >= args.min_lado]
        except Exception as e:
            print(f"[ERROR] {art}: {e}")
            continue

        # principal primero; el resto por área descendente
        infos.sort(key=lambda i: (i["file"] != principal, -(i["w"] * i["h"])))
        elegidas = infos[:args.por_articulo]
        print(f"{art}: {len(elegidas)} imagen(es)")

        for i in elegidas:
            if i["file"] in ya:
                continue
            ext = ".png" if i["mime"] == "image/png" else ".jpg"
            nombre = f"{slug(art)}__{slug(i['file'].removeprefix('File:'))}"
            if not nombre.lower().endswith(ext):
                nombre += ext
            destino = os.path.join(args.out, nombre)
            try:
                r = requests.get(i["thumb"], headers=UA, timeout=60)
                r.raise_for_status()
            except Exception as e:
                print(f"   [ERROR] {i['file']}: {e}")
                continue
            with open(destino, "wb") as f:
                f.write(r.content)
            manifest.append({
                "archivo": nombre, "articulo": art, "file": i["file"],
                "descripcion": i["descripcion"], "licencia": i["licencia"],
                "bytes": len(r.content),
            })
            ya.add(i["file"])
            nuevos += 1
            print(f"   [OK] {nombre} ({len(r.content) // 1024} KB)")
            time.sleep(args.pausa)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)

    print(f"\nListo: {nuevos} nuevas, {len(manifest)} en total -> {args.out}/")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
