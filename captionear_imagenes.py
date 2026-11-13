"""
Genera una descripción en español de cada imagen de imagenes/ con un
modelo de visión de Ollama (gemma3:12b por defecto) y la guarda en captions.json.

Esas descripciones se indexan como chunks de texto (ver `imagenes_dir` en los YAML),
así que el RAG puede recuperar imágenes con preguntas en texto.

Es re-ejecutable y reanudable: salta las imágenes que ya tienen descripción y
guarda después de cada una. Usa la GPU (Ollama).

Uso:
    python captionear_imagenes.py
    python captionear_imagenes.py --limite 5          # prueba rápida
    python captionear_imagenes.py --modelo llama3.2-vision
"""

import argparse
import base64
import json
import os
import sys
import time

import requests

PROMPT = """Describe esta imagen para un buscador de un corpus sobre astronomía y exploración espacial.
La imagen aparece en el artículo de Wikipedia «{articulo}».{desc}

Indica en español, en 3 a 5 oraciones: qué tipo de imagen es (fotografía, diagrama, mapa, ilustración o gráfica), qué objeto o escena muestra, colores y estructuras visibles, y cualquier texto, etiqueta o cifra legible.
No inventes datos que no se vean en la imagen ni se deduzcan del artículo. Responde solo con la descripción."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="imagenes")
    ap.add_argument("--modelo", default="gemma3:12b")
    ap.add_argument("--url", default="http://localhost:11434/api/generate")
    ap.add_argument("--limite", type=int, default=0, help="máx. imágenes a describir (0 = todas)")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    with open(os.path.join(args.dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)

    cap_path = os.path.join(args.dir, "captions.json")
    captions: dict[str, dict] = {}
    if os.path.exists(cap_path):
        with open(cap_path, encoding="utf-8") as f:
            captions = json.load(f)

    pendientes = [m for m in manifest if m["archivo"] not in captions]
    if args.limite:
        pendientes = pendientes[:args.limite]
    print(f"{len(captions)} ya descritas, {len(pendientes)} por describir con {args.modelo}")

    for n, m in enumerate(pendientes, 1):
        with open(os.path.join(args.dir, m["archivo"]), "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        desc = f" Descripción del archivo en Commons: {m['descripcion']}" if m.get("descripcion") else ""
        payload = {
            "model": args.modelo,
            "prompt": PROMPT.format(articulo=m["articulo"], desc=desc),
            "images": [b64],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 350},
        }
        t0 = time.time()
        try:
            r = requests.post(args.url, json=payload, timeout=args.timeout)
            r.raise_for_status()
            texto = r.json().get("response", "").strip()
        except Exception as e:
            print(f"[{n}/{len(pendientes)}] ERROR {m['archivo']}: {e}")
            continue
        if not texto:
            print(f"[{n}/{len(pendientes)}] vacía: {m['archivo']}")
            continue

        captions[m["archivo"]] = {"articulo": m["articulo"], "caption": texto, "modelo": args.modelo}
        with open(cap_path, "w", encoding="utf-8") as f:
            json.dump(captions, f, ensure_ascii=False, indent=1)
        print(f"[{n}/{len(pendientes)}] {time.time() - t0:.1f}s {m['archivo']}\n    {texto[:160]}...")

    print(f"\nListo: {len(captions)} descripciones en {cap_path}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
