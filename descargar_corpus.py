"""
Descarga el corpus de Astronomía (Sistema Solar y exploración espacial)
desde Wikipedia en español a data/.

Solo descarga texto: no carga modelos ni toca GPU/Ollama, así que es ligero.
Es seguro re-ejecutarlo: los artículos ya descargados se saltan (cache).

Uso:
    python descargar_corpus.py
    python descargar_corpus.py --out data --pausa 0.5
"""

import argparse
import os
import sys
import time

import wikipediaapi

ARTICULOS = [
    "Sistema Solar",
    "Sol",
    "Mercurio (planeta)",
    "Venus (planeta)",
    "Tierra",
    "Luna",
    "Marte (planeta)",
    "Júpiter (planeta)",
    "Saturno (planeta)",
    "Urano (planeta)",
    "Neptuno (planeta)",
    "Plutón (planeta enano)",
    "Cinturón de asteroides",
    "Voyager 2",
    "Apolo 11",
    "Estación Espacial Internacional",
    "Telescopio espacial Hubble",
    "Perseverance",
    "Curiosity",
    "Europa (satélite)",
    "Titán (satélite)",
]


def nombre_archivo(titulo: str) -> str:
    """Mismo criterio que rag_ollama.py (el título se recupera desde el nombre del archivo)."""
    return titulo.replace(" ", "_").replace("/", "-") + ".txt"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default="data", help="carpeta de salida")
    ap.add_argument("--pausa", type=float, default=0.5, help="segundos entre peticiones")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    wiki = wikipediaapi.Wikipedia(language="es", user_agent="RAG-Baseline-Tesis/1.0")

    ok, cache, faltan = 0, 0, []
    for titulo in ARTICULOS:
        path = os.path.join(args.out, nombre_archivo(titulo))
        if os.path.exists(path):
            print(f"[CACHE] {titulo}")
            cache += 1
            continue

        try:
            pagina = wiki.page(titulo)
            if not pagina.exists():
                print(f"[SKIP]  No encontrado: {titulo}")
                faltan.append(titulo)
                continue
            texto = pagina.text
        except Exception as e:  # red caída, timeout, etc.
            print(f"[ERROR] {titulo}: {e}")
            faltan.append(titulo)
            continue

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {pagina.title}\n\n{texto}")
        print(f"[OK]    {titulo} ({len(texto):,} chars)")
        ok += 1
        time.sleep(args.pausa)

    print(f"\nListo: {ok} nuevos, {cache} en cache, {len(faltan)} faltantes -> {args.out}/")
    if faltan:
        print("Faltantes (revisa el título exacto en Wikipedia):")
        for t in faltan:
            print(f"  - {t}")
    return 0 if not faltan else 1


if __name__ == "__main__":
    sys.exit(main())
