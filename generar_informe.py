"""
Corre un conjunto fijo de preguntas de demostración contra los 3 retrievers
(cosine, bm25, hybrid) y genera un informe HTML autocontenido para presentar:

  results/resultados.json   datos crudos (respuestas, chunks, tiempos)
  results/informe.html      informe visual (imágenes embebidas, se abre sin servidor)

El LLM (Ollama) responde solo con el retriever híbrido; los 3 retrievers se comparan
en qué recuperan y cuánto tardan. No es una evaluación formal: la columna
"artículo esperado" es una comprobación automática simple.

Uso:
    python generar_informe.py
    python generar_informe.py --sin-llm        # solo retrieval (rápido)
    python generar_informe.py --solo-html      # regenera el HTML desde resultados.json
"""

import argparse
import base64
import html
import json
import os
import re
import sys
import time
from datetime import datetime

from src.config import build_pipeline, load_chunks_for, load_config

# (tipo, pregunta, artículos esperados en el top-5; [] = pregunta trampa)
PREGUNTAS = [
    ("Comparativa", "¿Qué diferencias hay entre Marte y Venus?", ["Marte (planeta)", "Venus (planeta)"]),
    ("Causal", "¿Por qué Plutón ya no se considera un planeta?", ["Plutón (planeta enano)"]),
    ("Factual", "¿Qué es la Gran Mancha Roja de Júpiter?", ["Júpiter (planeta)"]),
    ("Relacional", "¿Qué misión llegó por primera vez a Neptuno?", ["Voyager 2", "Neptuno (planeta)"]),
    ("Multi-hop", "¿Qué diferencias hay entre los rovers Curiosity y Perseverance?", ["Curiosity", "Perseverance"]),
    ("Imagen", "¿Cómo se ve la superficie de Europa?", ["Europa (satélite)"]),
    ("Imagen", "¿Cómo se ve Titán en una fotografía a color real?", ["Titán (satélite)"]),
    ("Imagen", "¿Cómo es el disco de oro que llevan las sondas Voyager?", ["Voyager 2"]),
    ("Trampa (fuera del corpus)", "¿Quién ganó el mundial de fútbol de 1970?", []),
]

RETRIEVERS = {
    "cosine": ("Cosine / FAISS", "configs/cosine.yaml"),
    "bm25": ("BM25", "configs/bm25.yaml"),
    "hybrid": ("Híbrido (RRF)", "configs/hybrid.yaml"),
}


def ejecutar(sin_llm: bool) -> dict:
    pipelines = {}
    for key, (_, cfg_path) in RETRIEVERS.items():
        cfg = load_config(cfg_path)
        p = build_pipeline(cfg)
        chunks = load_chunks_for(cfg, p.chunker)
        p.index(chunks)
        p.retrieve("calentamiento")  # carga el modelo de embeddings antes de medir tiempos
        pipelines[key] = p
        n_img = sum(1 for c in chunks if c.imagen)
        stats = {"chunks_total": len(chunks), "chunks_imagen": n_img, "chunks_texto": len(chunks) - n_img}
        print(f"[{key}] listo ({len(chunks)} chunks)")

    filas = []
    for i, (tipo, pregunta, esperados) in enumerate(PREGUNTAS, 1):
        print(f"\n[{i}/{len(PREGUNTAS)}] {tipo}: {pregunta}")
        fila = {"tipo": tipo, "pregunta": pregunta, "esperados": esperados, "retrievers": {}}
        for key, p in pipelines.items():
            res, t = p.retrieve(pregunta)
            fila["retrievers"][key] = {
                "t_retrieval": t,
                "chunks": [{"fuente": r.chunk.fuente, "score": r.score, "imagen": r.chunk.imagen,
                            "texto": r.chunk.texto} for r in res],
            }
            print(f"   {key:7s} {t * 1000:6.1f} ms  {[r.chunk.fuente for r in res]}")
        if not sin_llm:
            r = pipelines["hybrid"].answer(pregunta)
            fila["respuesta"] = r.generation.respuesta
            fila["t_generacion"] = r.generation.latency_s
            fila["t_total"] = r.t_total
            print(f"   LLM {r.generation.latency_s:.1f}s -> {r.generation.respuesta[:120]!r}")
        filas.append(fila)

    return {"fecha": datetime.now().strftime("%Y-%m-%d %H:%M"), "modelo": "gemma3:12b",
            "stats": stats, "filas": filas, "con_llm": not sin_llm}


# ─────────────────────────── HTML ───────────────────────────

def b64_img(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    ext = "png" if path.lower().endswith(".png") else "jpeg"
    with open(path, "rb") as f:
        return f"data:image/{ext};base64,{base64.b64encode(f.read()).decode()}"


def md_basico(texto: str) -> str:
    """Convierte lo mínimo de markdown que devuelve el LLM (listas con * y **negritas**)."""
    out, en_lista = [], False
    for linea in texto.splitlines():
        m = re.match(r"^\s*[\*\-]\s+(.*)", linea)
        s = html.escape(m.group(1) if m else linea)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        if m:
            if not en_lista:
                out.append("<ul>")
                en_lista = True
            out.append(f"<li>{s}</li>")
        else:
            if en_lista:
                out.append("</ul>")
                en_lista = False
            if s.strip():
                out.append(f"<p>{s}</p>")
    if en_lista:
        out.append("</ul>")
    return "\n".join(out)


def articulo(fuente: str) -> str:
    return fuente.removeprefix("Imagen · ")


def hit(fila: dict, key: str) -> bool | None:
    if not fila["esperados"]:
        return None
    arts = {articulo(c["fuente"]) for c in fila["retrievers"][key]["chunks"]}
    return all(e in arts for e in fila["esperados"])


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#16181d;--mut:#5d6573;--line:#e3e6ec;--acc:#2f5bea;--ok:#14804a;--no:#b42318;--chip:#eef1fb}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#171a21;--ink:#e8eaf0;--mut:#98a1b3;--line:#262b36;--acc:#7d9bff;--ok:#4cc38a;--no:#ff7b72;--chip:#1f2536}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:36px 0 12px}h3{font-size:15px;margin:0 0 8px;color:var(--mut);font-weight:600}
.sub{color:var(--mut);margin:0 0 24px}
.grid{display:grid;gap:12px}.k4{grid-template-columns:repeat(auto-fit,minmax(200px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.kpi .n{font-size:30px;font-weight:700;line-height:1.1}.kpi .l{color:var(--mut);font-size:13px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.chip{display:inline-block;background:var(--chip);color:var(--acc);border-radius:99px;padding:2px 10px;font-size:12px;font-weight:600}
.q{margin-top:20px}.q .t{font-size:18px;font-weight:600;margin:6px 0 12px}
.ans{border-left:3px solid var(--acc);padding:2px 0 2px 14px;margin:8px 0 16px}.ans p{margin:6px 0}.ans ul{margin:6px 0;padding-left:20px}
.cols{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}
.src{font-size:13px;margin:0;padding:0;list-style:none}.src li{padding:3px 0;border-bottom:1px dashed var(--line)}
.src .im{color:var(--acc);font-weight:600}.sc{color:var(--mut);float:right;font-variant-numeric:tabular-nums}
.ok{color:var(--ok);font-weight:700}.no{color:var(--no);font-weight:700}.mut{color:var(--mut)}
.gal{display:flex;gap:12px;flex-wrap:wrap;margin-top:12px}.fig{width:230px}.fig img{width:100%;border-radius:8px;border:1px solid var(--line);display:block}
.fig figcaption{font-size:12px;color:var(--mut);margin-top:4px}
.arch{white-space:pre;font:13px/1.5 ui-monospace,Consolas,monospace;overflow-x:auto}
@media print{body{background:#fff}.card{break-inside:avoid}}
"""


def generar_html(d: dict) -> str:
    filas, st = d["filas"], d["stats"]
    keys = list(RETRIEVERS)

    def media(key):
        ts = [f["retrievers"][key]["t_retrieval"] for f in filas]
        return sum(ts) / len(ts)

    img_q = [f for f in filas if f["tipo"] == "Imagen"]
    n_con_llm = [f for f in filas if "t_generacion" in f]
    kpis = [
        (str(st["chunks_texto"]), "chunks de texto (21 artículos de Wikipedia)"),
        (str(st["chunks_imagen"]), "imágenes descritas e indexadas"),
        (f"{media('hybrid') * 1000:.0f} ms", "retrieval promedio (híbrido)"),
    ]
    if n_con_llm:
        kpis.append((f"{sum(f['t_generacion'] for f in n_con_llm) / len(n_con_llm):.0f} s", "generación promedio (gemma3:12b)"))

    resumen = ""
    for key in keys:
        aciertos = [hit(f, key) for f in filas if hit(f, key) is not None]
        n_img = [sum(1 for c in f["retrievers"][key]["chunks"] if c["imagen"]) for f in img_q]
        resumen += (f"<tr><td><b>{RETRIEVERS[key][0]}</b></td><td>{media(key) * 1000:.1f} ms</td>"
                    f"<td>{sum(aciertos)}/{len(aciertos)}</td>"
                    f"<td>{sum(n_img)} de {5 * len(img_q)}</td></tr>")

    tarjetas = []
    for n, f in enumerate(filas, 1):
        cols = ""
        for key in keys:
            r = f["retrievers"][key]
            h = hit(f, key)
            marca = "" if h is None else (' <span class="ok">✓</span>' if h else ' <span class="no">✗</span>')
            lis = "".join(
                f'<li><span class="sc">{c["score"]:.3f}</span>'
                f'{"<span class=im>[img]</span> " if c["imagen"] else ""}{html.escape(articulo(c["fuente"]))}</li>'
                for c in r["chunks"])
            cols += (f'<div><h3>{RETRIEVERS[key][0]}{marca} <span class="mut">· {r["t_retrieval"] * 1000:.0f} ms</span></h3>'
                     f'<ul class="src">{lis}</ul></div>')

        galeria, vistas = "", set()
        for c in f["retrievers"]["hybrid"]["chunks"]:
            if c["imagen"] and c["imagen"] not in vistas:
                vistas.add(c["imagen"])
                src = b64_img(c["imagen"])
                if src:
                    cap = html.escape(c["texto"].split("] ", 1)[-1][:170]) + "…"
                    galeria += (f'<figure class="fig"><img src="{src}" alt="{html.escape(articulo(c["fuente"]))}">'
                                f'<figcaption><b>{html.escape(articulo(c["fuente"]))}</b> — {cap}</figcaption></figure>')

        resp = ""
        if "respuesta" in f:
            resp = (f'<h3>Respuesta (híbrido + gemma3:12b · {f["t_generacion"]:.1f} s)</h3>'
                    f'<div class="ans">{md_basico(f["respuesta"])}</div>')
        tarjetas.append(
            f'<section class="card q"><span class="chip">{n}. {html.escape(f["tipo"])}</span>'
            f'<div class="t">{html.escape(f["pregunta"])}</div>{resp}'
            f'<div class="cols">{cols}</div>'
            f'{"<div class=gal>" + galeria + "</div>" if galeria else ""}</section>')

    kpi_html = "".join(f'<div class="card kpi"><div class="n">{n}</div><div class="l">{l}</div></div>' for n, l in kpis)
    arch = ("Pregunta\n  │\n  ▼\nRetriever (intercambiable por YAML)\n"
            "  ├─ Cosine / FAISS   embeddings multilingües\n  ├─ BM25             léxico\n"
            "  └─ Híbrido          fusión RRF de los dos\n  │   (índice = 601 chunks de texto + 60 descripciones de imágenes)\n"
            "  ▼\nGenerador: gemma3:12b (Ollama, local)\n  ▼\nRespuesta citando fragmentos y fuentes")
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG Astronomía — resultados</title><style>{CSS}</style></head><body><main>
<h1>RAG de Astronomía: texto + imágenes</h1>
<p class="sub">Corpus: Wikipedia en español (Sistema Solar y exploración espacial). Generado el {d["fecha"]} · {d["modelo"]} local, sin servicios externos.</p>
<div class="grid k4">{kpi_html}</div>
<h2>Arquitectura</h2><div class="card arch">{html.escape(arch)}</div>
<h2>Comparación de retrievers ({len(filas)} preguntas)</h2>
<div class="card"><table><tr><th>Retriever</th><th>Retrieval prom.</th><th>Artículo esperado en top-5</th><th>Imágenes en top-5 (preguntas de imagen)</th></tr>{resumen}</table>
<p class="mut" style="font-size:13px;margin:10px 0 0">“Artículo esperado” es una comprobación automática simple (¿aparecen en el top-5 los artículos que contienen la respuesta?), no una evaluación formal de relevancia. La pregunta trampa no cuenta.</p></div>
<h2>Resultados por pregunta</h2>
{"".join(tarjetas)}
</main></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-llm", action="store_true")
    ap.add_argument("--solo-html", action="store_true")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    json_path, html_path = "results/resultados.json", "results/informe.html"
    if args.solo_html:
        with open(json_path, encoding="utf-8") as f:
            datos = json.load(f)
    else:
        t0 = time.time()
        datos = ejecutar(args.sin_llm)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False, indent=1)
        print(f"\nCorrida completa en {time.time() - t0:.0f}s")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(generar_html(datos))
    print(f"Informe: {html_path} ({os.path.getsize(html_path) // 1024} KB)")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
