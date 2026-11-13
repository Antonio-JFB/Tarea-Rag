"""
Consola para probar el RAG con el corpus de Astronomía (sin Streamlit).

Uso:
    python rag_consola.py                              # interactivo, config cosine
    python rag_consola.py --config configs/hybrid.yaml
    python rag_consola.py -q "¿Cuántas lunas tiene Júpiter?"
    python rag_consola.py -q "..." --solo-retrieval    # no llama al LLM (rápido)
"""

import argparse

from src.config import build_pipeline, load_chunks_for, load_config


def mostrar(pipeline, query: str, solo_retrieval: bool) -> None:
    if solo_retrieval:
        resultados, t = pipeline.retrieve(query)
        print(f"\n  retrieval={t:.3f}s")
        for i, rc in enumerate(resultados, 1):
            extra = f"  -> {rc.chunk.imagen}" if rc.chunk.imagen else ""
            print(f"  [{i}] score={rc.score:.3f} · {rc.chunk.fuente}{extra}")
            print(f"      {rc.chunk.texto[:200].strip()}...")
        return

    r = pipeline.answer(query)
    print(f"\n  retrieval={r.t_retrieval:.3f}s  total={r.t_total:.3f}s")
    for i, rc in enumerate(r.retrieved, 1):
        extra = f"  -> {rc.chunk.imagen}" if rc.chunk.imagen else ""
        print(f"  [{i}] score={rc.score:.3f} · {rc.chunk.fuente}{extra}")
    print(f"\n{r.generation.respuesta}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cosine.yaml")
    ap.add_argument("-q", "--query", help="una sola pregunta (si se omite, modo interactivo)")
    ap.add_argument("--solo-retrieval", action="store_true", help="no llama al LLM")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"=== {cfg.nombre} ({args.config}) ===")
    pipeline = build_pipeline(cfg)
    chunks = load_chunks_for(cfg, pipeline.chunker)
    print(f"Indexando {len(chunks)} chunks de {cfg.corpus_dir}/ ...")
    pipeline.index(chunks)

    if args.query:
        mostrar(pipeline, args.query, args.solo_retrieval)
        return

    print("Escribe una pregunta ('salir' para terminar).")
    while True:
        q = input("\n> ").strip()
        if q.lower() in {"salir", "exit", "q"}:
            break
        if q:
            mostrar(pipeline, q, args.solo_retrieval)


if __name__ == "__main__":
    main()
