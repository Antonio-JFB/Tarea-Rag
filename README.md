# RAG — Astronomía (texto + imágenes)

RAG sobre un corpus de Wikipedia en español: Sistema Solar y exploración espacial
(21 artículos). Además del texto, indexa **imágenes de los artículos** descritas con un
modelo de visión, así que se pueden recuperar con preguntas en texto
("¿Cómo se ve Titán en una fotografía a color real?").

Proyecto independiente (tiene su propio `src/`, configs, datos e índices).

## Cómo funciona

```
descargar_corpus.py      Wikipedia -> data/*.txt              (texto de los artículos)
descargar_imagenes.py    Wikipedia -> imagenes/*.jpg|png      (+ manifest.json)
captionear_imagenes.py   imagen -> gemma3 (visión) -> imagenes/captions.json
                         └─ cada descripción se indexa como un chunk más (campo `imagen`)
src/ + configs/          pipeline: chunker -> retriever -> generador (Ollama)
rag_consola.py           consola          demo_app.py   demo Streamlit con miniaturas
```

Retrievers (se cambian desde el YAML, sin tocar código): `cosine` (FAISS), `bm25`, `hybrid` (RRF).

## Requisitos

- Python 3.11 y [Ollama](https://ollama.com/download) corriendo
- `ollama pull gemma3:12b` (generador y modelo de visión; es multimodal)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

Los datos ya vienen incluidos (`data/`, `imagenes/` con `captions.json`, `indices/`).
Para regenerarlos desde cero:

```bash
python descargar_corpus.py
python descargar_imagenes.py          # ~60 imágenes, hasta 3 por artículo
python captionear_imagenes.py         # ~4 s por imagen con GPU (reanudable)
```

Consultar:

```bash
python rag_consola.py                                # interactivo (configs/cosine.yaml)
python rag_consola.py --config configs/hybrid.yaml -q "¿Qué diferencias hay entre Marte y Venus?"
python rag_consola.py -q "¿Cómo se ve Europa?" --solo-retrieval   # sin LLM, rápido
streamlit run demo_app.py                            # demo con miniaturas
```

## Notas

- El RAG no "ve" la imagen al responder: usa la descripción en texto que generó el modelo de
  visión al indexarla. Si esa descripción falla, la imagen queda mal indexada.
- Los índices se reconstruyen solos si cambia el número de chunks. Si regeneras las
  descripciones sin cambiar la cantidad de imágenes, borra `indices/` para forzar la reconstrucción.
- `indices/astronomia_antiguo_601chunks/` es un índice viejo (solo texto) que ya no se usa; se puede borrar.
- La primera pregunta de una sesión puede tardar (Ollama carga el modelo en frío).
