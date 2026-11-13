"""
Demo Streamlit — RAG de Astronomía (texto + imágenes descritas).

Corre la misma pregunta contra los retrievers elegidos (Cosine/FAISS, BM25, Híbrido)
sobre el mismo corpus y el mismo LLM. Los chunks que son imágenes se muestran con su
miniatura.
"""

import os

import streamlit as st

from src.config import build_pipeline, load_chunks_for, load_config
from src.pipeline import RAGPipeline

st.set_page_config(page_title="RAG Astronomía", layout="wide")

PREGUNTAS_SUGERIDAS = [
    "(escribe tu propia pregunta)",
    "¿Qué diferencias hay entre Marte y Venus?",
    "¿Por qué Plutón ya no se considera un planeta?",
    "¿Qué es la Gran Mancha Roja de Júpiter?",
    "¿Cómo se ve Titán en una fotografía a color real?",
    "¿Cómo se ve la superficie de Europa?",
    "¿Quién ganó el mundial de fútbol de 1970?",  # pregunta trampa: fuera del corpus
]

CONFIGS = {
    "Cosine / FAISS (semántico)": "configs/cosine.yaml",
    "BM25 (léxico)": "configs/bm25.yaml",
    "Híbrido BM25 + embeddings (RRF)": "configs/hybrid.yaml",
}


@st.cache_resource(show_spinner="Cargando pipeline...")
def get_pipeline(config_path: str) -> RAGPipeline:
    cfg = load_config(config_path)
    pipeline = build_pipeline(cfg)
    pipeline.index(load_chunks_for(cfg, pipeline.chunker))
    return pipeline


st.title("RAG — Astronomía")
st.caption(
    "Corpus: artículos de Wikipedia en español sobre el Sistema Solar y la exploración espacial, "
    "más sus imágenes descritas con un modelo de visión (Ollama · gemma3:12b)."
)

elegidos = st.sidebar.multiselect(
    "Retrievers a comparar", list(CONFIGS), default=["Cosine / FAISS (semántico)", "Híbrido BM25 + embeddings (RRF)"]
)

pregunta_sel = st.selectbox("Pregunta sugerida", PREGUNTAS_SUGERIDAS)
query = st.text_input(
    "O escribe tu propia pregunta",
    value="" if pregunta_sel == PREGUNTAS_SUGERIDAS[0] else pregunta_sel,
)

comparar = st.button("Consultar", type="primary")

if comparar and not elegidos:
    st.warning("Elige al menos un retriever en la barra lateral.")
elif comparar and query.strip():
    columnas = st.columns(len(elegidos))
    for col, label in zip(columnas, elegidos):
        with col:
            st.subheader(label)
            pipeline = get_pipeline(CONFIGS[label])
            with st.spinner(f"Recuperando + generando ({label})..."):
                result = pipeline.answer(query)

            m1, m2 = st.columns(2)
            m1.metric("Retrieval", f"{result.t_retrieval:.3f}s")
            m2.metric("Total", f"{result.t_total:.3f}s")

            st.markdown("**Chunks recuperados**")
            for i, rc in enumerate(result.retrieved, 1):
                st.markdown(f"`[{i}]` score={rc.score:.3f} · **{rc.chunk.fuente}**")
                if rc.chunk.imagen and os.path.exists(rc.chunk.imagen):
                    st.image(rc.chunk.imagen, width=260)
                st.caption(rc.chunk.texto[:180].strip() + "...")

            st.markdown("**Respuesta**")
            st.write(result.generation.respuesta)

            with st.expander("Prompt completo enviado al LLM"):
                st.text(result.generation.prompt)
elif comparar:
    st.warning("Escribe o selecciona una pregunta primero.")
