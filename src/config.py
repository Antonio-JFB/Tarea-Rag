"""
Config validada con Pydantic + factory de pipelines.

build_pipeline() es el único lugar que conoce las clases concretas.
Cambiar de método de retrieval = cambiar `retriever: cosine` -> `retriever: bm25`
en el YAML; el pipeline no se toca.
"""

from typing import Literal

import yaml
from pydantic import BaseModel

from src.chunkers import ParagraphChunker, SlidingWindowChunker
from src.generators import OllamaGenerator
from src.interfaces import BaseChunker, Chunk
from src.pipeline import RAGPipeline, load_corpus_chunks, load_image_chunks
from src.postprocessors import NoOpPostprocessor
from src.preprocessing import build_preprocessor
from src.retrievers import BM25Retriever, CosineFaissRetriever, HybridRetriever


class PipelineConfig(BaseModel):
    nombre: str
    retriever: Literal["cosine", "bm25", "hybrid"]
    embed_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 5
    chunking: Literal["sliding_window", "paragraph"] = "sliding_window"
    chunk_size: int = 300
    chunk_overlap: int = 50
    chunk_max_words: int = 85
    corpus_dir: str = "data"  # carpeta con los .txt de los artículos
    imagenes_dir: str | None = "imagenes"  # carpeta con captions.json (None = solo texto)
    index_dir: str | None = None
    dominio: str = "astronomía y exploración espacial"  # se inserta en el prompt del LLM
    lexical_preprocessing: Literal["none", "lemma_stopwords"] = "none"
    rrf_k: int = 60
    dense_weight: float = 0.5
    lexical_weight: float = 0.5
    fusion_depth: int = 20
    ollama_model: str = "gemma3:12b"
    ollama_url: str = "http://localhost:11434/api/generate"


def load_config(path: str) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(**raw)


def _cosine(cfg: PipelineConfig) -> CosineFaissRetriever:
    kwargs = {"index_path": f"{cfg.index_dir}/faiss.index"} if cfg.index_dir else {}
    return CosineFaissRetriever(embed_model=cfg.embed_model, **kwargs)


def _bm25(cfg: PipelineConfig) -> BM25Retriever:
    kwargs = {"index_path": f"{cfg.index_dir}/bm25.pkl"} if cfg.index_dir else {}
    return BM25Retriever(preprocessor=build_preprocessor(cfg.lexical_preprocessing), **kwargs)


def _hybrid(cfg: PipelineConfig) -> HybridRetriever:
    return HybridRetriever(
        _cosine(cfg), _bm25(cfg),
        rrf_k=cfg.rrf_k, dense_weight=cfg.dense_weight,
        lexical_weight=cfg.lexical_weight, fusion_depth=cfg.fusion_depth,
    )


RETRIEVERS = {"cosine": _cosine, "bm25": _bm25, "hybrid": _hybrid}

CHUNKERS = {
    "sliding_window": lambda cfg: SlidingWindowChunker(chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap),
    "paragraph": lambda cfg: ParagraphChunker(max_words=cfg.chunk_max_words),
}


def load_chunks_for(config: PipelineConfig, chunker: BaseChunker) -> list[Chunk]:
    """Chunks a indexar: el texto de `corpus_dir` más, si hay `imagenes_dir`, un chunk por
    cada imagen descrita (captions.json)."""
    chunks = load_corpus_chunks(chunker, config.corpus_dir)
    if config.imagenes_dir:
        chunks = chunks + load_image_chunks(config.imagenes_dir)
    return chunks


def build_pipeline(config: PipelineConfig) -> RAGPipeline:
    chunker = CHUNKERS[config.chunking](config)
    retriever = RETRIEVERS[config.retriever](config)
    postprocessor = NoOpPostprocessor()
    generator = OllamaGenerator(model=config.ollama_model, url=config.ollama_url, dominio=config.dominio)
    return RAGPipeline(chunker, retriever, postprocessor, generator, top_k=config.top_k)
