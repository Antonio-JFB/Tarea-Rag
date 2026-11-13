"""Orquestador del framework (Proposal 2)."""

import glob
import json
import logging
import os
import time

from src.interfaces import (
    BaseChunker,
    BaseGenerator,
    BasePostprocessor,
    BaseRetriever,
    Chunk,
    PipelineResult,
    RetrievedChunk,
)

log = logging.getLogger(__name__)


def load_chunks(meta_path: str = "embeddings/metadata.json") -> list[Chunk]:
    """Carga los chunks ya construidos en Fase 0 (mismo corpus, mismo chunking)."""
    with open(meta_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Chunk(**c) for c in raw]


def load_corpus_chunks(chunker: BaseChunker, data_dir: str = "data") -> list[Chunk]:
    """Chunkea el corpus de data/ con el chunker dado (mismo criterio de titulo que rag_ollama.py)."""
    chunks: list[Chunk] = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        titulo = os.path.basename(path).replace(".txt", "").replace("_", " ")
        with open(path, "r", encoding="utf-8") as f:
            chunks.extend(chunker.chunk(f.read(), titulo))
    return chunks


def load_image_chunks(imagenes_dir: str) -> list[Chunk]:
    """Un chunk por imagen descrita (captions.json, generado por captionear_imagenes.py).
    El chunk lleva la descripción como texto —así los retrievers de texto lo indexan igual
    que a cualquier otro— y la ruta de la imagen en `imagen` para poder mostrarla."""
    cap_path = os.path.join(imagenes_dir, "captions.json")
    if not os.path.exists(cap_path):
        log.warning(f"No hay {cap_path}; se indexa solo texto.")
        return []
    with open(cap_path, "r", encoding="utf-8") as f:
        captions = json.load(f)

    chunks: list[Chunk] = []
    for archivo, info in sorted(captions.items()):
        texto = f"[Imagen del artículo {info['articulo']}] {info['caption']}"
        n = len(texto.split())
        chunks.append(Chunk(
            id=f"img_{archivo}",
            texto=texto,
            fuente=f"Imagen · {info['articulo']}",
            inicio_palabra=0,
            fin_palabra=n,
            num_palabras=n,
            imagen=os.path.join(imagenes_dir, archivo),
        ))
    return chunks


class RAGPipeline:
    """Orquesta retrieve -> postprocess -> generate y captura observabilidad
    completa: query, chunks recuperados + scores, tiempos, prompt completo,
    respuesta (7/7 campos, cierra el hueco de prompt logging del baseline)."""

    def __init__(
        self,
        chunker: BaseChunker,
        retriever: BaseRetriever,
        postprocessor: BasePostprocessor,
        generator: BaseGenerator,
        top_k: int = 5,
    ):
        self.chunker = chunker
        self.retriever = retriever
        self.postprocessor = postprocessor
        self.generator = generator
        self.top_k = top_k

    def index(self, chunks: list[Chunk]) -> None:
        self.retriever.build_index(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> tuple[list[RetrievedChunk], float]:
        """Solo recuperacion (sin generacion): devuelve los chunks y el tiempo de retrieval."""
        t0 = time.time()
        resultados = self.retriever.retrieve(query, top_k or self.top_k)
        resultados = self.postprocessor.process(query, resultados)
        return resultados, time.time() - t0

    def answer(self, query: str, top_k: int | None = None) -> PipelineResult:
        top_k = top_k or self.top_k
        t0 = time.time()

        t_r0 = time.time()
        resultados = self.retriever.retrieve(query, top_k)
        resultados = self.postprocessor.process(query, resultados)
        t_retrieval = time.time() - t_r0

        generation = self.generator.generate(query, resultados)
        t_total = time.time() - t0

        result = PipelineResult(
            query=query,
            retrieved=resultados,
            generation=generation,
            t_retrieval=t_retrieval,
            t_total=t_total,
        )
        self._log(result)
        return result

    def _log(self, r: PipelineResult) -> None:
        log.info("=" * 70)
        log.info(f"QUERY     : {r.query}")
        log.info(
            f"RETRIEVAL : {r.t_retrieval:.3f}s | LLM: {r.generation.latency_s:.3f}s | "
            f"Total: {r.t_total:.3f}s"
        )
        log.info(f"CHUNKS RECUPERADOS ({len(r.retrieved)}):")
        for i, rc in enumerate(r.retrieved, 1):
            log.info(
                f"  [{i}] score={rc.score:.4f} | fuente={rc.chunk.fuente} | "
                f"palabras {rc.chunk.inicio_palabra}-{rc.chunk.fin_palabra}"
            )
        log.info(f"PROMPT_ENVIADO:\n{r.generation.prompt}")
        log.info(f"RESPUESTA:\n{r.generation.respuesta}")
        log.info("=" * 70)
