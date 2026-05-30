import re  
import numpy as np  
from typing import List, Dict, Any  
from langchain_core.documents import Document  
from openai import AzureOpenAI  
from backend.config import (  
    AZURE_ENDPOINT, AZURE_API_KEY,  
    AZURE_API_VERSION, EMBEDDING_MODEL  
)  
from backend.utils.logger import get_logger

log = get_logger("semantic_chunker")


class SemanticChunker:  
    def __init__(  
        self,  
        breakpoint_threshold: float = 0.3,  
        min_chunk_size: int = 100,  
        max_chunk_size: int = 1000,  
        buffer_size: int = 1,  
    ):  
        self.breakpoint_threshold = breakpoint_threshold  
        self.min_chunk_size       = min_chunk_size  
        self.max_chunk_size       = max_chunk_size  
        self.buffer_size          = buffer_size

        self.client = AzureOpenAI(  
            azure_endpoint=AZURE_ENDPOINT,  
            api_key=AZURE_API_KEY,  
            api_version=AZURE_API_VERSION,  
        )  
        self.model = EMBEDDING_MODEL  
        log.info(f"SemanticChunker initialized — threshold: {breakpoint_threshold}")

    def _split_into_sentences(self, text: str) -> List[str]:  
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())  
        return [s.strip() for s in sentences if s.strip()]

    def _embed_sentences(self, sentences: List[str]) -> List[List[float]]:  
        log.info(f"[SEMANTIC] Embedding {len(sentences)} sentences")  
        all_embeddings = []  
        for i in range(0, len(sentences), 100):  
            batch = sentences[i: i + 100]  
            response = self.client.embeddings.create(  
                input=batch,  
                model=self.model,  
            )  
            all_embeddings.extend([item.embedding for item in response.data])  
        return all_embeddings

    def _cosine_distance(self, v1, v2) -> float:  
        a, b = np.array(v1), np.array(v2)  
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    def _compute_distances(self, embeddings: List[List[float]]) -> List[float]:  
        distances = []  
        for i in range(len(embeddings) - 1):  
            start = max(0, i - self.buffer_size)  
            end   = min(len(embeddings), i + self.buffer_size + 1)  
            cur   = np.mean(embeddings[start: i + 1], axis=0).tolist()  
            nxt   = np.mean(embeddings[i + 1: end + 1], axis=0).tolist()  
            distances.append(self._cosine_distance(cur, nxt))  
        return distances

    def _find_breakpoints(self, distances: List[float]) -> List[int]:  
        return [i + 1 for i, d in enumerate(distances) if d > self.breakpoint_threshold]

    def _group_into_chunks(  
        self, sentences, breakpoints, page_number, filename, doc_id  
    ) -> List[Document]:  
        groups = []  
        start  = 0  
        for bp in breakpoints:  
            groups.append(sentences[start:bp])  
            start = bp  
        groups.append(sentences[start:])

        chunks    = []  
        chunk_idx = 0

        for group in groups:  
            chunk_text = " ".join(group).strip()  
            if not chunk_text:  
                continue

            while len(chunk_text) > self.max_chunk_size:  
                split_point = chunk_text[:self.max_chunk_size].rfind('. ')  
                if split_point == -1:  
                    split_point = self.max_chunk_size  
                sub = chunk_text[:split_point + 1].strip()  
                if len(sub) >= self.min_chunk_size:  
                    chunks.append(Document(  
                        page_content=sub,  
                        metadata={  
                            "chunk_index": chunk_idx,  
                            "chunk_size":  len(sub),  
                            "page":        page_number,  
                            "filename":    filename,  
                            "doc_id":      doc_id,  
                            "chunk_type":  "semantic",  
                        }  
                    ))  
                    chunk_idx += 1  
                chunk_text = chunk_text[split_point + 1:].strip()

            if len(chunk_text) >= self.min_chunk_size:  
                chunks.append(Document(  
                    page_content=chunk_text,  
                    metadata={  
                        "chunk_index": chunk_idx,  
                        "chunk_size":  len(chunk_text),  
                        "page":        page_number,  
                        "filename":    filename,  
                        "doc_id":      doc_id,  
                        "chunk_type":  "semantic",  
                    }  
                ))  
                chunk_idx += 1

        return chunks

    def chunk_document(self, doc_result: Dict[str, Any]) -> Dict[str, Any]:  
        filename = doc_result.get("filename", "unknown")  
        pages    = doc_result.get("documents", [])  
        log.info(f"[SEMANTIC] Starting — {filename} | {len(pages)} pages")

        if not pages:  
            doc_result["chunks"]       = []  
            doc_result["total_chunks"] = 0  
            return doc_result

        try:  
            all_chunks = []  
            for page in pages:  
                page_text   = page.page_content.strip()  
                page_number = page.metadata.get("page", 0)  
                doc_id      = page.metadata.get("doc_id", "")

                if not page_text:  
                    continue

                sentences = self._split_into_sentences(page_text)  
                if len(sentences) <= 1:  
                    if len(page_text) >= self.min_chunk_size:  
                        all_chunks.append(Document(  
                            page_content=page_text,  
                            metadata={  
                                "chunk_index": len(all_chunks),  
                                "chunk_size":  len(page_text),  
                                "page":        page_number,  
                                "filename":    filename,  
                                "doc_id":      doc_id,  
                                "chunk_type":  "semantic",  
                            }  
                        ))  
                    continue

                embeddings  = self._embed_sentences(sentences)  
                distances   = self._compute_distances(embeddings)  
                breakpoints = self._find_breakpoints(distances)  
                page_chunks = self._group_into_chunks(  
                    sentences, breakpoints, page_number, filename, doc_id  
                )

                for chunk in page_chunks:  
                    chunk.metadata["chunk_index"] = len(all_chunks)  
                    all_chunks.append(chunk)

            doc_result["chunks"]       = all_chunks  
            doc_result["total_chunks"] = len(all_chunks)  
            log.info(f"[SEMANTIC] ✅ {filename} — {len(pages)} pages → {len(all_chunks)} chunks")

        except Exception as e:  
            log.error(f"[SEMANTIC] ❌ FAILED — {type(e).__name__}: {e}", exc_info=True)  
            doc_result["chunks"]       = []  
            doc_result["total_chunks"] = 0

        return doc_result

    def chunk_documents(self, doc_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:  
        return [self.chunk_document(r) for r in doc_results]  
