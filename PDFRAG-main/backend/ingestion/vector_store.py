import chromadb  
from typing import List, Dict, Any  
import uuid  
from backend.utils.logger import get_logger

log = get_logger("vector_store")


class VectorStore:  
    def delete_by_filename(self, filename: str) -> int:   
        try:  
            # Get all IDs matching this filename  
            results = self.collection.get(  
                where={"filename": filename},  
                include=["metadatas"],  
            )

            ids_to_delete = results.get("ids", [])

            if not ids_to_delete:  
                log.warning(f"[STORE] No vectors found for filename: {filename}")  
                return 0

            self.collection.delete(ids=ids_to_delete)  
            log.info(f"[STORE] ✅ Deleted {len(ids_to_delete)} vectors for: {filename}")  
            self._filenames.discard(filename)  # keep filename cache in sync
            return len(ids_to_delete)

        except Exception as e:  
            log.error(f"[STORE] ❌ Delete FAILED for {filename} — {type(e).__name__}: {e}", exc_info=True)  
            raise  

    def __init__(self, persist_dir: str = "chroma_db"):  
        log.info(f"VectorStore initializing — persist_dir: {persist_dir}")  
        try:  
            self.client = chromadb.PersistentClient(path=persist_dir)  
            self.collection = self.client.get_or_create_collection(  
                name="pdf_rag",  
                metadata={"hnsw:space": "cosine"},  
            )  
            # Cache of distinct filenames currently stored, kept in sync on
            # add/delete/clear so per-query lookups don't rescan all chunk text.
            self._filenames = set()
            self._refresh_filename_cache()
            log.info(f"VectorStore ✅ Collection loaded — {self.collection.count()} existing vectors")  
        except Exception as e:  
            log.error(f"VectorStore ❌ Failed to initialize — {type(e).__name__}: {e}", exc_info=True)  
            raise

    def _refresh_filename_cache(self):
        """Rebuild the cached set of distinct filenames from stored metadata."""
        try:
            data = self.collection.get(include=["metadatas"])
            self._filenames = {
                str(m.get("filename", "")).strip()
                for m in data.get("metadatas", [])
                if m.get("filename")
            }
        except Exception as e:
            log.error(f"[STORE] Failed to refresh filename cache — {type(e).__name__}: {e}")
            self._filenames = set()

    def list_filenames(self) -> List[str]:
        """Return the sorted list of distinct filenames currently in the store."""
        return sorted(f for f in self._filenames if f)

    def add(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]):  
        log.info(f"[STORE] Adding {len(embeddings)} vectors")  
        try:  
            ids       = [str(uuid.uuid4()) for _ in embeddings]  
            documents = [m.get("text", "") for m in metadata]  
            clean_meta = [  
                {  
                    "text":        str(m.get("text", "")),  
                    "filename":    str(m.get("filename", "")),  
                    "doc_id":      str(m.get("doc_id", "")),  
                    "chunk_index": int(m.get("chunk_index") or 0),  
                    "page":        int(m.get("page") or 0),  
                }  
                for m in metadata  
            ]  
            self.collection.add(  
                ids=ids,  
                embeddings=embeddings,  
                documents=documents,  
                metadatas=clean_meta,  
            )  
            for m in clean_meta:  # keep filename cache in sync
                if m["filename"]:
                    self._filenames.add(m["filename"])
            log.info(f"[STORE] ✅ {len(embeddings)} vectors added — total now: {self.collection.count()}")  
        except Exception as e:  
            log.error(f"[STORE] ❌ Add FAILED — {type(e).__name__}: {e}", exc_info=True)  
            raise

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:  
        log.info(f"[STORE] Searching top-{top_k} — total vectors: {self.collection.count()}")  
        if self.collection.count() == 0:  
            log.warning("[STORE] ⚠️ Collection is empty — no results")  
            return []  
        try:  
            results = self.collection.query(  
                query_embeddings=[query_embedding],  
                n_results=min(top_k, self.collection.count()),  
                include=["documents", "metadatas", "distances"],  
            )  
            hits = []  
            for i in range(len(results["ids"][0])):  
                meta  = results["metadatas"][0][i]  
                score = 1 - results["distances"][0][i]  
                hits.append({**meta, "score": round(score, 4)})

            log.info(f"[STORE] ✅ {len(hits)} hits returned | top score: {hits[0]['score'] if hits else 'N/A'}")  
            return hits  
        except Exception as e:  
            log.error(f"[STORE] ❌ Search FAILED — {type(e).__name__}: {e}", exc_info=True)  
            raise

    def clear(self):  
        self.client.delete_collection("pdf_rag")  
        self.collection = self.client.get_or_create_collection(  
            name="pdf_rag",  
            metadata={"hnsw:space": "cosine"},  
        )  
        self._filenames = set()  # reset filename cache
        log.info("[STORE] Collection cleared.")

    @property  
    def count(self):  
        return self.collection.count()  
