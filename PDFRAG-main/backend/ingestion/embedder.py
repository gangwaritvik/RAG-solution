import httpx  
from typing import List  
from concurrent.futures import ThreadPoolExecutor, as_completed  
from openai import AzureOpenAI  
from backend.config import AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, EMBEDDING_MODEL  
from backend.utils.logger import get_logger

log = get_logger("embedder")


class Embedder:  
    def __init__(self, max_workers: int = 10):  
        log.info(f"Embedder initializing — endpoint: {AZURE_ENDPOINT} | model: {EMBEDDING_MODEL}")

        self.client = AzureOpenAI(  
            azure_endpoint=AZURE_ENDPOINT,  
            api_key=AZURE_API_KEY,  
            api_version=AZURE_API_VERSION,  
            timeout=httpx.Timeout(30.0, connect=10.0),  
            max_retries=2,  
        )  
        self.model = EMBEDDING_MODEL  
        self.max_workers = max_workers

        log.info(f"Embedder ✅ initialized | max_workers: {max_workers}")

    def _embed_batch(self, batch: List[str], batch_num: int) -> dict:  
        """Embed a single batch — runs inside a thread."""  
        log.debug(f"[EMBED] Batch {batch_num} — {len(batch)} texts (thread started)")

        try:  
            response = self.client.embeddings.create(  
                input=batch,  
                model=self.model,  
            )  
            embeddings = [item.embedding for item in response.data]  
            dim = len(embeddings[0]) if embeddings else 0

            log.info(f"[EMBED] ✅ Batch {batch_num} — {len(batch)} texts embedded | dim: {dim}")

            return {  
                "batch_num": batch_num,  
                "embeddings": embeddings,  
                "error": None,  
            }

        except Exception as e:  
            log.error(f"[EMBED] ❌ Batch {batch_num} FAILED — {type(e).__name__}: {e}", exc_info=True)  
            return {  
                "batch_num": batch_num,  
                "embeddings": [],  
                "error": e,  
            }

    def embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:  
        batch_size = int(batch_size)

        if not texts:  
            log.warning("[EMBED] No texts to embed")  
            return []

        # ── Build batches ──  
        batches = []  
        for i in range(0, len(texts), batch_size):  
            batch = texts[i:i + batch_size]  
            batch_num = i // batch_size + 1  
            batches.append((batch, batch_num))

        total_batches = len(batches)  
        log.info(f"[EMBED] Embedding {len(texts)} texts in {total_batches} batch(es) | workers: {self.max_workers}")

        # ── Sequential if only 1 batch ──  
        if total_batches == 1:  
            result = self._embed_batch(batches[0][0], batches[0][1])  
            if result["error"]:  
                raise result["error"]  
            return result["embeddings"]

        # ── Concurrent embedding ──  
        all_results = [None] * total_batches

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:  
            future_to_batch = {  
                executor.submit(self._embed_batch, batch, batch_num): batch_num  
                for batch, batch_num in batches  
            }

            for future in as_completed(future_to_batch):  
                result = future.result()  
                batch_num = result["batch_num"]

                if result["error"]:  
                    log.error(f"[EMBED] ❌ Batch {batch_num} failed — aborting")  
                    raise result["error"]

                # Store in correct order  
                all_results[batch_num - 1] = result["embeddings"]

        # ── Flatten results in order ──  
        all_embeddings = []  
        for batch_embeddings in all_results:  
            if batch_embeddings:  
                all_embeddings.extend(batch_embeddings)

        log.info(f"[EMBED] ✅ All done — {len(all_embeddings)} embeddings total (concurrent)")  
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:  
        log.info(f"[EMBED] Embedding query: '{query[:60]}...' " if len(query) > 60 else f"[EMBED] Embedding query: '{query}'")

        try:  
            response = self.client.embeddings.create(  
                input=[query],  
                model=self.model,  
            )  
            vec = response.data[0].embedding  
            log.info(f"[EMBED] ✅ Query embedded — dim: {len(vec)}")  
            return vec

        except Exception as e:  
            log.error(f"[EMBED] ❌ Query embedding FAILED — {type(e).__name__}: {e}", exc_info=True)  
            raise  
