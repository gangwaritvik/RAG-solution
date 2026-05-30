from typing import List  
from openai import AzureOpenAI  
from backend.config import AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, EMBEDDING_MODEL  
from backend.utils.logger import get_logger

log = get_logger("embedder")


class Embedder:  
    def __init__(self):  
        log.info(f"Embedder initializing — endpoint: {AZURE_ENDPOINT} | model: {EMBEDDING_MODEL}")  
        log.debug(f"API Key (first 10 chars): {str(AZURE_API_KEY)[:10]}...")  
        log.debug(f"API Version: {AZURE_API_VERSION}")

        try:  
            self.client = AzureOpenAI(  
                azure_endpoint=AZURE_ENDPOINT,  
                api_key=AZURE_API_KEY,  
                api_version=AZURE_API_VERSION,  
            )  
            self.model = EMBEDDING_MODEL  
            log.info("Embedder ✅ AzureOpenAI client created successfully")  
        except Exception as e:  
            log.error(f"Embedder ❌ Failed to create client — {type(e).__name__}: {e}", exc_info=True)  
            raise

    def embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:  
        log.info(f"[EMBED] Embedding {len(texts)} texts in batches of {batch_size}")  
        all_embeddings = []

        for i in range(0, len(texts), batch_size):  
            batch = texts[i: i + batch_size]  
            batch_num = i // batch_size + 1  
            log.debug(f"[EMBED] Sending batch {batch_num} — {len(batch)} texts")

            try:  
                response = self.client.embeddings.create(  
                    input=batch,  
                    model=self.model,  
                )  
                embeddings = [item.embedding for item in response.data]  
                all_embeddings.extend(embeddings)  
                log.info(f"[EMBED] ✅ Batch {batch_num} — {len(batch)} texts embedded | vector dim: {len(embeddings[0])}")

            except Exception as e:  
                log.error(f"[EMBED] ❌ Batch {batch_num} FAILED — {type(e).__name__}: {e}", exc_info=True)  
                raise

        log.info(f"[EMBED] ✅ All done — {len(all_embeddings)} embeddings total")  
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:  
        log.info(f"[EMBED] Embedding query: '{query[:60]}...' " if len(query) > 60 else f"[EMBED] Embedding query: '{query}'")  
        try:  
            response = self.client.embeddings.create(  
                input=[query],  
                model=self.model,  
            )  
            vec = response.data[0].embedding  
            log.info(f"[EMBED] ✅ Query embedded — vector dim: {len(vec)}")  
            return vec  
        except Exception as e:  
            log.error(f"[EMBED] ❌ Query embedding FAILED — {type(e).__name__}: {e}", exc_info=True)  
            raise  
