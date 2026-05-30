from typing import List, Dict, Any  
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_core.documents import Document  
from backend.utils.logger import get_logger

log = get_logger("chunker")


class Chunker:  
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):  
        self.chunk_size    = chunk_size  
        self.chunk_overlap = chunk_overlap  
        self.splitter = RecursiveCharacterTextSplitter(  
            chunk_size=chunk_size,  
            chunk_overlap=chunk_overlap,  
            separators=["\n\n", "\n", ".", " ", ""],  
            length_function=len,  
        )  
        log.info(f"Chunker initialized — chunk_size: {chunk_size} | overlap: {chunk_overlap}")

    def chunk_document(self, doc_result: Dict[str, Any]) -> Dict[str, Any]:  
        filename = doc_result.get("filename", "unknown")  
        pages    = doc_result.get("documents", [])  
        log.info(f"[CHUNK] Starting — {filename} | {len(pages)} pages")

        if not pages:  
            log.warning(f"[CHUNK] ⚠️ No pages found for {filename} — skipping")  
            doc_result["chunks"]       = []  
            doc_result["total_chunks"] = 0  
            return doc_result

        try:  
            chunks: List[Document] = self.splitter.split_documents(pages)  
            log.debug(f"[CHUNK] Split complete — {len(chunks)} raw chunks")

            for i, chunk in enumerate(chunks):  
                chunk.metadata["chunk_index"] = i  
                chunk.metadata["chunk_size"]  = len(chunk.page_content)

            doc_result["chunks"]       = chunks  
            doc_result["total_chunks"] = len(chunks)  
            log.info(f"[CHUNK] ✅ {filename} — {len(pages)} pages → {len(chunks)} chunks")

        except Exception as e:  
            log.error(f"[CHUNK] ❌ FAILED for {filename} — {type(e).__name__}: {e}", exc_info=True)  
            doc_result["chunks"]       = []  
            doc_result["total_chunks"] = 0

        return doc_result

    def chunk_documents(self, doc_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:  
        log.info(f"[CHUNK] Processing {len(doc_results)} document(s)")  
        return [self.chunk_document(r) for r in doc_results]  
