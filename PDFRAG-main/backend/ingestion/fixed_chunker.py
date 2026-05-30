from typing import List, Dict, Any  
from langchain_core.documents import Document


class FixedChunker:  
    def __init__(self, chunk_size: int = 500):  
        if chunk_size <= 0:  
            raise ValueError("chunk_size must be greater than 0")  
        self.chunk_size = chunk_size

    def chunk_documents(  self,  
        doc_results: List[Dict[str, Any]],  
        chunk_size: int | None = None,  
    ) -> List[Dict[str, Any]]:  
        return [  
            self.chunk_document(doc_result, chunk_size=chunk_size)  
            for doc_result in doc_results  
        ]

    def chunk_document(  
        self,  
        doc_result: Dict[str, Any],  
        chunk_size: int | None = None,  
    ) -> Dict[str, Any]:  
        pages: List[Document] = doc_result.get("documents", [])  
        size = chunk_size or self.chunk_size

        if size <= 0:  
            raise ValueError("chunk_size must be greater than 0")

        if not pages:  
            doc_result["chunks"] = []  
            doc_result["total_chunks"] = 0  
            return doc_result

        chunks: List[Document] = []

        for page in pages:  
            text = (page.page_content or "").strip()  
            if not text:  
                continue

            for start in range(0, len(text), size):  
                chunk_text = text[start:start + size].strip()  
                if not chunk_text:  
                    continue

                chunks.append(  
                    Document(  
                        page_content=chunk_text,  
                        metadata={  
                            **page.metadata,  
                            "page": page.metadata.get("page", 0),  
                            "start_index": start,  
                            "end_index": start + len(chunk_text),  
                            "chunk_method": "fixed",  
                        },  
                    )  
                )

        for i, chunk in enumerate(chunks):  
            chunk.metadata["chunk_index"] = i  
            chunk.metadata["chunk_size"] = len(chunk.page_content)  
            chunk.metadata["doc_id"] = doc_result.get("doc_id")  
            chunk.metadata["filename"] = doc_result.get("filename")

        doc_result["chunks"] = chunks  
        doc_result["total_chunks"] = len(chunks)

        print(  
            f"[FIXED CHUNKED] {doc_result.get('filename')} | "  
            f"pages={len(pages)} -> chunks={len(chunks)} | chunk_size={size}"  
        )  
        return doc_result  

# ── Adapter: makes FixedChunker compatible with the pipeline ──  
def chunk_documents_pipeline(self, doc_results):  
    """  
    Wraps chunk_documents() to match pipeline dict structure.  
    Converts LangChain Document lists from doc_results.  
    """  
    return self.chunk_documents(doc_results)  
