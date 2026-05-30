from typing import List, Dict, Any  
from langchain_core.documents import Document


class SlidingWindowChunker:  
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):  
        if chunk_size <= 0:  
            raise ValueError("chunk_size must be greater than 0")  
        if chunk_overlap < 0:  
            raise ValueError("chunk_overlap cannot be negative")  
        if chunk_overlap >= chunk_size:  
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size  
        self.chunk_overlap = chunk_overlap

    def chunk_documents(  
        self,  
        doc_results: List[Dict[str, Any]],  
        chunk_size: int | None = None,  
        chunk_overlap: int | None = None,  
    ) -> List[Dict[str, Any]]:  
        return [  
            self.chunk_document(  
                doc_result,  
                chunk_size=chunk_size,  
                chunk_overlap=chunk_overlap,  
            )  
            for doc_result in doc_results  
        ]

    def chunk_document(  
        self,  
        doc_result: Dict[str, Any],  
        chunk_size: int | None = None,  
        chunk_overlap: int | None = None,  
    ) -> Dict[str, Any]:  
        pages: List[Document] = doc_result.get("documents", [])  
        size = chunk_size or self.chunk_size  
        overlap = self.chunk_overlap if chunk_overlap is None else chunk_overlap

        if size <= 0:  
            raise ValueError("chunk_size must be greater than 0")  
        if overlap < 0:  
            raise ValueError("chunk_overlap cannot be negative")  
        if overlap >= size:  
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        if not pages:  
            doc_result["chunks"] = []  
            doc_result["total_chunks"] = 0  
            return doc_result

        step = size - overlap  
        chunks: List[Document] = []

        for page in pages:  
            text = (page.page_content or "").strip()  
            if not text:  
                continue

            for start in range(0, len(text), step):  
                chunk_text = text[start:start + size].strip()  
                if not chunk_text:  
                    continue

                chunks.append(  
                    Document(  
                        page_content=chunk_text,  
                        metadata={  
                            **page.metadata,  
                            "page": page.metadata.get("page", 0),  
                            "window_start": start,  
                            "window_end": start + len(chunk_text),  
                            "chunk_method": "sliding",  
                        },  
                    )  
                )

                if start + size >= len(text):  
                    break

        for i, chunk in enumerate(chunks):  
            chunk.metadata["chunk_index"] = i  
            chunk.metadata["chunk_size"] = len(chunk.page_content)  
            chunk.metadata["doc_id"] = doc_result.get("doc_id")  
            chunk.metadata["filename"] = doc_result.get("filename")

        doc_result["chunks"] = chunks  
        doc_result["total_chunks"] = len(chunks)

        print(  
            f"[SLIDING CHUNKED] {doc_result.get('filename')} | "  
            f"pages={len(pages)} -> chunks={len(chunks)} | "  
            f"chunk_size={size} | overlap={overlap}"  
        )  
        return doc_result  
