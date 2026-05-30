from openai import AzureOpenAI  
from backend.config import AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, CHAT_MODEL  
from backend.utils.logger import get_logger

log = get_logger("generator")


class Generator:  
    def __init__(self):  
        self.client = AzureOpenAI(  
            azure_endpoint=AZURE_ENDPOINT,  
            api_key=AZURE_API_KEY,  
            api_version=AZURE_API_VERSION,  
        )

    def generate(self, query: str, context_chunks: list, temperature: float = 0.2) -> str:  
        context = "\n\n".join([  
            f"[Source: {c['filename']} | Page: {c['page']} | Chunk: {c['chunk_index']}]\n{c['text']}"  
            for c in context_chunks  
        ])

        messages = [  
            {
                "role": "system",  
                "content": (  
                    "You are a helpful assistant analyzing multiple PDF documents. "  
                    "The context below contains chunks from MULTIPLE different documents — each labeled with its source filename. "  
                    "When answering, address EACH document separately by name. "  
                    "If a document has insufficient context, say so explicitly rather than repeating filler text. "  
                    "Never fabricate information not present in the context."  
                ),  
            },  
            {
                "role": "user",  
                "content": f"Context:\n{context}\n\nQuestion: {query}",  
            },  
        ]

        response = self.client.chat.completions.create(  
            model=CHAT_MODEL,  
            messages=messages,  
            temperature=temperature,  
        )

        answer = response.choices[0].message.content  
        log.info("[GENERATOR] ✅ Answer generated")  
        return answer  
