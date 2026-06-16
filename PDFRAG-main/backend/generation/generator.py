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
                    "You are a precise, expert assistant analysing document content. "  
                    "Formatting rules:\n"   
                    "- Use ## headings to separate major sections.\n"  
                    "- Use **bold** for key terms.\n"  
                    "- Use bullet points (- item) for lists, never plain paragraphs.\n"  
                    "- Always include a | Column | Column | markdown table at the end.\n"  
                    "- Never write walls of text — break everything into points.\n"  
                    "- If information is missing from context, state it clearly in one line.\n"  
                    "- Never fabricate. Stick strictly to the provided document context.\n" 
                    "- Scan the ENTIRE provided context thoroughly before answering.\n"  
                    "- Do not stop listing items early — include EVERY quantity mentioned across ALL source chunks.\n"  
                    "- If the same quantity appears in multiple chunks, list it only once with the most complete information.\n"  

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
