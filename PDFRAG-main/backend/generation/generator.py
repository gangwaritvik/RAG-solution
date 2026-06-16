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

    def generate(self, query: str, context_chunks: list, temperature: float = 0.2) -> tuple:  
        """
        Generate answer and memory summary from query and context.
        
        Returns:
            tuple: (answer, memory_summary)
                - answer: Full detailed response for user
                - memory_summary: Compressed key points for conversation memory
        """
        context = "\n\n".join([  
            f"[Source: {c['filename']} | Page: {c['page']} | Chunk: {c['chunk_index']}]\n{c['text']}"  
            for c in context_chunks  
        ])

        messages = [  
            {
                "role": "system",  
                "content": (  
                    "You are a precise, expert assistant analysing document content.\n"
                    "IMPORTANT: Provide TWO sections:\n\n"
                    "1. ANSWER: Full detailed response for the user\n"
                    "   - Use ## headings to separate major sections\n"  
                    "   - Use **bold** for key terms\n"  
                    "   - Use bullet points (- item) for lists, never plain paragraphs\n"  
                    "   - Always include a | Column | Column | markdown table at the end\n"  
                    "   - Never write walls of text — break everything into points\n"  
                    "   - If information is missing from context, state it clearly in one line\n"  
                    "   - Never fabricate. Stick strictly to the provided document context\n" 
                    "   - Scan the ENTIRE provided context thoroughly before answering\n"  
                    "   - Do not stop listing items early — include EVERY quantity mentioned\n"  
                    "   - If the same quantity appears in multiple chunks, list only once\n\n"
                    "2. MEMORY_SUMMARY: Brief compressed version with ONLY key points\n"
                    "   - Bullet format: key fact 1; key fact 2; key fact 3\n"
                    "   - No explanations, just facts\n"
                    "   - Must be under 150 characters\n\n"
                    "Format your response exactly as:\n"
                    "ANSWER:\n[Your full answer here]\n\n"
                    "MEMORY_SUMMARY:\n[Key points only, 1-2 lines]"
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

        full_response = response.choices[0].message.content  
        answer, memory_summary = self._parse_answer_and_summary(full_response)
        
        log.info("[GENERATOR] ✅ Answer generated with memory summary")  
        return answer, memory_summary
    
    def _parse_answer_and_summary(self, response: str) -> tuple:
        """
        Parse LLM response into answer and memory summary.
        
        Expected format:
        ANSWER:
        [full answer content]
        
        MEMORY_SUMMARY:
        [summary content]
        """
        try:
            # Split by MEMORY_SUMMARY marker
            if "MEMORY_SUMMARY:" in response:
                parts = response.split("MEMORY_SUMMARY:")
                answer = parts[0].replace("ANSWER:", "").strip()
                memory_summary = parts[1].strip()
            else:
                # Fallback: use entire response as answer, create summary from it
                answer = response.replace("ANSWER:", "").strip()
                memory_summary = self._extract_key_points(answer)
            
            # Validate lengths
            if not answer:
                answer = "Unable to generate answer from context."
            if not memory_summary:
                memory_summary = answer[:150]
            
            return answer, memory_summary
            
        except Exception as e:
            log.error(f"[GENERATOR] ❌ Failed to parse response: {e}", exc_info=True)
            # Return full response as answer, first 150 chars as summary
            return response, response[:150]
    
    def _extract_key_points(self, text: str, max_length: int = 150) -> str:
        """
        Extract key points from text as fallback summary method.
        
        Args:
            text: The text to extract from
            max_length: Maximum length of summary
            
        Returns:
            Extracted key points
        """
        import re
        
        # Find bullet points
        lines = text.split('\n')
        bullets = [line.strip() for line in lines if line.strip().startswith('-') or line.strip().startswith('•')]
        
        if bullets:
            # Use first 2-3 bullets
            summary = "; ".join(bullets[:3])
        else:
            # Use first 2 sentences
            sentences = re.split(r'[.!?]+', text)
            summary = ". ".join([s.strip() for s in sentences[:2] if s.strip()]) + "."
        
        # Limit length
        if len(summary) > max_length:
            summary = summary[:max_length].rsplit(' ', 1)[0] + "..."
        
        return summary  
