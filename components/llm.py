from typing import List,Dict
from components.chunking import PageChunk
from components.parse_llm_response import parse_llm_response
from groq import Groq

def create_qa_pairs(chunks: List[PageChunk], api_key: str, context_window = 2048) -> List[Dict]:
        client = Groq(api_key=api_key)
        context_window = context_window
        qa_pairs = []
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{total_chunks} from page {chunk.page_number}")
            try:
                prompt = f"""
                Generate question-answer pairs from this text chunk.
                This is chunk {chunk.chunk_number} of {chunk.total_chunks} from page {chunk.page_number}.

                Create possible user asked diverse questions covering the content can be upto 9 to 10 QA pairs. Include:
                1. Basic factual questions
                2. Conceptual understanding questions
                3. Analysis or application questions
                4. Each question should be relevant based on the provided context and do not try to add extra in answer by yourself.

                Required format for each QA pair (MUST be valid JSON):
                {{
                    "question": "Clear, specific question",
                    "answer": "Concise answer (50-100 words)",
                    "difficulty": "number 1-5",
                    "type": "factual/conceptual/analytical",
                    "tags": ["relevant", "topic", "tags"],
                    "metadata": {{
                        "page": {chunk.page_number},
                        "chunk": {chunk.chunk_number}
                    }}
                }}

                Text to process: {chunk.text}

                Return only the JSON array with QA pairs, no additional text.
                """

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",  # Ensure model name is correct
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at creating question-answer pairs for fine tunning. Always respond with valid JSON arrays containing QA pairs."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=context_window
                )

                content = response.choices[0].message.content
                chunk_qa_pairs = parse_llm_response(content)
                
                if chunk_qa_pairs:
                    qa_pairs.extend(chunk_qa_pairs)
                    print(f"Generated {len(chunk_qa_pairs)} pairs from chunk {i}")
                
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                continue

        return qa_pairs