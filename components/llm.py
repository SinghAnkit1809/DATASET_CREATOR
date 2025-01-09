from typing import List, Dict
from components.chunking import PageChunk
from components.parse_llm_response import parse_llm_response
from groq import Groq

def create_qa_pairs(chunks: List[PageChunk], api_key: str, context_window = 2048) -> List[Dict]:
        client = Groq(api_key=api_key)
        qa_pairs = []
        total_chunks = len(chunks)
        
        # Keep track of current context just for reference
        current_context = {
            "subjects_discussed": set(),
            "organizations": set(),
            "places": set()
        }
        
        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{total_chunks} from page {chunk.page_number}")
            try:
                # Extract any names/entities from the current chunk
                # We're keeping this simple with basic text analysis
                chunk_lines = chunk.text.split('\n')
                words = chunk.text.split()
                capitalized_words = [word for word in words if word[0].isupper()] if words else []
                
                # Update context with any new capitalized terms
                for word in capitalized_words:
                    if len(word) > 1:  # Avoid single letters
                        current_context["subjects_discussed"].add(word)

                prompt = f"""
                Generate question-answer pairs from this text chunk.
                This is chunk {chunk.chunk_number} of {chunk.total_chunks} from page {chunk.page_number}.

                Important: When creating questions and answers, if you refer to any person, organization, or place, 
                always use their specific name from the text. Never use generic terms like "someone", "the person", 
                or "the organization" when a specific name is available in the text.

                Create possible user asked diverse questions covering the content (up to 9-10 QA pairs). Include:
                1. Basic factual questions
                2. Conceptual understanding questions
                3. Analysis or application questions
                4. Each question should be relevant based on the provided context and do not try to add extra in answer by yourself.
                5. When creating questions, ensure they clearly indicate the specific names of people, organizations, or places mentioned in the text.
                6. If multiple people or organizations are mentioned, make it clear which specific entity you're asking about.

                Required format for each QA pair (MUST be valid JSON):
                {{
                    "question": "Clear, specific question using actual names of people/organizations",
                    "answer": "Concise answer (50-100 words) using actual names of people/organizations",
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
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at creating question-answer pairs for fine tunning. Always respond with valid JSON arrays containing QA pairs. Always use specific names of people and organizations instead of generic terms."
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