import gradio as gr
import json
import PyPDF2  # Consider switching to 'pypdf'
from groq import Groq  # Ensure this library is correct and installed
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv
import re
from dataclasses import dataclass
import tempfile

load_dotenv()

@dataclass
class PageChunk:
    text: str
    page_number: int
    chunk_number: int
    total_chunks: int

class DatasetCreator:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.context_window = 2048

    def extract_text_from_pdf(self, pdf_file) -> List[Tuple[int, str]]:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            pages = []
            for i in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[i]
                page_text = page.extract_text()
                page_text = re.sub(r'\s+', ' ', page_text)
                page_text = page_text.strip()
                if page_text:
                    pages.append((i + 1, page_text))  # Page numbers start from 1
            return pages
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return []

    def create_chunks(self, pages: List[Tuple[int, str]], chunk_size: int) -> List[PageChunk]:
        chunks = []
        for page_num, page_text in pages:
            words = page_text.split()
            current_chunk = []
            current_size = 0
            chunk_num = 1
            total_chunks = -(-len(words) // (chunk_size // 5))  # Adjust divisor based on word length estimation
            
            for word in words:
                word_size = len(word) + 1  # Account for space
                if current_size + word_size > chunk_size and current_chunk:
                    chunks.append(PageChunk(
                        text=' '.join(current_chunk),
                        page_number=page_num,
                        chunk_number=chunk_num,
                        total_chunks=total_chunks
                    ))
                    current_chunk = [word]
                    current_size = word_size
                    chunk_num += 1
                else:
                    current_chunk.append(word)
                    current_size += word_size
            
            if current_chunk:
                chunks.append(PageChunk(
                    text=' '.join(current_chunk),
                    page_number=page_num,
                    chunk_number=chunk_num,
                    total_chunks=total_chunks
                ))
        
        return chunks

    def parse_llm_response(self, content: str) -> List[Dict]:
        try:
            content = re.sub(r'```json\s*|\s*```', '', content)
            content = content.strip()
            if not content.startswith('['):
                content = f'[{content}]'
            if content.endswith(',]'):
                content = content[:-1] + ']'
            qa_pairs = json.loads(content)
            if not isinstance(qa_pairs, list):
                raise json.JSONDecodeError("Not a list", content, 0)
            return qa_pairs
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return [{
                "question": "What is the main topic discussed in this text?",
                "answer": content[:150] + "...",
                "difficulty": 3,
                "type": "factual",
                "tags": ["auto-generated"],
                "metadata": {"error": "JSON parsing failed"}
            }]

    def create_qa_pairs(self, chunks: List[PageChunk]) -> List[Dict]:
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

                response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",  # Ensure model name is correct
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at creating question-answer pairs for fine tunning. Always respond with valid JSON arrays containing QA pairs."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=self.context_window
                )

                content = response.choices[0].message.content
                chunk_qa_pairs = self.parse_llm_response(content)
                
                if chunk_qa_pairs:
                    qa_pairs.extend(chunk_qa_pairs)
                    print(f"Generated {len(chunk_qa_pairs)} pairs from chunk {i}")
                
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                continue

        return qa_pairs

def create_interface():
    creator = DatasetCreator(os.getenv("GROQ_API_KEY"))

    def process_pdf(pdf_file, chunk_size, context_size):
        try:
            if pdf_file is None:
                return {"error": "Please upload a PDF file"}, None
                
            creator.context_window = context_size
            
            pages = creator.extract_text_from_pdf(pdf_file)
            if not pages:
                return {"error": "Could not extract text from PDF"}, None

            chunks = creator.create_chunks(pages, chunk_size)
            qa_pairs = creator.create_qa_pairs(chunks)
            
            if not qa_pairs:
                return {"error": "No QA pairs could be generated"}, None

            dataset = {
                "metadata": {
                    "total_pairs": len(qa_pairs),
                    "total_pages": len(pages),
                    "chunk_size": chunk_size,
                    "context_window": context_size
                },
                "qa_pairs": qa_pairs
            }
            
            # Create a temporary file for download
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
                json.dump(dataset, tmp_file, indent=2)
                tmp_file_path = tmp_file.name

            return dataset, tmp_file_path

        except Exception as e:
            return {"error": f"Processing failed: {str(e)}"}, None

    with gr.Blocks() as app:
        gr.Markdown("# Enhanced PDF to QA Dataset Creator")
        
        with gr.Row():
            pdf_input = gr.File(label="Upload PDF")
            
        with gr.Row():
            chunk_size = gr.Slider(
                minimum=500,
                maximum=2000,
                value=1000,
                step=100,
                label="Text Chunk Size"
            )
            context_size = gr.Slider(
                minimum=1024,
                maximum=4096,
                value=2048,
                step=256,
                label="LLM Context Window Size"
            )
        
        create_button = gr.Button("Create Dataset")
        
        preview = gr.JSON(label="Dataset Preview")
        download_btn = gr.File(
            label="Download Dataset",
            file_count="single",
            type="filepath",
            visible=False
        )
        
        gr.Markdown("""
        ## Features:
        - Generates diverse question types (factual, conceptual, analytical)
        - Maintains page and chunk context for better coherence
        - Configurable chunk size and context window
        - Complete dataset available for download
        - Progress shown in terminal
        """)
        
        create_button.click(
            fn=process_pdf,
            inputs=[pdf_input, chunk_size, context_size],
            outputs=[preview, download_btn]
        )
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch()