# dataset_creator.py
import gradio as gr
import json
import PyPDF2
from groq import Groq
from typing import List, Dict
import os
from dotenv import load_dotenv
import re

load_dotenv()

class DatasetCreator:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = []
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                page_text = re.sub(r'\s+', ' ', page_text)
                page_text = page_text.strip()
                text.append(page_text)
            return '\n'.join(text)
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return ""

    def create_qa_pairs(self, text: str, chunk_size: int = 1000) -> List[Dict]:
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        qa_pairs = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{total_chunks}")
            try:
                response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are an expert at creating comprehensive question-answer pairs for training datasets."},
                        {"role": "user", "content": f"""
                        Create 2-3 question-answer pairs from this text that would be suitable for training an AI model.
                        Include different types of questions (factual, conceptual, analytical).
                        
                        For each pair, include:
                        1. A specific question
                        2. A detailed answer with context
                        3. A difficulty rating (1-5)
                        4. Relevant topic tags
                        
                        Format as a JSON array like:
                        [
                            {{
                                "question": "specific question",
                                "answer": "detailed answer",
                                "difficulty": 3,
                                "tags": ["topic1", "topic2"]
                            }}
                        ]

                        Text: {chunk}
                        """}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )

                content = response.choices[0].message.content.strip()
                # Remove any markdown formatting
                content = content.replace('```json', '').replace('```', '').strip()
                
                chunk_qa_pairs = json.loads(content)
                qa_pairs.extend(chunk_qa_pairs)
                print(f"Successfully generated {len(chunk_qa_pairs)} pairs from chunk {i}")
                
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                continue

        print(f"Total QA pairs generated: {len(qa_pairs)}")
        return qa_pairs

    def save_dataset(self, qa_pairs: List[Dict], output_file: str = "dataset.json"):
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
            return output_file
        except Exception as e:
            print(f"Error saving dataset: {e}")
            return None

def create_interface():
    creator = DatasetCreator(os.getenv("GROQ_API_KEY"))

    def process_pdf(pdf_file, chunk_size):
        if pdf_file is None:
            return "Please upload a PDF file", None
            
        # Extract text from PDF
        text = creator.extract_text_from_pdf(pdf_file)
        if not text:
            return "Error: Could not extract text from PDF", None

        # Create QA pairs
        qa_pairs = creator.create_qa_pairs(text, chunk_size)
        
        # Check if we have any successful pairs
        if qa_pairs and len(qa_pairs) > 0:
            # Save dataset
            output_file = "dataset.json"
            creator.save_dataset(qa_pairs, output_file)
            
            # Create preview
            preview = {
                "total_pairs_generated": len(qa_pairs),
                "sample_pairs": qa_pairs[:3]  # Show first 3 pairs as preview
            }
            return json.dumps(preview, indent=2), output_file
        else:
            return "No QA pairs could be generated. Please try with a different chunk size or check the PDF content.", None

    # Create Gradio interface
    with gr.Blocks() as app:
        gr.Markdown("# PDF to QA Dataset Creator")
        
        with gr.Row():
            pdf_input = gr.File(label="Upload PDF")
            chunk_size = gr.Slider(
                minimum=500,
                maximum=2000,
                value=1000,
                step=100,
                label="Text Chunk Size"
            )
        
        create_button = gr.Button("Create Dataset")
        
        with gr.Row():
            output_text = gr.TextArea(label="Generated Dataset Preview")
            output_file = gr.File(label="Download Complete Dataset")
        
        gr.Markdown("""
        ## Notes:
        - The preview shows the first 3 QA pairs and total count
        - The complete dataset is available for download
        - If processing fails for some chunks, successful pairs will still be saved
        """)
        
        create_button.click(
            fn=process_pdf,
            inputs=[pdf_input, chunk_size],
            outputs=[output_text, output_file]
        )
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch()