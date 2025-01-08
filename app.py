import gradio as gr
import json
import os
from dotenv import load_dotenv
import tempfile

from components.process_pdf import extract_text_from_pdf
from components.chunking import create_chunks
from components.parse_llm_response import parse_llm_response
from components.llm import create_qa_pairs

load_dotenv()

def create_interface():
    api_key = os.getenv("GROQ_API_KEY")

    def process_pdf(pdf_file, chunk_size, context_size):
        try:
            if pdf_file is None:
                return {"error": "Please upload a PDF file"}, None
                
            context_window = context_size
            
            pages = extract_text_from_pdf(pdf_file)
            if not pages:
                return {"error": "Could not extract text from PDF"}, None

            chunks = create_chunks(pages, chunk_size)
            qa_pairs = create_qa_pairs(chunks,api_key,context_window)
            
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