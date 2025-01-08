from typing import List, Tuple
import PyPDF2
import re 

def extract_text_from_pdf(pdf_file) -> List[Tuple[int, str]]:
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