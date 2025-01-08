from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class PageChunk:
    text: str
    page_number: int
    chunk_number: int
    total_chunks: int

def create_chunks(pages: List[Tuple[int, str]], chunk_size: int) -> List[PageChunk]:
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