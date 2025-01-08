from typing import List, Dict
import json
import re

def parse_llm_response(content: str) -> List[Dict]:
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