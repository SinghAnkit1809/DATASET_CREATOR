**DATASET_CREATOR**
=====================

**Purpose and Functionality**
-----------------------------

The DATASET_CREATOR is a Python-based application designed to create high-quality datasets from PDF files. The project's primary goal is to provide an efficient and user-friendly interface for generating diverse question types (factual, conceptual, analytical) from PDF files, maintaining page and chunk context for better coherence.

**Key Features and Components**
-------------------------------

### Interface

The application features a user-friendly interface built using Gradio, allowing users to upload PDF files, adjust chunk size and context window size, and generate datasets.

### PDF Processing

The project utilizes PyPDF2 to extract text from PDF files, and then applies chunking and LLM response parsing to create question-answer pairs.

### LLM Context Window

The application allows users to configure the LLM context window size, enabling the generation of more accurate and context-aware question-answer pairs.

### Dataset Generation

The project creates a comprehensive dataset in JSON format, including metadata and question-answer pairs, which can be downloaded for further analysis or use.

**Setup and Usage Instructions**
---------------------------------

### Prerequisites

* Python 3.x (recommended)
* Gradio installed (`pip install gradio`)
* PyPDF2 installed (`pip install PyPDF2`)
* Groq installed (`pip install groq`)
* Python-dotenv installed (`pip install python-dotenv`)

### Setup

1. Clone the repository: `git clone https://github.com/SinghAnkit1809/DATASET_CREATOR.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your Groq API key: `GROQ_API_KEY = "your_api_key"`

### Usage

1. Run the application: `python app.py`
2. Upload a PDF file and adjust chunk size and context window size as needed
3. Click the "Create Dataset" button to generate the dataset
4. Download the dataset in JSON format

**Technical Details**
--------------------

### Code Structure

The project consists of the following components:

* `app.py`: The main application file, responsible for creating the Gradio interface and processing PDF files
* `components/`: A directory containing helper functions for PDF processing, chunking, and LLM response parsing
* `.env`: A file storing the Groq API key
* `requirements.txt`: A file listing project dependencies

### Dependencies

* Gradio: A Python library for building user interfaces
* PyPDF2: A Python library for reading and extracting information from PDF files
* Groq: A library for interacting with the Groq API
* Python-dotenv: A library for loading environment variables from a `.env` file

**License**
----------

This project is licensed under the Apache License 2.0. See `LICENSE` for details.