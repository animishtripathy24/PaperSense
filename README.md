# PaperSense

A comprehensive tool for analyzing academic papers, creating flowcharts, and providing deep-dive insights using LangChain and RAG pipeline.

## Features

- **PDF Text Extraction**: Extract text from uploaded PDF research papers.
- **Chat with Paper**: Ask questions about the paper and get context-aware responses.
- **Plagiarism Check**: Verify that plagiarism is less than 10%.
- **Flowchart Generation**: Automatically create flowcharts from paper structure.
- **Research Highlights**: Auto-generate TL;DR with key contributions, novel findings, and applications.
- **Similar Paper Matching**: Find similar papers using FAISS vector search with TF-IDF embeddings.
- **Paper Translation**: Translate academic papers into multiple languages.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/paper-sense.git
   cd paper-sense
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your TogetherAI API key:
   ```plaintext
   TOGETHER_API_KEY="your-together-api-key-here"
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (usually http://localhost:8501).

3. Upload a PDF research paper using the file uploader.

4. Use the different tabs to:
   - View paper analysis and check for plagiarism.
   - Chat with the paper by asking questions.
   - Generate visualizations like flowcharts and concept maps.
   - Extract research highlights.
   - Find similar papers.
   - Translate the paper to different languages.

## Requirements

- Python 3.8+
- TogetherAI API key
- See `requirements.txt` for all Python dependencies.

## How It Works

The application uses:
- **LangChain**: For creating a RAG (Retrieval-Augmented Generation) pipeline.
- **FAISS**: For vector similarity search.
- **TF-IDF Vectorization**: To create vector representations of text.
- **TogetherAI Mixtral Model**: For language model capabilities.
- **Streamlit**: For the web interface.
- **PyPDF2**: For PDF text extraction.
- **Graphviz & Plotly**: For visualization generation.

## License

MIT License (No specific license file found)

## Version Information

Version: 1  
Last Updated: 2025-03-25T10:01:14.433Z  

## Release Notes

### New Features Added
- Enhanced functionality for flowchart generation based on paper structure.

### Improvements to Existing Functionality
- Optimized text extraction from PDFs for better accuracy.

### Release Date 
Released on March 25, 2025.