# PaperSense

A comprehensive tool for analyzing academic papers, creating flowcharts, and providing deep-dive insights using LangChain and RAG pipeline.

## Features

- **PDF Text Extraction**: Efficiently extracts text from uploaded PDF research papers using `PyPDF2`.
- **Interactive Chat with Papers**: Users can ask questions about the paper to receive context-aware responses, enhancing comprehension.
- **Plagiarism Check**: Uses AI to ensure that plagiarism is less than 10%, assessing originality through detailed analysis.
- **Flowchart Generation**: Automatically generates visualizations of the paper's structure and methodology utilizing Graphviz.
- **Research Highlights Extraction**: Summarizes key contributions, novel findings, and practical applications into concise, actionable insights.
- **Similar Paper Matching**: Utilizes FAISS vector search along with TF-IDF embeddings to identify similar papers based on content analysis.
- **Paper Translation Support**: Provides functionality for translating academic papers into multiple languages (specific implementation not shown in provided files).

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
   SERPAPI_KEY="your-serpapi-key-here"
   PROJECT_TOKEN="your-project-token-here"
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (usually http://localhost:8501).

3. Upload a PDF research paper using the file uploader.

4. Use the different tabs to access functionalities such as:
    - Analyzing paper content.
    - Checking for plagiarism issues.
    - Engaging in interactive Q&A with the paper's content.
    - Visualizing data through flowcharts.
    - Extracting highlights and finding similar research.

## Requirements

- Python 3.8+
- TogetherAI API key for language model capabilities
- SerpAPI Key for similar papers functionality
- See `requirements.txt` for all Python dependencies.

## How It Works

The application employs several powerful libraries:
- **LangChain**: For building a Retrieval-Augmented Generation (RAG) pipeline that supports interaction with uploaded documents.
- **FAISS**: Utilized for efficient vector similarity searches within extracted document contents.
- **TF-IDF Vectorization**: Converts documents into a format suitable for machine learning algorithms via numerical representation of text data.
- **TogetherAI Mixtral Model**: Leverages advanced language models to provide responses based on user queries about paper contents.
- **Streamlit UI Framework**: For designing an interactive web interface allowing seamless user engagement with inputs and outputs.

## Analytics Tracking

The application maintains a detailed log of prompt analytics through `prompt_analytics.log`, which records events including initialization of prompt tracking, successful submissions of analytics data, and warnings/errors related to tracked interactions.

### Log Example:
```
2025-03-25 17:39:03,254 - pypeprompts.main - INFO - Analytics data submitted successfully. Response: {"message":"Analytics data saved successfully"}
...
```

## License

MIT License (No specific license file found)

## Version Information

Version: 1  
Last Updated: 2025-03-27T10:52:02Z  

## Release Notes

### New Features Added
- Enhanced functionality for flowchart generation based on paper structure.

### Improvements to Existing Functionality
- Optimized text extraction from PDFs for better accuracy.

### Release Date 
Released on March 25, 2025.

### Key Files Modified Since Last Update
1. `app.py`: Revised processing logic supporting new features added in this version including enhanced error handling and logging mechanisms related to prompt analytics captured in `prompt_analytics.log`.

2. `prompt_analytics.log`: Updated to dynamically capture submission data that helps improve future iterations through analyzed performance metrics.