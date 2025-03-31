# 📚 PaperSense

Your AI-Powered Research Companion that transforms complex academic papers into actionable insights.

![PaperSense Demo](https://i.imgur.com/sample_image.png)

## 🌟 Overview

PaperSense is an innovative application that helps researchers, students, and professionals quickly understand and extract value from academic research papers. Using advanced AI techniques, PaperSense analyzes PDF research papers to provide comprehensive insights, interactive Q&A, visual representations, and practical applications of research findings.

## ✨ Key Features

- **💬 Interactive Paper Chat**: Ask questions about any aspect of the paper and get contextually relevant answers from the AI.
- **📑 Comprehensive Paper Analysis**: Extract core concepts, methodology, and findings with AI-powered analysis.
- **📊 Visual Flowcharts**: Generate structured flowcharts representing the paper's methodology and logical structure.
- **🔍 Research Highlights**: Quickly grasp key takeaways and business applications of academic research.
- **📚 Similar Papers Discovery**: Find and explore related publications to broaden understanding of the topic.
- **💼 Startup Insights**: Generate detailed step-by-step implementation roadmaps for transforming research into real-world businesses.
- **🔗 Citation Analysis**: Explore key references and understand how they support the paper's arguments.
- **🔎 Plagiarism Detection**: Evaluate the originality of papers with section-by-section plagiarism analysis.
- **🧪 What-If Scenarios**: Explore alternative research methodologies by modifying key parameters or approaches.

## 🛠️ Technical Stack

- **Frontend**: Streamlit for interactive web interface
- **Natural Language Processing**: OpenAI's GPT-4o-mini for text analysis and generation
- **Vector Storage**: FAISS for efficient similarity search
- **Text Processing**: TF-IDF Vectorization for document embeddings
- **PDF Processing**: PyPDF2 for text extraction
- **Visualization**: QuickChart API for flowchart generation
- **External Integration**: SerpAPI for similar paper searches

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/animishtripathy24/PaperSense.git
   cd PaperSense
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your API keys:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   SERPAPI_KEY=your-serpapi-key-here
   PROJECT_TOKEN=your-prompt-analytics-token-here
   ```

## 📋 Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Upload your academic paper(s) in PDF format using the sidebar.

3. Click "Process Papers" to analyze the documents.

4. Use the chat interface to ask specific questions about the paper.

5. Explore the various analysis tools available in the tabs:
   - 📑 Paper Analysis
   - 📊 Research Flowchart
   - 🔍 Research Highlights
   - 📚 Similar Papers
   - 💼 Startup Insights
   - 🧪 What-If Scenarios

## ⚙️ Application Workflow

1. **Document Upload**: Users upload PDF research papers through the Streamlit interface.
2. **Text Extraction**: PyPDF2 extracts text content from the PDFs.
3. **Text Chunking**: Content is divided into manageable chunks for processing.
4. **Vector Embedding**: TF-IDF vectorization creates embeddings for efficient retrieval.
5. **Interactive Analysis**: Users interact with the paper's content through various analysis tools.
6. **AI-Powered Insights**: OpenAI's GPT-4o-mini generates contextual responses and insights.

## 📦 Requirements

- Python 3.8+
- OpenAI API key
- SerpAPI key (for similar papers functionality)
- Internet connection for external API calls

## 🔮 Future Enhancements

- Multi-language support for paper analysis
- Advanced visualization of citation networks
- Collaborative research tools for team environments
- Mobile application for on-the-go research analysis
- Integration with reference management systems

## 👥 Contributors

- Animish Tripathy

## 📄 License

[MIT License](LICENSE)

## 🙏 Acknowledgments

- This project uses various open-source libraries and APIs to provide its functionality.
- Special thanks to the developers of Streamlit, LangChain, FAISS, and other tools that make this application possible. 