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

5. Explore various analysis tools available in different tabs:
   - 📑 Paper Analysis 
   - 📊 Research Flowchart 
   - 🔍 Research Highlights 
   - 📚 Similar Papers 
   - 💼 Startup Insights 
   - 🧪 What-If Scenarios 

## ⚙️ Application Workflow

1. **Document Upload**: Users upload PDF research papers through the Streamlit interface.
2. **Text Extraction & Processing**: PyPDF2 extracts text content from PDFs efficiently; content is then divided into manageable chunks using TF-IDF vectorization to create embeddings for effective retrieval.
3. **Interactive Analysis & Insights Generation**: Users interact with paper content through various tools while OpenAI's GPT model provides contextual responses.

## 📦 Requirements

- Python 3.8+
- OpenAI API key (for NLP features)
- SerpAPI key (for similar paper functionalities)

An internet connection is required due to external API calls.

## 🔮 Future Enhancements

Future updates may include:

- Multi-language support for broader accessibility in paper analysis.
- Advanced visualization features such as citation networks to better illustrate connections between works.
- Collaborative tools enabling team environments within research settings.
- A mobile application to facilitate on-the-go access to analytical features.

## 👥 Contributors

### Maintainer:
* Animish Tripathy *(Main Contributor)*

## 📄 License

This project is licensed under an [MIT License](LICENSE).

## 📝 Release Notes

### Version 1  
Released on 2025-03-31T14:31:51.990Z  

#### New Features Added:
  - Enhanced interactive chat functionality allowing deeper engagement with uploaded papers.
  
#### Improvements Made:
  - Improved efficiency in document processing within `app.py`, leading to faster response times during analyses.

### Acknowledgments 

Special thanks to all open-source libraries utilized in this project including but not limited to Streamlit, LangChain, FAISS, PyPDF2, and others which contribute significantly towards enhancing this application's capabilities.
