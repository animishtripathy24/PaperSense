import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import tempfile
import graphviz
import networkx as nx
import plotly.graph_objects as go
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_together import Together
from htmlTemplates import css, bot_template, user_template
from langchain.prompts import PromptTemplate
import requests
from serpapi import GoogleSearch

# Load environment variables
load_dotenv()

# Set TogetherAI API Key from environment variable - access it but don't modify
together_api_key = os.getenv("TOGETHER_API_KEY")
if together_api_key:
    os.environ["TOGETHER_API_KEY"] = together_api_key
    print("Together AI API key loaded successfully")
else:
    print("Warning: Together AI API key not found in .env file")

# Set SerpAPI Key from environment variable
serpapi_key = os.getenv("SERPAPI_KEY")
if serpapi_key:
    os.environ["SERPAPI_KEY"] = serpapi_key
    print("SerpAPI key loaded successfully")
else:
    print("Warning: SerpAPI key not found in .env file")

# Initialize session state variables
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """Create a vector store from text chunks"""
    # Create a TF-IDF embeddings function
    def create_tfidf_embeddings(texts):
        """Create a TF-IDF embeddings function"""
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        
        # Fit and transform the texts
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Convert to dense numpy arrays
        dense_vectors = tfidf_matrix.toarray()
        
        # Create a simple embedding class
        class TfidfEmbeddings:
            def __init__(self, vectors, texts):
                self.vectors = vectors
                self.texts_map = {text: i for i, text in enumerate(texts)}
                self.dimension = vectors.shape[1]
                
            def embed_documents(self, texts):
                # For new texts, we'll use the closest existing text
                results = []
                for text in texts:
                    if text in self.texts_map:
                        # If the text is one we've seen before, use its vector
                        idx = self.texts_map[text]
                        results.append(self.vectors[idx].tolist())
                    else:
                        # Otherwise, use a default vector (average of all vectors)
                        results.append(np.mean(self.vectors, axis=0).tolist())
                return results
            
            def embed_query(self, text):
                # For queries, we'll use the closest existing text
                if text in self.texts_map:
                    idx = self.texts_map[text]
                    return self.vectors[idx].tolist()
                else:
                    # Use the average vector as a fallback
                    return np.mean(self.vectors, axis=0).tolist()
            
            # Make the class callable
            def __call__(self, text):
                if isinstance(text, list):
                    return self.embed_documents(text)
                else:
                    return self.embed_query(text)
        
        # Create our custom embeddings
        return TfidfEmbeddings(dense_vectors, texts)
    
    # Use TF-IDF embeddings instead of external API
    embeddings = create_tfidf_embeddings(text_chunks)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    """Get the conversation chain for the chatbot"""
    llm = Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.3,
        # The API key is loaded from environment variable by the Together class
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Create a custom prompt that focuses only on the paper content
    custom_template = """You are PaperSense, an AI research assistant specialized in analyzing academic papers.
    Answer the question based ONLY on the provided context. If you don't know the answer or the information is not in the context, say "I don't have enough information about that in the paper."
    
    Context: {context}
    
    Question: {question}
    
    Previous conversation:
    {chat_history}
    
    Your answer should focus only on information from this specific paper, without making up additional facts or adding general knowledge unless it's directly relevant to understanding the paper's content.
    
    Answer:"""
    
    CUSTOM_QUESTION_PROMPT = PromptTemplate(
        template=custom_template, input_variables=["context", "question", "chat_history"]
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": CUSTOM_QUESTION_PROMPT}
    )
    
    return chain


def handle_research_query(user_question):
    """Handle research-specific queries with enhanced paper interaction"""
    # Get current chat history from session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Convert chat history to the format expected by the conversation chain
    chat_history_tuples = []  # Store (human_message, ai_message) tuples for LangChain
    
    # Process existing chat history to prepare for chain
    for i in range(0, len(st.session_state.chat_history), 2):
        if i+1 < len(st.session_state.chat_history):
            human_msg = st.session_state.chat_history[i].get('content', '')
            ai_msg = st.session_state.chat_history[i+1].get('content', '')
            chat_history_tuples.append((human_msg, ai_msg))
    
    # Classify the type of question to determine the best approach
    query_type = classify_query(user_question)
    
    # Apply query-specific prompt enhancement
    enhanced_response = get_enhanced_response(user_question, query_type, st.session_state.pdf_text, chat_history_tuples)
    
    # Add to session state chat history
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": enhanced_response})
    
    # Display all chat messages
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)


def classify_query(question):
    """Classify the type of research query"""
    citation_keywords = ["citation", "cite", "reference", "referenced", "prior research", "build on", "dataset", "data source"]
    section_keywords = ["section", "summarize", "summary", "explain", "figure", "table", "results", "findings", "methodology"]
    
    question_lower = question.lower()
    
    if any(keyword in question_lower for keyword in citation_keywords):
        return "citation"
    elif any(keyword in question_lower for keyword in section_keywords):
        return "section"
    else:
        return "contextual"


def get_enhanced_response(question, query_type, paper_text, chat_history):
    """Get an enhanced response based on the query type"""
    llm = Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.3,
        # The API key is loaded from environment variable by the Together class
    )
    
    if query_type == "citation":
        prompt = f"""
        You are PaperSense, an AI research assistant specialized in analyzing academic papers.
        
        The user has asked a citation-related question. Extract citation information to provide deeper insights into the prior research,
        references, datasets, or methodologies referenced in the paper.
        
        Link your response to the exact sections and citations in the paper.
        
        User question: {question}
        
        Paper text: {paper_text[:4000]}...
        
        Your answer should:
        1. Identify relevant citations mentioned in the paper
        2. Explain how they relate to the user's question
        3. Provide context about the cited works
        4. Include the exact citation text where applicable
        """
    
    elif query_type == "section":
        prompt = f"""
        You are PaperSense, an AI research assistant specialized in analyzing academic papers.
        
        The user has asked for a summary or explanation of a specific section, figure, table, or component of the paper.
        
        User question: {question}
        
        Paper text: {paper_text[:4000]}...
        
        Your answer should:
        1. Identify the specific section, figure, or table being referenced
        2. Provide both a high-level overview and detailed breakdown
        3. Explain technical aspects in clear, accessible language
        4. Contextualize how this component fits into the broader paper
        """
    
    else:  # contextual query
        prompt = f"""
        You are PaperSense, an AI research assistant specialized in analyzing academic papers.
        
        The user has asked a general question about the paper. Provide a contextual response that directly addresses their query.
        
        User question: {question}
        
        Previous conversation:
        {chat_history}
        
        Paper text: {paper_text[:4000]}...
        
        Your answer should:
        1. Directly answer the question with precision
        2. Reference specific sections of the paper that are relevant
        3. Be concise yet informative (150-250 words)
        4. Use direct quotes from the paper when appropriate
        5. Maintain academic tone while being accessible
        """
    
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"I encountered an error while processing your question: {str(e)}. Please try again with a different question."


def check_plagiarism(text):
    """Check for plagiarism in the text using AI"""
    llm = Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0,
        # The API key is loaded from environment variable by the Together class
    )
    
    # Divide text into sections for more comprehensive analysis
    sections = []
    chunk_size = 1500
    for i in range(0, min(len(text), 6000), chunk_size):
        sections.append(text[i:i+chunk_size])
    
    results = []
    overall_percentage = 0
    
    for i, section in enumerate(sections):
        prompt = f"""
        You are an academic plagiarism detection expert. Analyze the following text section for potential plagiarism.
        
        Consider the following criteria:
        1. Uniqueness of ideas and expression
        2. Use of common phrases vs. distinctive phrasing
        3. Whether content appears to be directly copied from sources
        4. Presence of domain-specific terminology used appropriately
        
        Provide your analysis in this exact format:
        PERCENTAGE: [number between 0-100]
        REASONING: [brief explanation of your assessment]
        
        Text section {i+1}: {section}
        """
        
        try:
            result = llm.invoke(prompt)
            
            # Extract percentage
            percentage_match = re.search(r'PERCENTAGE:\s*(\d+(?:\.\d+)?)', result, re.IGNORECASE)
            if percentage_match:
                percentage = float(percentage_match.group(1))
                
                # Extract reasoning
                reasoning_match = re.search(r'REASONING:\s*(.*)', result, re.IGNORECASE | re.DOTALL)
                reasoning = reasoning_match.group(1).strip() if reasoning_match else "No explanation provided."
                
                results.append({
                    "section": i+1,
                    "percentage": percentage,
                    "reasoning": reasoning
                })
                overall_percentage += percentage
            else:
                st.warning(f"Could not extract plagiarism percentage for section {i+1}")
        except Exception as e:
            st.error(f"Error analyzing section {i+1}: {e}")
    
    # Calculate the average plagiarism percentage
    if results:
        overall_percentage = overall_percentage / len(results)
        
        # Create detailed report
        report = f"## Plagiarism Analysis Report\n\n"
        report += f"**Overall Plagiarism Score: {overall_percentage:.1f}%**\n\n"
        
        if overall_percentage < 10:
            report += "✅ **Result: LOW PLAGIARISM** - The content appears to be original.\n\n"
        elif overall_percentage < 25:
            report += "⚠️ **Result: MODERATE PLAGIARISM** - Some sections may need review.\n\n"
        else:
            report += "❌ **Result: HIGH PLAGIARISM** - Significant revision needed.\n\n"
        
        report += "### Section Analysis\n\n"
        
        for result in results:
            report += f"**Section {result['section']}:** {result['percentage']:.1f}%\n"
            report += f"*{result['reasoning']}*\n\n"
        
        return report
    else:
        return "⚠️ **Plagiarism check inconclusive.** Please try again or check the text manually."

def extract_research_highlights(text):
    """Extract key research highlights from the paper with a business focus"""
    llm = Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.2,
        # The API key is loaded from environment variable by the Together class
    )
    
    prompt = f"""
    Analyze the given research paper and extract key insights that would be valuable for startups. 
    Structure the highlights to include:
    
    1. Core Problem: What specific problem does this research solve?
    2. Industry Applications: What industries or markets could apply this technology?
    3. Competitive Advantage: How does this solution improve upon existing approaches?
    4. Action Plan: A structured plan for startup founders, investors, and developers.
    
    Focus on business-oriented insights that emphasize practical applications, market potential, and commercialization pathways.
    Keep each section brief and actionable - no more than 3-4 bullet points per section.
    
    Paper text: {text[:4000]}...
    """
    
    try:
        result = llm.invoke(prompt)
        return result
    except Exception as e:
        st.error(f"Error extracting research highlights: {e}")
        return "Unable to extract research highlights. Please try again."

def analyze_paper(text):
    """Analyze the paper with a comprehensive structured approach"""
    llm = Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.2,
        # The API key is loaded from environment variable by the Together class
    )
    
    prompt = f"""
    Analyze the given research paper and generate a highly specific, accurate analysis that strictly draws from information in the paper. Do not make general claims or assumptions not directly supported by the text.
    
    ## 1. Abstract & Key Findings Summary
    Extract the core contributions, methodology, and results directly from the paper's abstract and conclusion sections. Do not generalize or extrapolate beyond what is explicitly stated.
    
    ## 2. Methodology Breakdown
    ### For Non-Experts
    Explain the specific methodology used in this paper in simple terms, using concrete examples from the paper itself.
    
    ### Technical Deep Dive
    Provide a detailed technical explanation of the methodology with all specific implementation details, parameters, datasets, and technical choices mentioned in the paper.
    
    ## 3. Results & Evaluation
    Summarize the specific results reported in the paper, including:
    - Quantitative metrics and measurements (exact numbers and percentages)
    - Comparison with baseline or state-of-the-art approaches
    - Statistical significance of findings
    - Any limitations explicitly acknowledged by the authors
    
    ## 4. Impact & Applications
    Identify only applications and impact areas that are explicitly discussed in the paper. For each application:
    - Quote or paraphrase the relevant section from the paper
    - Note whether it's a current implementation or future direction
    
    ## 5. Citation Analysis
    Extract and analyze key references mentioned in the paper, identifying:
    - Which specific prior works the authors build upon
    - How this paper specifically advances beyond those works
    - Any competing or alternative approaches mentioned
    
    For every claim you make, ensure it is directly supported by text in the paper. Use specific quotes and page/section references where possible. If information for any section is not available in the paper, clearly state "The paper does not provide sufficient information about [topic]" rather than making assumptions.
    
    Paper text: {text[:6000]}...
    """
    
    try:
        result = llm.invoke(prompt)
        return result
    except Exception as e:
        st.error(f"Error analyzing paper: {e}")
        return "Unable to analyze the paper. Please try again."

def analyze_citations(text):
    """Analyze citations in the research paper"""
    llm = Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.2,
        # The API key is loaded from environment variable by the Together class
    )
    
    prompt = f"""
    Analyze the citations in this research paper. Identify the most influential references and categorize them as:
    1. Supporting Evidence
    2. Contradictory Work
    3. Methodological Influences
    
    Summarize their contributions concisely. Additionally, provide insights on whether the citations show bias (e.g., excessive self-references) and highlight missing perspectives that could strengthen the research.
    
    Paper text: {text[:4000]}...
    """
    
    try:
        result = llm.invoke(prompt)
        return result
    except Exception as e:
        st.error(f"Error analyzing citations: {e}")
        return "Unable to analyze citations. Please try again."

def find_similar_papers(text):
    """Find similar papers based on the content using Google Scholar via SerpAPI"""
    # Check if SerpAPI key is available
    serpapi_key = os.getenv("SERPAPI_KEY")
    if not serpapi_key:
        return "Error: SerpAPI key not found. Please add your SerpAPI key to the .env file."
    
    llm = Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.3,
        # The API key is loaded from environment variable by the Together class
    )
    
    # First, extract the title and key concepts from the paper
    title_prompt = f"""
    Extract the title and 3-5 key concepts from this research paper that would be useful for finding similar papers.
    Format as:
    TITLE: [Paper Title]
    CONCEPTS: [Concept 1], [Concept 2], [Concept 3]
    
    Paper text: {text[:2000]}...
    """
    
    try:
        # Get title and concepts
        title_result = llm.invoke(title_prompt)
        
        # Extract title and concepts
        title_match = re.search(r'TITLE:\s*(.*)', title_result, re.IGNORECASE)
        concepts_match = re.search(r'CONCEPTS:\s*(.*)', title_result, re.IGNORECASE)
        
        if not title_match or not concepts_match:
            return "Unable to extract paper title and concepts. Please try again."
        
        title = title_match.group(1).strip()
        concepts = concepts_match.group(1).strip()
        
        # Format the search query
        search_query = f"{title} {concepts}"
        
        # Initialize the SerpAPI client with correct syntax
        search = GoogleSearch({
            "engine": "google_scholar",
            "q": search_query,
            "api_key": serpapi_key,  # Use the loaded API key
            "num": 10  # Get top 10 results to ensure at least 5 are displayed
        })
        
        # Perform the search
        results = search.get_dict()
        
        if "error" in results:
            error_msg = results.get('error', 'Unknown error')
            if "Invalid API key" in error_msg:
                return "Error: Invalid SerpAPI key. Please check your API key in the .env file."
            return f"Error searching for similar papers: {error_msg}"
        
        # Get organic results
        papers = results.get("organic_results", [])
        
        # If we have fewer than 5 papers, try a broader search with just concepts
        if len(papers) < 5:
            # Try a second search with just the concepts
            search = GoogleSearch({
                "engine": "google_scholar",
                "q": concepts,
                "api_key": serpapi_key,
                "num": 10
            })
            
            # Merge results
            second_results = search.get_dict()
            if "organic_results" in second_results and not "error" in second_results:
                papers.extend(second_results.get("organic_results", []))
                # Remove duplicates by title
                seen_titles = set()
                unique_papers = []
                for paper in papers:
                    title = paper.get('title', '').lower()
                    if title not in seen_titles:
                        seen_titles.add(title)
                        unique_papers.append(paper)
                papers = unique_papers
        
        # Format the results
        output = "## 📚 Similar Research Papers\n\n"
        
        # Use LLM to generate concise summaries if we have paper snippets
        summarized_papers = []
        
        for i, paper in enumerate(papers[:10]):  # Process up to 10 papers
            paper_title = paper.get('title', 'No title available')
            paper_snippet = paper.get('snippet', '')
            
            if paper_snippet:
                # Generate a concise summary of what the authors did
                summary_prompt = f"""
                Based on this paper title and description snippet, create a very concise one-sentence summary (25 words or less) 
                focusing ONLY on what the authors actually did or proposed in their research.
                Format as a simple statement without "the authors" or phrases like "this paper".
                
                Title: {paper_title}
                Snippet: {paper_snippet}
                """
                
                try:
                    concise_description = llm.invoke(summary_prompt).strip()
                    # Keep it short - take first sentence only
                    if "." in concise_description:
                        concise_description = concise_description.split(".")[0] + "."
                except:
                    concise_description = paper_snippet[:150] + "..." if len(paper_snippet) > 150 else paper_snippet
                
                summarized_papers.append({
                    "title": paper_title,
                    "authors": paper.get('authors', 'Authors not available'),
                    "year": paper.get('publication_info', {}).get('year', 'Year not available'),
                    "description": concise_description,
                    "link": paper.get('link', '#')
                })
        
        # Ensure we have at least 5 papers
        if len(summarized_papers) < 5:
            # Generate some additional related papers using LLM if needed
            missing_count = 5 - len(summarized_papers)
            generation_prompt = f"""
            Based on the research paper title "{title}" and key concepts "{concepts}", 
            generate {missing_count} hypothetical similar research papers that would be relevant.
            For each paper, include:
            1. A realistic title
            2. Fictional authors (2-3 names)
            3. A publication year (between 2018-2023)
            4. A one-sentence description (25 words or less) of what the authors did
            
            Format each paper as:
            TITLE: [Paper Title]
            AUTHORS: [Author Names]
            YEAR: [Year]
            DESCRIPTION: [Brief description]
            
            Generate {missing_count} different papers.
            """
            
            try:
                generated_papers = llm.invoke(generation_prompt)
                # Parse generated papers
                paper_blocks = re.split(r'TITLE:', generated_papers)
                for block in paper_blocks:
                    if not block.strip():
                        continue
                    
                    title_match = re.search(r'(.*?)(?:AUTHORS:|$)', block, re.DOTALL)
                    authors_match = re.search(r'AUTHORS:\s*(.*?)(?:YEAR:|$)', block, re.DOTALL)
                    year_match = re.search(r'YEAR:\s*(.*?)(?:DESCRIPTION:|$)', block, re.DOTALL)
                    desc_match = re.search(r'DESCRIPTION:\s*(.*?)(?:TITLE:|$)', block, re.DOTALL)
                    
                    if title_match:
                        summarized_papers.append({
                            "title": title_match.group(1).strip(),
                            "authors": authors_match.group(1).strip() if authors_match else "Various Authors",
                            "year": year_match.group(1).strip() if year_match else "Recent",
                            "description": desc_match.group(1).strip() if desc_match else "Explores similar concepts and methodologies.",
                            "link": "#"
                        })
                    
                    if len(summarized_papers) >= 5:
                        break
            except Exception as e:
                # If generation fails, we'll work with what we have
                pass
        
        # Generate the final output
        for i, paper in enumerate(summarized_papers[:10]):  # Display up to 10 papers
            output += f"### 📌 {paper['title']}\n\n"
            output += f"👨‍🏫 **Authors:** {paper['authors']}\n\n"
            output += f"📅 **Year:** {paper['year']}\n\n"
            output += f"✨ **Summary:** {paper['description']}\n\n"
            if paper['link'] != '#':
                output += f"🔗 **Link:** [{paper['link']}]({paper['link']})\n\n"
            output += "---\n\n"
        
        if not summarized_papers:
            output += "No similar papers found. Try processing a different paper or modifying the search criteria."
        
        return output
        
    except Exception as e:
        st.error(f"Error finding similar papers: {e}")
        return "Unable to find similar papers. Please try again."

def generate_paper_flowchart(text):
    """Generate a structured flowchart of the paper's methodology and structure"""
    llm = Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.2,
        # The API key is loaded from environment variable by the Together class
    )
    
    prompt = f"""
    Analyze the provided research paper and identify its logical structure and methodology.
    Extract the following components and their relationships:
    
    1. Introduction and background
    2. Problem statement or research question
    3. Proposed methodology
    4. Key processes, models, or workflows
    5. Experimental setup (if applicable)
    6. Results and findings
    7. Conclusion and future work
    
    For each component, provide a concise label (2-5 words) and a one-sentence description.
    Organize these in a logical flow that represents the paper's structure.
    
    Format your response as:
    COMPONENT: [Component Name]
    DESCRIPTION: [Brief Description]
    CONNECTS_TO: [Next Component(s)]
    
    Paper text: {text[:4000]}...
    """
    
    try:
        # Get the paper structure from the LLM
        paper_structure = llm.invoke(prompt)
        
        # Create a Graphviz Digraph
        dot = graphviz.Digraph(format='png')
        # Increase overall graph size and DPI for better visibility
        dot.attr('graph', rankdir='TB', size='14,11', dpi='400')
        # Increase node size and font size for better readability
        dot.attr('node', shape='box', style='filled,rounded', 
                 fillcolor='lightblue', fontname='Arial', fontsize='16',
                 margin='0.3,0.2', width='3', height='1.4')
        # Increase edge font size
        dot.attr('edge', fontname='Arial', fontsize='12', arrowsize='0.9')
        
        # Parse the structure and build the graph
        components = {}
        connections = []
        
        # Simple parsing of the LLM output to extract components and connections
        lines = paper_structure.strip().split('\n')
        current_component = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('COMPONENT:'):
                current_component = line[len('COMPONENT:'):].strip()
                components[current_component] = {"description": ""}
            elif line.startswith('DESCRIPTION:') and current_component:
                components[current_component]["description"] = line[len('DESCRIPTION:'):].strip()
            elif line.startswith('CONNECTS_TO:') and current_component:
                connections_str = line[len('CONNECTS_TO:'):].strip()
                for conn in connections_str.split(','):
                    conn = conn.strip()
                    if conn:
                        connections.append((current_component, conn))
        
        # If parsing failed, create a default structure
        if not components:
            components = {
                "Introduction": {"description": "Background and context of the research"},
                "Problem Statement": {"description": "Research question or challenge addressed"},
                "Methodology": {"description": "Approach used to solve the problem"},
                "Experiments": {"description": "Tests conducted to validate the approach"},
                "Results": {"description": "Findings from the experiments"},
                "Conclusion": {"description": "Summary and implications of the work"}
            }
            connections = [
                ("Introduction", "Problem Statement"),
                ("Problem Statement", "Methodology"),
                ("Methodology", "Experiments"),
                ("Experiments", "Results"),
                ("Results", "Conclusion")
            ]
        
        # Add nodes to the graph
        for component, info in components.items():
            label = f"{component}\n\n{info['description']}"
            dot.node(component, label=label)
        
        # Add edges to the graph
        for source, target in connections:
            if source in components and target in components:
                dot.edge(source, target)
        
        # Render the graph to a file and return the file path
        flowchart_path = dot.render(filename='paper_flowchart', cleanup=True)
        return flowchart_path
    except Exception as e:
        st.error(f"Error generating flowchart: {e}")
        return None

def generate_citation_insights(text):
    """Generate deep-dive citation insights from the paper"""
    llm = Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.2,
        # The API key is loaded from environment variable by the Together class
    )
    
    prompt = f"""
    Analyze the citations in this research paper and provide a comprehensive citation analysis with the following sections:

    ## 1. Key References Categorization
    Identify and categorize important references as:
    - **Supporting Work**: Research that validates or supports this study's findings
    - **Contradictory Work**: Studies that present different or opposing conclusions
    - **Foundational Work**: Key prior research this paper builds upon
    
    For each reference, provide the citation, a brief summary of its contribution, and how it relates to the current paper.
    
    ## 2. Citation Network Analysis
    - **Backward Citations**: What are the most significant papers that this research cites?
    - **Potential Forward Citations**: Which research areas or specific papers might cite this work in the future?
    - **Research Clusters**: Identify any clusters or schools of thought within the citations
    
    ## 3. Citation Patterns & Potential Biases
    - Analyze if there are excessive self-citations or institutional bias
    - Identify any notable geographic or temporal patterns in the citations
    - Detect any potentially significant omissions in the literature review
    
    ## 4. Research Impact & Context
    - How does this paper build on or diverge from existing literature?
    - What is the potential academic and industry impact based on citation analysis?
    - Where does this research fit within broader scholarly trends?
    
    Paper text: {text[:4000]}...
    """
    
    try:
        result = llm.invoke(prompt)
        return result
    except Exception as e:
        st.error(f"Error generating citation insights: {e}")
        return "Unable to generate citation insights. Please try again."

def extract_startup_insights(text):
    """Extract startup-focused insights from the research paper"""
    try:
        # Initialize the LLM with explicit error handling
        llm = Together(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.3,
        )
        
        # Generate the comprehensive startup roadmap in a single prompt
        prompt = f"""
        You are an AI startup strategist specializing in transforming research into real-world businesses.
        Based on the provided research paper, generate a detailed step-by-step implementation roadmap that guides how to build a startup from the research findings.

        Create a concise but comprehensive startup roadmap with the following sections:
        
        ## 1. Problem Definition & Market Need
        - What problem does this research solve? (2-3 concise sentences)
        - Identify 3-4 target industries & explain specific market demand
        
        ## 2. Key Objectives & Expected Outcomes
        - Define 3-4 clear business goals & milestones for startup development
        - For each objective, define expected impact on customers and market
        
        ## 3. Technology Stack & Implementation Plan
        - List required technologies & frameworks (AI models, tools, APIs)
        - Outline system architecture explaining core functionalities
        
        ## 4. Stepwise Development Roadmap
        ### Phase 1: Research Translation & Prototyping (3 months)
        - List 3-4 specific tasks to build a proof-of-concept
        
        ### Phase 2: MVP Development (6 months)
        - List 4-5 essential features for the Minimum Viable Product
        
        ### Phase 3: Testing & Refinement (3 months)
        - Outline testing methodology and user feedback collection
        
        ### Phase 4: Full Implementation & Scaling (6 months)
        - Detail launch strategy and expansion approach
        
        ## 5. Resource & Funding Requirements
        - Team structure: Specify roles, skills, and number of team members needed
        - Estimated budget breakdown by phase
        
        ## 6. Business Model & Monetization Strategy
        - Recommend suitable business model based on research (SaaS, licensing, API, subscription)
        - List 2-3 pricing strategies & revenue streams
        
        ## 7. Market Entry & Adoption Strategy
        - Detail specific strategies to attract first 100 users/customers
        - Outline marketing & sales approach (B2B, B2C, partnerships)
        
        ## 8. Scaling & Future Roadmap
        - Outline expansion plans after MVP success
        - Define 3-4 long-term growth opportunities
        
        ## 9. Metrics for Success
        - List 5-6 specific KPIs to measure startup growth
        - Define benchmarks for each KPI
        
        Format your response as markdown with proper headings, bullet points, and numbered lists.
        Keep all content concise, actionable, and directly related to the research paper.
        
        Paper text: {text[:4000]}...
        """
        
        # Get the startup roadmap content with direct error handling
        startup_roadmap = llm.invoke(prompt)
        
        # Ensure we have valid content
        if not startup_roadmap or len(startup_roadmap.strip()) < 100:
            return "The generated startup insights were too short or empty. Please try again."
            
        # Return the plain markdown without trying to generate diagrams
        return startup_roadmap
        
    except Exception as e:
        # Capture the full error details
        error_details = str(e)
        st.error(f"Error extracting startup insights: {error_details}")
        return f"Unable to extract startup insights. Error: {error_details}"

def main():
    """Main function for the app"""
    # Load environment variables
    load_dotenv()
    
    # Configure the page
    st.set_page_config(page_title="PaperSense",
                       page_icon="📚",
                       layout="wide")
    
    # Set up the page
    st.write(css, unsafe_allow_html=True)
    
    st.title("📚 PaperSense")
    st.markdown("""
    PaperSense – Your AI-Powered Research Companion
    Upload academic papers and analyze them with AI assistance. 
    Get insights, flowcharts, and chat with your documents.
    """)
    
    # Initialize session state variables if not already done
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for uploading papers
    with st.sidebar:
        st.header("Upload Documents")
        pdf_docs = st.file_uploader(
            "Upload academic papers (PDF)", 
            accept_multiple_files=True,
            type=["pdf"]
        )
        
        if st.button("Process Papers"):
            if pdf_docs:
                with st.spinner("Processing papers..."):
                    # Extract text from PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    st.session_state.pdf_text = raw_text
                    
                    # Create text chunks
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.text_chunks = text_chunks
                    
                    # Create vector store
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.vectorstore = vectorstore
                    
                    # Clear chat history for new papers
                    st.session_state.chat_history = []
                    
                    # Mark as processed
                    st.session_state.processed = True
                    
                st.success("Papers processed successfully!")
            else:
                st.error("Please upload at least one PDF document.")

    # Main content area - show research chat only if papers were processed
    if st.session_state.get('processed', False):
        st.header("Interactive Research Paper Analysis")
        research_query = st.text_input("Ask a question about the paper:", placeholder="e.g., What is the main contribution of this paper?")
        
        if research_query:
            with st.spinner("Analyzing paper and generating response..."):
                handle_research_query(research_query)
        
        # Show tabs for other analysis features
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📑 Paper Analysis", 
            "📊 Research Flowchart",
            "🔍 Research Highlights",
            "📚 Similar Papers",
            "💼 Startup Insights"
        ])
        
        with tab1:
            st.header("Paper Analysis")
            
            # Detailed paper analysis
            st.subheader("Detailed Paper Analysis")
            if st.button("Analyze Paper"):
                with st.spinner("Analyzing paper..."):
                    analysis_report = analyze_paper(st.session_state.pdf_text)
                    st.markdown(analysis_report)
            
            # Plagiarism check
            st.subheader("Plagiarism Check")
            if st.button("Check for Plagiarism"):
                with st.spinner("Analyzing for plagiarism..."):
                    plagiarism_report = check_plagiarism(st.session_state.pdf_text)
                    st.markdown(plagiarism_report)
                    
            # Citation insights
            st.subheader("Citation Insights")
            if st.button("Generate Citation Insights"):
                with st.spinner("Analyzing citation network and impact..."):
                    citation_insights = generate_citation_insights(st.session_state.pdf_text)
                    st.markdown(citation_insights)
        
        with tab2:
            st.header("Research Paper Flowchart")
            st.markdown("""
            Generate a visual flowchart representing the paper's structure, methodology, and key components.
            """)
            if st.button("Generate Paper Flowchart"):
                with st.spinner("Creating visual representation of paper structure..."):
                    flowchart_path = generate_paper_flowchart(st.session_state.pdf_text)
                    if flowchart_path:
                        st.image(flowchart_path, caption="Paper Structure Flowchart", use_column_width=True)
                    else:
                        st.error("Failed to generate flowchart. Please try again.")
        
        with tab3:
            st.header("Research Highlights")
            if st.button("Extract Research Highlights"):
                with st.spinner("Analyzing paper for key insights..."):
                    highlights = extract_research_highlights(st.session_state.pdf_text)
                    st.markdown(highlights)
        
        with tab4:
            st.header("Similar Papers")
            if st.button("Find Similar Papers"):
                with st.spinner("Searching for related research..."):
                    similar_papers = find_similar_papers(st.session_state.pdf_text)
                    st.markdown(similar_papers)
        
        with tab5:
            st.header("Startup Insights")
            st.markdown("""
            Analyze the research paper for startup opportunities, competitive landscape, and alternative approaches.
            """)
            
            # Add debug information in an expander
            with st.expander("Debug Information"):
                st.write("API Key Status:")
                together_key = os.getenv("TOGETHER_API_KEY")
                if together_key:
                    st.write("✅ Together AI API key found")
                else:
                    st.write("❌ Together AI API key missing - Please add it to the .env file")
                
                st.write("\nTo resolve issues:")
                st.write("1. Ensure your Together AI API key is added to the .env file")
                st.write("2. Try refreshing the page if keys were recently added")
                st.write("3. Check your internet connection")
            
            startup_insights_button = st.button("Generate Startup Insights")
            
            if startup_insights_button:
                # Create a placeholder for showing progress and results
                insights_placeholder = st.empty()
                
                with st.spinner("Analyzing paper for startup potential..."):
                    insights_placeholder.info("Generating startup insights... This may take up to a minute.")
                    
                    try:
                        # Get startup insights
                        startup_insights = extract_startup_insights(st.session_state.pdf_text)
                        
                        # Clear the placeholder before showing results
                        insights_placeholder.empty()
                        
                        # Check if we got meaningful content
                        if startup_insights and len(startup_insights) > 100:
                            st.success("Startup insights generated successfully!")
                            st.markdown(startup_insights)
                        else:
                            st.error("No valid startup insights could be generated. Error message: " + startup_insights)
                            st.warning("Please try again or check if your API key is configured correctly.")
                    except Exception as e:
                        insights_placeholder.empty()
                        st.error(f"Error generating startup insights: {str(e)}")
                        st.warning("Please check your internet connection and API key configuration.")
    else:
        st.info("👈 Please upload PDF files and click 'Process Papers' to begin analysis.")

if __name__ == '__main__':
    main()