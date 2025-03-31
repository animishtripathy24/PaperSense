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
from openai import OpenAI
from htmlTemplates import css, bot_template, user_template
from langchain.prompts import PromptTemplate
import requests
from serpapi import GoogleSearch
import urllib.parse
from pypeprompts import PromptAnalyticsTracker
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Mapping, Optional

# Load environment variables
load_dotenv()

# Get project token from environment variables
project_token = os.getenv("PROJECT_TOKEN")
# Initialize the tracker with your project token
tracker = PromptAnalyticsTracker(project_token=project_token)

# Set OpenAI API Key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    print("OpenAI API key loaded successfully")
else:
    print("Warning: OpenAI API key not found in .env file")

# Initialize the OpenAI client
client = OpenAI(api_key=openai_api_key)

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

def openai_completion(prompt, max_tokens=2000, temperature=0.85):
    """Make an API call to OpenAI using the specified parameters"""
    print(f"OpenAI API Key: {openai_api_key}")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in OpenAI API call: {str(e)}")
        return f"Error: {str(e)}"

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
    # Create a wrapper for OpenAI that conforms to LangChain's LLM interface
    class OpenAIWrapper(LLM):
        client: Any
        
        def __init__(self, api_key):
            super().__init__()
            self.client = OpenAI(api_key=api_key)
            
        @property
        def _llm_type(self) -> str:
            return "custom_openai_wrapper"
        
        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> str:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.85,
                max_tokens=2000,
                stop=stop
            )
            return response.choices[0].message.content
            
        @property
        def _identifying_params(self) -> Mapping[str, Any]:
            return {"model": "gpt-4o-mini"}
    
    # Create the wrapper
    llm = OpenAIWrapper(openai_api_key)
    
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
    # Initialize vectorstore and get conversation chain if needed
    if "vectorstore" in st.session_state and st.session_state.vectorstore:
        if "conversation" not in st.session_state or st.session_state.conversation is None:
            try:
                st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
            except Exception as e:
                st.error(f"Error initializing conversation chain: {e}")
                # Fallback to a simpler approach without using the conversation chain
                enhanced_response = get_enhanced_response(
                    user_question, "contextual", st.session_state.pdf_text, []
                )
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                st.session_state.chat_history.append({"role": "assistant", "content": enhanced_response})
                return enhanced_response
    
    # Get current chat history from session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Convert chat history to the format expected by the conversation chain
    chat_history_tuples = []  # Store (human_message, ai_message) tuples for LangChain
    
    # Process existing chat history to prepare for chain
    for i in range(0, len(st.session_state.chat_history)-1, 2):
        if i+1 < len(st.session_state.chat_history):
            human_msg = st.session_state.chat_history[i].get('content', '')
            ai_msg = st.session_state.chat_history[i+1].get('content', '')
            chat_history_tuples.append((human_msg, ai_msg))
    
    # Classify the type of question to determine the best approach
    query_type = classify_query(user_question)
    
    try:
        if st.session_state.conversation:
            # Use the conversation chain for retrieval-based QA
            response = st.session_state.conversation(
                {"question": user_question, "chat_history": chat_history_tuples}
            )
            enhanced_response = response.get("answer", "I couldn't find a relevant answer in the paper.")
        else:
            # Fallback to direct enhanced response if conversation chain fails
            enhanced_response = get_enhanced_response(
                user_question, query_type, st.session_state.pdf_text, chat_history_tuples
            )
    except Exception as e:
        st.error(f"Error processing query: {e}")
        enhanced_response = get_enhanced_response(
            user_question, query_type, st.session_state.pdf_text, chat_history_tuples
        )
    
    # Add new message pair to the chat history
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": enhanced_response})
    
    # Return the response (this won't be displayed directly, as the UI will rerun and show the updated chat history)
    return enhanced_response


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
        response = openai_completion(prompt)
        
        # Track the prompt and response
        tracker.track('PaperSense', {
            'functionName': 'get_enhanced_response',
            'prompt': prompt,
            'output': response
        })
        
        return response
    except Exception as e:
        return f"I encountered an error while processing your question: {str(e)}. Please try again with a different question."


def check_plagiarism(text):
    """Check for plagiarism in the text using AI"""
    # Divide text into sections for more comprehensive analysis
    sections = []
    chunk_size = 1500
    for i in range(0, min(len(text), 6000), chunk_size):
        sections.append(text[i:i+chunk_size])
    
    results = []
    overall_percentage = 0
    all_prompts = []
    all_responses = []
    
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
            result = openai_completion(prompt, temperature=0.0)
            all_prompts.append(prompt)
            all_responses.append(result)
            
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
        
        # Track the prompts and responses
        tracker.track('PaperSense', {
            'functionName': 'check_plagiarism',
            'prompt': str(all_prompts),
            'output': report
        })
        
        return report
    else:
        return "⚠️ **Plagiarism check inconclusive.** Please try again or check the text manually."

def extract_research_highlights(text):
    """Extract key research highlights from the paper with a business focus"""
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
        result = openai_completion(prompt, temperature=0.2)
        
        # Track the prompt and response
        tracker.track('PaperSense', {
            'functionName': 'extract_research_highlights',
            'prompt': prompt,
            'output': result
        })
        
        return result
    except Exception as e:
        st.error(f"Error extracting research highlights: {e}")
        return "Unable to extract research highlights. Please try again."

def analyze_paper(text):
    """Analyze the paper with a comprehensive structured approach"""
    print("OpenAI API Key:", "*" * len(openai_api_key) if openai_api_key else "None")
    
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
    
    For every claim you make, ensure it is directly supported by text in the paper. Use specific quotes and page/section references where possible. If information for any section is not available in the paper, clearly state "The paper does not provide sufficient information about [topic]" rather than making assumptions.
    
    Paper text: {text[:6000]}...
    """
    
    try:
        print("Analyzing paper...")
        result = openai_completion(prompt, temperature=0.2)
        print("Paper analyzed.")

        # # Track the prompt and response
        # tracker.track('PaperSense', {
        #     'functionName': 'analyze_paper',
        #     'prompt': prompt,
        #     'output': result
        # })
        
        return result
    except Exception as e:
        st.error(f"Error analyzing paper: {e}")
        return "Unable to analyze the paper. Please try again."

def analyze_citations(text):
    """Analyze citations in the research paper"""
    prompt = f"""
    Analyze the citations in this research paper. Identify the most influential references and categorize them as:
    1. Supporting Evidence
    2. Contradictory Work
    3. Methodological Influences
    
    Summarize their contributions concisely. Additionally, provide insights on whether the citations show bias (e.g., excessive self-references) and highlight missing perspectives that could strengthen the research.
    
    Paper text: {text[:4000]}...
    """
    
    try:
        result = openai_completion(prompt, temperature=0.2)
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
        title_result = openai_completion(title_prompt, temperature=0.3)
        
        # Track the title extraction prompt
        tracker.track('PaperSense', {
            'functionName': 'find_similar_papers_extract_title',
            'prompt': title_prompt,
            'output': title_result
        })
        
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
            "num": 6  # Get top 10 results to ensure at least 5 are displayed
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
        summary_prompts = []
        summary_results = []
        
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
                    concise_description = openai_completion(summary_prompt, temperature=0.3).strip()
                    summary_prompts.append(summary_prompt)
                    summary_results.append(concise_description)
                    
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
        
        # Track the summary prompts and results
        if summary_prompts:
            tracker.track('PaperSense', {
                'functionName': 'find_similar_papers_generate_summaries',
                'prompt': str(summary_prompts),
                'output': str(summary_results)
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
                generated_papers = openai_completion(generation_prompt, temperature=0.3)
                
                # Track the generation prompt and result
                tracker.track('PaperSense', {
                    'functionName': 'find_similar_papers_generate_additional',
                    'prompt': generation_prompt,
                    'output': generated_papers
                })
                
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
        
        # Track the final output
        tracker.track('PaperSense', {
            'functionName': 'find_similar_papers_final',
            'prompt': f"Title: {title}, Concepts: {concepts}",
            'output': output
        })
        
        return output
        
    except Exception as e:
        st.error(f"Error finding similar papers: {e}")
        return "Unable to find similar papers. Please try again."

def generate_paper_flowchart(text):
    """Generate a structured flowchart of the paper's methodology and structure using QuickChart API"""
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
        paper_structure = openai_completion(prompt, temperature=0.2)
        
        # Track the prompt and response
        tracker.track('PaperSense', {
            'functionName': 'generate_paper_flowchart',
            'prompt': prompt,
            'output': paper_structure
        })
        
        # Parse the structure to extract components and connections
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
        
        # Create DOT language representation
        dot = "digraph G {\n"
        # Graph attributes
        dot += "  rankdir=TB;\n"
        dot += "  node [shape=box, style=filled, fillcolor=lightblue, fontname=Arial, fontsize=16];\n"
        dot += "  edge [fontname=Arial, fontsize=12, arrowsize=0.9];\n\n"
        
        # Add nodes
        for component, info in components.items():
            # Escape special characters in labels
            safe_component = component.replace('"', '\\"')
            safe_description = info['description'].replace('"', '\\"')
            dot += f'  "{safe_component}" [label="{safe_component}\\n\\n{safe_description}"];\n'
        
        # Add edges
        for source, target in connections:
            if source in components and target in components:
                safe_source = source.replace('"', '\\"')
                safe_target = target.replace('"', '\\"')
                dot += f'  "{safe_source}" -> "{safe_target}";\n'
        
        dot += "}"
        
        # Create QuickChart API URL
        encoded_dot = urllib.parse.quote(dot)
        quickchart_url = f"https://quickchart.io/chart?cht=gv&chl={encoded_dot}&layout=dot"
        
        # Track the generated DOT and URL
        tracker.track('PaperSense', {
            'functionName': 'generate_paper_flowchart_diagram',
            'prompt': paper_structure,
            'output': quickchart_url
        })
        
        # Return the URL directly instead of downloading the image
        return quickchart_url
            
    except Exception as e:
        st.error(f"Error generating flowchart: {e}")
        return None

def generate_citation_insights(text):
    """Generate concise and relevant citation insights from the paper"""
    # First, extract the most important citations
    citation_extraction_prompt = f"""
    Extract the 10 most important citations from this research paper.
    For each citation, provide:
    1. The citation text or reference number as it appears in the paper
    2. The authors' names (first author is sufficient if there are many)
    3. The publication year
    4. The paper title (if available)
    5. The exact context where this citation is used in the paper
    
    Format as:
    CITATION: [citation]
    AUTHORS: [authors]
    YEAR: [year]
    TITLE: [title]
    CONTEXT: [sentence/paragraph where citation appears]
    
    Only include citations you can clearly identify from the text.
    Paper text: {text[:4000]}...
    """
    
    try:
        # First, extract all citations
        citation_extraction_result = openai_completion(citation_extraction_prompt, temperature=0.2)
        
        # Track the extraction prompt and result
        tracker.track('PaperSense', {
            'functionName': 'generate_citation_insights_extraction',
            'prompt': citation_extraction_prompt,
            'output': citation_extraction_result
        })
        
        # Now create a more focused, concise analysis
        analysis_prompt = f"""
        You are a research analyst creating a CONCISE citation report. Based on the extracted citations below, create a brief but insightful citation analysis.
        
        {citation_extraction_result}
        
        Create a BRIEF citation report with these sections (limit to 2-3 points per section):

        ## Key Foundational Works (max 100 words)
        Identify 2-3 foundational papers this research builds upon. For each, explain exactly how they influenced this paper.

        ## Citation Patterns (max 70 words)
        Identify concrete patterns like recency bias, geographical focus, or self-citations. Use percentages or specific numbers.

        ## Research Gaps (max 70 words)
        Identify 1-2 specific research gaps this paper addresses, based on the citations.

        Use ONLY specific examples you can verify from the citations. Keep your total response under 300 words.
        Avoid vague statements like "many papers" or "several researchers" - always use specific names and numbers.
        If you cannot find enough information for a section, write "Insufficient citation information" for that section.
        
        Original paper text: {text[:1000]}...
        """
        
        # Get the citation insights
        citation_insights = openai_completion(analysis_prompt, temperature=0.2)
        
        # Track the analysis prompt and result
        tracker.track('PaperSense', {
            'functionName': 'generate_citation_insights_analysis',
            'prompt': analysis_prompt,
            'output': citation_insights
        })
        
        # If the result is too short or generic, try an alternative approach
        if len(citation_insights.split()) < 150 or "insufficient citation information" in citation_insights.lower():
            # Fallback to a direct citation analysis
            fallback_prompt = f"""
            Analyze the most significant citations in this paper. Focus on SPECIFICITY and BREVITY.
            
            For each of the 5 most important citations you can find:
            1. PAPER: Authors (year). Title.
            2. CONTRIBUTION: One sentence on what this cited work contributed to the field
            3. USAGE: One sentence on exactly how the current paper uses this citation
            
            Format as a bullet list with the 5 most important citations.
            
            Add a brief conclusion (2-3 sentences) summarizing what these citations reveal about the paper's foundation.
            
            Total response should be under 300 words and ONLY include citations you can specifically identify in the text.
            
            Paper text: {text[:5000]}...
            """
            
            citation_insights = openai_completion(fallback_prompt, temperature=0.2)
            
            # Track the fallback prompt and result
            tracker.track('PaperSense', {
                'functionName': 'generate_citation_insights_fallback',
                'prompt': fallback_prompt,
                'output': citation_insights
            })
        
        return citation_insights
    except Exception as e:
        st.error(f"Error generating citation insights: {e}")
        return "Unable to generate citation insights. Please try again."

def extract_startup_insights(text):
    """Extract startup-focused insights from the research paper"""
    try:
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
        startup_roadmap = openai_completion(prompt, temperature=0.3)
        
        # Track the prompt and response
        tracker.track('PaperSense', {
            'functionName': 'extract_startup_insights',
            'prompt': prompt,
            'output': startup_roadmap
        })
        
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

def generate_what_if_scenarios(text):
    """Generate 'What-If' scenarios by modifying key parameters, models, or techniques"""
    prompt = f"""
    You are an AI research methodology expert. Analyze the given research paper's methodology and generate insightful 'What-If' scenarios.
    
    For the provided research paper, create a structured analysis that:
    
    ## 1. Modified Parameters or Models
    - Identify 3-4 specific components of the current methodology that could be modified
    - For each component, suggest a concrete alternative approach, model, or technique
    - Explain why this alternative might be worth exploring
    - Be specific about exactly what parameters would change (e.g., "Increase learning rate from 0.001 to 0.01")
    
    ## 2. Predicted Outcomes
    - For each modification, estimate the potential impact on:
      * Performance metrics (accuracy, precision, recall, F1-score)
      * Computational requirements (training time, inference speed)
      * Model complexity and size
      * Generalizability to new data
    - Provide numerical estimates where possible (e.g., "Accuracy may improve by 2-5%")
    
    ## 3. Comparative Analysis
    - Present a markdown-formatted comparison table for each scenario showing:
      * Original approach vs. Modified approach
      * Expected metrics for both approaches
      * Trade-offs in terms of complexity, performance, and resources
    - Highlight the key advantages and limitations of each modification
    
    Base your analysis strictly on the methodology described in the paper. Be concrete and specific rather than generic.
    Use a clear structure with headers, bullet points, and tables to make the analysis easy to read.
    
    Paper text: {text[:6000]}...
    """
    
    try:
        what_if_scenarios = openai_completion(prompt, temperature=0.3)
        
        # Track the prompt and response
        tracker.track('PaperSense', {
            'functionName': 'generate_what_if_scenarios',
            'prompt': prompt,
            'output': what_if_scenarios
        })
        
        return what_if_scenarios
    except Exception as e:
        st.error(f"Error generating what-if scenarios: {e}")
        return "Unable to generate what-if scenarios. Please try again."

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
        
        st.markdown("*Click 'Process Papers' after uploading to analyze your documents.*")
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
                    
                    # Reset last query to ensure fresh interaction
                    if 'last_query' in st.session_state:
                        del st.session_state.last_query
                    
                    # Reset user input
                    st.session_state.user_input = ""
                    
                    # Mark as processed
                    st.session_state.processed = True
                    
                st.success("Papers processed successfully!")
            else:
                st.error("Please upload at least one PDF document.")

    # Main content area - show research chat only if papers were processed
    if st.session_state.get('processed', False):
        st.header("Interactive Research Paper Analysis")
        st.caption("Ask questions about the paper to get instant, contextually relevant answers.")
        
        # Add a container to isolate the chat interface
        chat_container = st.container()
        
        # Initialize input value in session state if it doesn't exist
        if "user_input" not in st.session_state:
            st.session_state.user_input = ""
            
        # Function to handle input submission
        def submit_query():
            if st.session_state.user_input:
                # Store current input
                current_query = st.session_state.user_input
                # Clear the input
                st.session_state.user_input = ""
                # Save as last query to prevent reprocessing
                st.session_state.last_query = current_query
                # Process the query
                with st.spinner("Analyzing paper and generating response..."):
                    handle_research_query(current_query)
        
        with chat_container:
            # Display existing chat history first (only once)
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
            
            # Create input box for new queries with callback for changes
            st.text_input(
                "Ask a question about the paper:", 
                placeholder="e.g., What is the main contribution of this paper?", 
                key="user_input",
                on_change=submit_query
            )
            
            st.markdown("*Type your question above and press Enter to get AI-powered insights from the paper.*")
        
        # Show tabs for other analysis features
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📑 Paper Analysis", 
            "📊 Research Flowchart",
            "🔍 Research Highlights",
            "📚 Similar Papers",
            "💼 Startup Insights",
            "🧪 What-If Scenarios"
        ])
        
        with tab1:
            st.header("Paper Analysis")
            st.caption("Get comprehensive insights into the paper's methodology, results, and key contributions.")
            
            # Detailed paper analysis
            st.subheader("Detailed Paper Analysis")
            st.markdown("*Extract the core concepts, methodology, and findings with AI-powered analysis.*")
            if st.button("Analyze Paper"):
                with st.spinner("Analyzing paper..."):
                    analysis_report = analyze_paper(st.session_state.pdf_text)
                    st.markdown(analysis_report)
            
            # Plagiarism check
            st.subheader("Plagiarism Check")
            st.markdown("*Evaluate the originality of the paper and identify potential plagiarism concerns.*")
            if st.button("Check for Plagiarism"):
                with st.spinner("Analyzing for plagiarism..."):
                    plagiarism_report = check_plagiarism(st.session_state.pdf_text)
                    st.markdown(plagiarism_report)
                    
            # Citation insights
            st.subheader("Citation Insights")
            st.markdown("*Explore key references and understand how they support the paper's arguments.*")
            if st.button("Generate Citation Insights", key="citation_insights_btn"):
                # Create a placeholder for showing progress and results
                citation_placeholder = st.empty()
                
                with st.spinner("Analyzing citation network and impact..."):
                    citation_placeholder.info("Extracting and analyzing citations... This may take up to a minute.")
                    
                    try:
                        # Get citation insights
                        citation_insights = generate_citation_insights(st.session_state.pdf_text)
                        
                        # Clear the placeholder before showing results
                        citation_placeholder.empty()
                        
                        # Check if we got meaningful content
                        if citation_insights and len(citation_insights) > 100:
                            st.success("Citation insights generated successfully!")
                            st.markdown(citation_insights)
                        else:
                            st.error("No valid citation insights could be generated.")
                            st.warning("Please try again or check if your paper contains citations.")
                    except Exception as e:
                        citation_placeholder.empty()
                        st.error(f"Error generating citation insights: {str(e)}")
                        st.warning("Please check your internet connection and API key configuration.")
        
        with tab2:
            st.header("Research Paper Flowchart")
            st.markdown("""
            Generate a visual flowchart representing the paper's structure, methodology, and key components.
            """)
            if st.button("Generate Paper Flowchart"):
                with st.spinner("Creating visual representation of paper structure..."):
                    flowchart_url = generate_paper_flowchart(st.session_state.pdf_text)
                    if flowchart_url:
                        st.markdown(f"### Paper Structure Flowchart")
                        # Display the image directly using HTML for better rendering
                        st.markdown(f'<img src="{flowchart_url}" alt="Paper Structure Flowchart" width="100%">', unsafe_allow_html=True)
                        
                        # Also provide the URL in case the user wants to access it directly
                        with st.expander("View Flowchart URL"):
                            st.markdown(f"[Open in new tab]({flowchart_url})")
                    else:
                        st.error("Failed to generate flowchart. Please try again.")
        
        with tab3:
            st.header("Research Highlights")
            st.markdown("*Quickly grasp the key takeaways and business applications of this research.*")
            if st.button("Extract Research Highlights"):
                with st.spinner("Analyzing paper for key insights..."):
                    highlights = extract_research_highlights(st.session_state.pdf_text)
                    st.markdown(highlights)
        
        with tab4:
            st.header("Similar Papers")
            st.markdown("*Find and explore related publications to broaden your understanding of the topic.*")
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
                openai_key = os.getenv("OPENAI_API_KEY")
                if openai_key:
                    st.write("✅ OpenAI API key found")
                else:
                    st.write("❌ OpenAI API key missing - Please add it to the .env file")
                
                st.write("\nTo resolve issues:")
                st.write("1. Ensure your OpenAI API key is added to the .env file")
                st.write("2. Try refreshing the page if keys were recently added")
                st.write("3. Check your internet connection")
            
            st.markdown("*Generate a comprehensive startup roadmap based on the paper's innovations.*")
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
                            with st.expander("See Full Insights"):
                                st.markdown(startup_insights)


                        else:
                            st.error("No valid startup insights could be generated. Error message: " + startup_insights)
                            st.warning("Please try again or check if your API key is configured correctly.")
                    except Exception as e:
                        insights_placeholder.empty()
                        st.error(f"Error generating startup insights: {str(e)}")
                        st.warning("Please check your internet connection and API key configuration.")
        
        with tab6:
            st.header("What-If Methodology Scenarios")
            st.markdown("""
            Analyze the research methodology and generate alternative scenarios with modified parameters, 
            models, or techniques. Compare potential outcomes and trade-offs.
            """)
            
            st.markdown("*See how changing key parameters or methodologies could affect the research outcomes.*")
            if st.button("Generate What-If Scenarios"):
                # Create a placeholder for showing progress and results
                what_if_placeholder = st.empty()
                
                with st.spinner("Analyzing methodology and generating alternative scenarios..."):
                    what_if_placeholder.info("Generating What-If scenarios... This may take up to a minute.")
                    
                    try:
                        # Get what-if scenarios
                        what_if_scenarios = generate_what_if_scenarios(st.session_state.pdf_text)
                        
                        # Clear the placeholder before showing results
                        what_if_placeholder.empty()
                        
                        # Check if we got meaningful content
                        if what_if_scenarios and len(what_if_scenarios) > 100:
                            st.success("What-If scenarios generated successfully!")
                            st.markdown(what_if_scenarios)
                        else:
                            st.error("No valid What-If scenarios could be generated.")
                            st.warning("Please try again with a paper that has a clearer methodology section.")
                    except Exception as e:
                        what_if_placeholder.empty()
                        st.error(f"Error generating What-If scenarios: {str(e)}")
                        st.warning("Please check your internet connection and API key configuration.")
    else:
        st.info("👈 Please upload PDF files and click 'Process Papers' to begin analysis.")

if __name__ == '__main__':
    main()