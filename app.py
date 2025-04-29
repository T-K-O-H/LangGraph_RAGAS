import streamlit as st
from rag_graph import rag_graph, vectorstore
from langchain_core.messages import HumanMessage
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
import chardet
from PyPDF2 import PdfReader
import io
import re
import logging
from langchain_openai import OpenAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Set page config
st.set_page_config(
    page_title="RAG Application with RAGAS Metrics",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Title
st.title("🤖 RAG Application with RAGAS Metrics")

def clean_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    return text.strip()

def split_into_sentences(text):
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def process_text(text):
    # Clean the text
    text = clean_text(text)
    
    # Split into sentences
    sentences = split_into_sentences(text)
    
    # Initialize text splitter with semantic chunking
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=25  # Lower threshold for more semantic grouping
    )
    
    # First chunk sentences semantically
    sentence_chunks = text_splitter.split_text("\n".join(sentences))
    
    # Then combine into paragraphs greedily
    paragraphs = []
    current_paragraph = []
    current_size = 0
    max_chunk_size = 1000  # Maximum characters per chunk
    
    for chunk in sentence_chunks:
        chunk_size = len(chunk)
        if current_size + chunk_size <= max_chunk_size:
            current_paragraph.append(chunk)
            current_size += chunk_size
        else:
            if current_paragraph:
                paragraphs.append("\n".join(current_paragraph))
            current_paragraph = [chunk]
            current_size = chunk_size
    
    if current_paragraph:
        paragraphs.append("\n".join(current_paragraph))
    
    return paragraphs

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Only add if text was extracted
                text += page_text + "\n"
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
            
        return clean_text(text)
    except Exception as e:
        raise ValueError(f"Error extracting text from PDF: {str(e)}")

# Sidebar for document upload
with st.sidebar:
    st.header("Document Management")
    uploaded_file = st.file_uploader("Upload a document (max 10MB)", type=["txt", "pdf"])
    if uploaded_file:
        try:
            # Check file size (10MB = 10 * 1024 * 1024 bytes)
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("File size exceeds 10MB limit. Please upload a smaller file.")
            else:
                logger.info(f"Processing uploaded file: {uploaded_file.name}")
                # Process the document based on file type
                if uploaded_file.type == "application/pdf":
                    text = extract_text_from_pdf(uploaded_file)
                else:
                    # For text files, detect encoding
                    raw_data = uploaded_file.getvalue()
                    result = chardet.detect(raw_data)
                    encoding = result['encoding']
                    text = raw_data.decode(encoding)
                
                if not text.strip():
                    raise ValueError("No text content found in the document")
                
                # Process text into semantic chunks
                chunks = process_text(text)
                
                if not chunks:
                    raise ValueError("No valid text chunks could be created from the document")
                
                # Add to vectorstore
                logger.info(f"Adding {len(chunks)} chunks to vectorstore")
                vectorstore.add_texts(chunks)
                
                st.success("Document processed and added to the knowledge base!")
                st.info(f"Processed {len(chunks)} chunks of text")
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            st.error(f"Error processing document: {str(e)}")

# Main chat interface
st.header("Chat with the RAG System")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
default_question = "What is quantum computing?"
question = st.text_input("Ask a question", value=default_question)
if st.button("Submit") or question != default_question:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)
    
    # Prepare the state for the RAG graph
    state = {
        "messages": [HumanMessage(content=question)],
        "context": "",  # Initialize empty context
        "response": "",  # Initialize empty response
        "next": "retrieve"  # Start with retrieval
    }
    
    # Run the RAG graph
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                logger.info("Starting RAG process")
                result = rag_graph.invoke(state)
                logger.info("RAG process completed")
                
                # Display the response and metrics
                st.markdown(result["response"])
                st.write("Full Response Data:")
                st.json(result)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"]
                })
            except Exception as e:
                logger.error(f"Error in RAG process: {str(e)}")
                st.error(f"Error generating response: {str(e)}") 