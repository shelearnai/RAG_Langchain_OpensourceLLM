# RAG Pipeline with LangChain: Streamlit App
## Open-Sourced LLM

This repository contains a Streamlit app that integrates several powerful NLP tools for text processing, including Retrieval-Augmented Generation (RAG), LangChain, text summarization, and question-answering. Instead of relying on proprietary APIs like OpenAI, we use an open-source LLM (Language Model) for our RAG pipeline.

RAG (Retrieval-Augmented Generation):
Combine retrieval-based models with creative language generation.
Retrieve relevant passages and generate context-aware responses.
LangChain:
A lightweight language processing library.
Perform tokenization, stemming, and other essential NLP tasks.
Text Summarization:
Summarize lengthy articles, research papers, or reports.
Get concise overviews without losing critical details.
Question-Answering from PDFs:
Extract structured information from PDF documents.
Ask questions based on the content and receive answers.

Installation
1. git clone https://github.com/your-username/rag-langchain-app.git
cd rag-langchain-app

2. pip install -r requirements.txt

3. streamlit run streamlit_app.py

Docker
1. docker build -t streamlitapp .
2. docker run -p 8501:8501 streamlitapp

