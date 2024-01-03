
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader

import streamlit as st
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

embeddings = HuggingFaceEmbeddings()

# function to parse PDFs
@st.cache_resource
def parse_pdf(file):
    '''
        Loading the pdf file.
    '''
    loader = PyPDFLoader(file)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
    # separator = "Q:",
    chunk_size = 100,
    chunk_overlap  = 10,
    length_function = len,
    is_separator_regex = False,
    )

    documents = text_splitter.split_documents(pages)
    return documents

# We can't fit the whole document inside the prompt so split the document into smaller chunks
@st.cache_resource
def embed_text(_documents):
    
    split_text = parse_pdf(_documents)
    faiss = FAISS.from_documents(split_text, embeddings)

    return faiss

@st.cache_data
def create_buffer_memory():
   
    memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)
    return memory

def create_llm():
    
    # Getting LLM from HuggingFaceHub
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":1, "max_length":1000})
    return llm

def get_answer(llm,retriever, memory, query):
   
    conversation_chain =  ConversationalRetrievalChain.from_llm(
    llm=llm, 
    retriever=retriever, 
    memory=memory,
    get_chat_history=lambda h : h)
    answer = conversation_chain.run(query)
    return answer



st.header("FAQ QnA")
uploaded_file = r"Document\\GenAI.pdf"

if uploaded_file is not None:
    faiss = embed_text(uploaded_file)
    retriever = faiss.as_retriever(chain_type="stuff", search_kwargs={"k": 3})
    memory = create_buffer_memory()
    llm = create_llm()
    query = st.text_area("Hello there! Welcome to our Chat Bot. How can I assist you today?")
    button = st.button("Submit")
    if button:
        st.write(get_answer(llm, retriever, memory, query))
    
# Run this code with the following command: streamlit run app.py