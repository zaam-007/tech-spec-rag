import os
from dotenv import load_dotenv
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load your secret API keys
load_dotenv()

# 2. Configure the Browser Window Layout
st.set_page_config(page_title="Technical Assistant RAG", page_icon="⚙️", layout="wide")

st.title("⚙️ Production Technical Spec RAG Assistant")
st.write("Query your technical document with absolute precision layout-aware formatting.")

# 3. CRITICAL PERFORMANCE TRICK: Caching
@st.cache_resource
def build_rag_pipeline(pdf_path):
    import pymupdf4llm
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # 1. Parse the local PDF straight to structured Markdown text using the path variable
    md_text = pymupdf4llm.to_markdown(pdf_path)

    # 2. Wrap it in a LangChain Document object
    docs = [Document(page_content=md_text, metadata={"source": pdf_path})]

    # UPDATE 1: Bump up the chunk size so entire tables and sections stay whole!
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500,        # Increased from 1000
        chunk_overlap=400,      # Increased from 150
        separators=["\n## ", "\n### ", "\n\n", "\n", " "] 
    )
    splits = text_splitter.split_documents(docs)
    
    # Generate vectors locally
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Index splits inside the Chroma database
    vectorstore = Chroma.from_documents(splits, embeddings)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) # Fetch a few more chunks initially
    
    # 1. Import FlashRank Contextual Compressor
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import FlashrankRerank

    # 2. Initialize the FlashRank Re-ranker engine (ultra-lightweight, runs locally)
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
    
    # 3. Wrap your base retriever with the re-ranker compressor
    # This will take the top 10 chunks, re-rank them, and keep only the top 4 most precise ones
    compressor.top_n = 4
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    # Initialize the high-speed Groq LLM
    from langchain_groq import ChatGroq

    # Pass the secret directly to the model initialization
    llm = ChatGroq(
        groq_api_key=st.secrets["groq_api_key"], 
        model_name="llama-3.1-8b-instant"
    )
    
    # Define production-grade prompt templates
    template = """You are a precise technical engineering assistant. 
    Answer the question using ONLY the following retrieved context. 
    If the context does not contain the answer, say "The provided document does not contain this information." 
    Do not hallucinate or make up facts.

    Context:
    {context}

    Question: {question}
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # Assemble the engine chain (This stays exactly the same!)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
    

# 4. Check if your test PDF exists
target_pdf = "document.pdf"

if not os.path.exists(target_pdf):
    st.error(f"❌ '{target_pdf}' not found! Please drop your technical manual PDF into your project folder and rename it to '{target_pdf}'.")
    st.stop()

# 5. Initialize the pipeline
with st.spinner("Initializing RAG Engine... Embedding document pages into vector memory..."):
    try:
        rag_engine = build_rag_pipeline(target_pdf)
        st.success("🤖 System engine online! Vector store indexed successfully.")
    except Exception as e:
        st.error(f"Failed to compile the RAG pipeline: {e}")
        st.stop()

# 6. Build the Visual User Interface Components
st.markdown("---")
user_query = st.text_input("💬 Ask a technical specification question from the document:")

# 7. Execute the request when the user interacts
if user_query:
    if user_query.strip() == "":
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Searching database coordinates and generating answer..."):
            try:
                response = rag_engine.invoke(user_query)
                
                # Render the clean markdown result inside a nice box
                st.markdown("### 🤖 Engine Output:")
                st.info(response)
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")
