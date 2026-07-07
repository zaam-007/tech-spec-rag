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
st.write("Upload or reference a complex technical document and query it with absolute precision.")

# 3. CRITICAL PERFORMANCE TRICK: Caching
# Without this, Streamlit will re-read the PDF and re-embed the data on every single mouse click.
@st.cache_resource
def build_rag_pipeline(pdf_path):
    # Load the PDF file
    loader = PyPDFLoader(pdf_path)
    raw_documents = loader.load()
    
    # Split into structured chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = text_splitter.split_documents(raw_documents)
    
    # Generate vectors locally
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Index chunks inside the Chroma database
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Initialize the high-speed Groq LLM
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
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
    
    # Assemble the engine chain
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
        # Create a visually pleasing loading status container
        with st.spinner("Searching database coordinates and generating answer..."):
            try:
                response = rag_engine.invoke(user_query)
                
                # Render the clean markdown result inside a nice box
                st.markdown("### 🤖 Engine Output:")
                st.info(response)
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")