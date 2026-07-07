import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load Environment (Groq API Key)
load_dotenv()

print("⏳ Step 1: Loading your technical PDF...")
# 2. Load the PDF file instead of a text file
loader = PyPDFLoader("document.pdf")
raw_documents = loader.load()

print(f"⏳ Step 2: Splitting {len(raw_documents)} pages into small chunks...")
# 3. Production Chunking: Split text by paragraphs/sentences with overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,      # Each chunk is roughly 120-150 words
    chunk_overlap=100    # 20-30 words of overlap so context isn't lost at edges
)
chunks = text_splitter.split_documents(raw_documents)
print(f"Created {len(chunks)} distinct chunks from your PDF.")

print("⏳ Step 3: Generating mathematical embeddings (this runs locally)...")
# 4. Real Embeddings: Using a highly-rated open-source model from HuggingFace
# This converts sentences into 384-dimensional vectors based on their deep meaning.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("⏳ Step 4: Storing chunks in ChromaDB vector database...")
# 5. Indexing: Create the vector store from our chunks
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Pull top 3 most relevant chunks

# 6. Initialize Groq Brain
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# 7. Strictly bounded Prompt Template
template = """You are a precise technical engineering assistant. 
Answer the question using ONLY the following retrieved context. 
If the context does not contain the answer, say "The provided document does not contain this information." 
Do not make up facts.

Context:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# 8. Assembly Line
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 9. Test your system against your PDF!
question = "What are the main power consumption modes or pin configurations mentioned?" 
# Change this question to match something specific inside YOUR downloaded PDF!

print(f"\n🚀 System Ready! Asking: '{question}'\n")
response = rag_chain.invoke(question)

print(f"Production-RAG Response:\n{response}\n")