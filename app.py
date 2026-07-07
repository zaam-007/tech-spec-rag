import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DeterministicFakeEmbedding
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load your API key from the .env file
load_dotenv()

# 2. Load your custom data file
loader = TextLoader("sample.txt.txt")
documents = loader.load()

# 3. Create a Local Vector Database (Using an efficient fake embedding for day 1 to keep it 100% free)
# This processes the text into coordinates locally without using external API calls.
embeddings = DeterministicFakeEmbedding(size=1536)
vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1}) # Pull top 1 closest text chunk

# 4. Initialize the Ultra-Fast, Free Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# 5. Define instructions (The Template)
template = """Answer the question using ONLY the following context. If you don't know, say "I can't find that in the documents."

Context: {context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 6. Chain the components together (RAG Data Flow Pipeline)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Ask your question!
question = "Who is leading Project Aegis and what version is the prototype?"
print(f"\n--- Asking AI: {question} ---")

response = rag_chain.invoke(question)
print(f"\nAI Response:\n{response}\n")