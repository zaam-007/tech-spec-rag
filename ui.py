import os
from dotenv import load_dotenv
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool

# 1. Load your secret API keys
load_dotenv()

# 2. Configure the Browser Window Layout
st.set_page_config(page_title="Technical Assistant RAG", page_icon="⚙️", layout="wide")

st.title("⚙️ Production Technical Spec RAG Assistant")
st.write("Query your technical document with an Agentic fallback search routing layout.")

# 3. CRITICAL PERFORMANCE TRICK: Caching the Agent Setup
@st.cache_resource
def build_agentic_pipeline(pdf_path):
    import pymupdf4llm
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import FlashrankRerank

    # 1. Parse local PDF to structured Markdown
    md_text = pymupdf4llm.to_markdown(pdf_path)
    docs = [Document(page_content=md_text, metadata={"source": pdf_path})]

    # Split sections
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500,        
        chunk_overlap=400,      
        separators=["\n## ", "\n### ", "\n\n", "\n", " "] 
    )
    splits = text_splitter.split_documents(docs)
    
    # Base retrieval engine
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(splits, embeddings)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # Reranking compressor configuration
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
    compressor.top_n = 4
    compressed_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    # Initialize high-speed Groq LLM
    llm = ChatGroq(
        groq_api_key=st.secrets["groq_api_key"], 
        model_name="llama-3.1-8b-instant"
    )

    # Define the Document retriever tool explicitly for the agent
    @tool
    def search_pdf_specifications(query: str) -> str:
        """Useful when you need to answer technical questions directly from the 
        uploaded local engineering specification PDF documents."""
        retrieved_docs = compressed_retriever.invoke(query)
        return "\n\n".join([d.page_content for d in retrieved_docs])

    # Initialize the external web search tool
    web_search_tool = DuckDuckGoSearchRun()
    
    # Register the toolbox
    tools = [search_pdf_specifications, web_search_tool]

    # Create the agent systemic router prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a precise technical engineering assistant.\n"
            "First, ALWAYS use the `search_pdf_specifications` tool to search the provided document for answers.\n"
            "If the document context completely lacks the required information or explicitly states it is missing, "
            "seamlessly use the `duckduckgo_search` tool to look up relevant technical concepts, definitions, or standard conventions online.\n"
            "Be descriptive, accurate, and do not make up fake metrics."
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Assemble the functional agent brain configuration
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor

# 4. Check if your test PDF exists
target_pdf = "document.pdf"

if not os.path.exists(target_pdf):
    st.error(f"❌ '{target_pdf}' not found! Please drop your technical manual PDF into your project folder and rename it to '{target_pdf}'.")
    st.stop()

# 5. Initialize the pipeline
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.spinner("Initializing Agentic RAG Engine... Mapping tools and memory elements..."):
    try:
        agent_engine = build_agentic_pipeline(target_pdf)
        st.success("🤖 Agentic system online! Multimodal routing tools loaded.")
    except Exception as e:
        st.error(f"Failed to compile the RAG pipeline: {e}")
        st.stop()

# 6. Build the Visual User Interface Components
st.markdown("---")
user_query = st.text_input("💬 Ask a technical specification question from the document (or a general industry concept):")

# 7. Execute the request when the user interacts
if user_query:
    if user_query.strip() == "":
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Agent running reasoning loops and tool execution routes..."):
            try:
                # Execute agent pipeline processing
                response = agent_engine.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history
                })
                
                output_text = response["output"]
                
                # Update persistent memory array
                st.session_state.chat_history.append(("human", user_query))
                st.session_state.chat_history.append(("ai", output_text))
                
                # Render the clean markdown result inside a nice box
                st.markdown("### 🤖 Engine Output:")
                st.info(output_text)
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")
