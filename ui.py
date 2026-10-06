import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# 1. Load security keys
load_dotenv()

# 2. Configure Streamlit Page
st.set_page_config(page_title="Technical Assistant RAG", page_icon="⚙️", layout="wide")

st.title("⚙️ Production Technical Spec RAG Assistant")
st.write("An intelligent engineering agent equipped with layout-aware memory, guardrails, and live evaluation capabilities.")


# Security and Guardrails Check
def sanitize_and_check_input(query: str) -> tuple[bool, str]:
    clean_query = query.strip()
    if not clean_query:
        return False, "Please enter a valid technical question."
    
    injection_patterns = [
        r"ignore (all )?previous instructions",
        r"disregard the above",
        r"system prompt",
        r"you are now an? unrestricted",
        r"override safety"
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, clean_query, re.IGNORECASE):
            return False, "⚠️ Prompt manipulation pattern detected. Please rephrase your technical question."
            
    return True, clean_query


# 3. Cached Embeddings Initialization (Prevents HuggingFace model download hangs)
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# 4. Cached Agent & Index Pipeline
@st.cache_resource(show_spinner=False)
def build_agentic_pipeline(pdf_path):
    import pymupdf4llm
    from langchain_core.documents import Document
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_community.document_compressors import FlashrankRerank

    # Load layout-aware PDF markdown
    md_text = pymupdf4llm.to_markdown(pdf_path)
    docs = [Document(page_content=md_text, metadata={"source": pdf_path})]

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500,        
        chunk_overlap=400,      
        separators=["\n## ", "\n### ", "\n\n", "\n", " "] 
    )
    splits = text_splitter.split_documents(docs)
    
    # Fast Embeddings & VectorStore
    embeddings = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings
    )
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # FlashRank Reranker Setup
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
    compressor.top_n = 4
    compressed_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    # Initialize Core LLM
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("groq_api_key")
    if not api_key:
        raise ValueError("Groq API Key not found in environment or secrets!")

    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

    # Define Agent Tools
    @tool
    def search_pdf_specifications(query: str) -> str:
        """Useful when you need to answer technical questions directly from the 
        uploaded local engineering specification PDF documents."""
        retrieved_docs = compressed_retriever.invoke(query)
        st.session_state.last_retrieved_context = "\n\n".join([d.page_content for d in retrieved_docs])
        return st.session_state.last_retrieved_context

    web_search = DuckDuckGoSearchRun()
    
    @tool
    def duckduckgo_search(query: str) -> str:
        """Useful for searching the internet to get real-time information, 
        industry definitions, or online technical documentation."""
        return web_search.run(query)
    
    tools = [search_pdf_specifications, duckduckgo_search]

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a precise technical engineering assistant.\n"
            "A technical specification document (`document.pdf`) is ALREADY uploaded, chunked, and indexed in your vector store.\n"
            "ALWAYS execute the `search_pdf_specifications` tool to look up technical details in this indexed document before answering.\n"
            "Do NOT output raw function syntax like '<function=...>'—execute tools directly.\n"
            "If the document context completely lacks the required information, "
            "seamlessly call the `duckduckgo_search` tool to look up technical concepts online.\n"
            "Be descriptive, accurate, and do not make up fake metrics."
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# 5. Real-time LLM QA Judge Function
def run_llm_judge(query, response, context):
    eval_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("groq_api_key")
    eval_llm = ChatGroq(groq_api_key=eval_api_key, model_name="llama-3.3-70b-versatile")
    
    safe_context = context[:8000] + "\n...[Context truncated]..." if len(context) > 8000 else context

    eval_template = """You are an independent QA quality controller evaluating a technical RAG system.
    Evaluate the System Response based on the User Query and retrieved Context.
    
    Provide two scores between 0.0 and 1.0:
    1. Faithfulness: 1.0 means the response contains zero hallucinations and derives entirely from the context or web results.
    2. Answer Relevance: 1.0 means the system answered exactly what the user asked.
    
    Return your evaluation strictly in this text format:
    Faithfulness Score: [score]
    Relevance Score: [score]
    Reasoning: [one brief sentence explaining the grades]
    
    User Query: {query}
    Retrieved Context: {context}
    System Response: {response}
    """
    eval_prompt = ChatPromptTemplate.from_template(eval_template)
    eval_chain = eval_prompt | eval_llm | StrOutputParser()
    return eval_chain.invoke({"query": query, "response": response, "context": safe_context})


# 6. Initialize State and Load Pipeline safely
target_pdf = "document.pdf"

if not os.path.exists(target_pdf):
    st.error(f"❌ '{target_pdf}' not found! Please ensure your PDF file is renamed to '{target_pdf}' in the project root.")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_retrieved_context" not in st.session_state:
    st.session_state.last_retrieved_context = "No document context called yet."

# Progress feedback during initialization to prevent infinite spinning illusion
if "agent_engine" not in st.session_state:
    with st.status("🚀 Initializing Vector Index & Agent Pipeline...", expanded=True) as status:
        st.write("📥 Loading FastEmbed ONNX models...")
        _ = get_embedding_model()
        st.write("📄 Chunking document and populating Chroma vector DB...")
        st.session_state.agent_engine = build_agentic_pipeline(target_pdf)
        status.update(label="✅ System Ready!", state="complete", expanded=False)

agent_engine = st.session_state.agent_engine


# 7. Sidebar Controls
with st.sidebar:
    st.markdown("### 📊 Judge Controls")
    enable_eval = st.checkbox("Enable LLM-as-a-Judge", value=True, help="Runs an automated quality evaluation on each response.")
    if st.button("🗑 Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# 8. Render Existing Chat Feed
for role, message in st.session_state.chat_history:
    if role == "human":
        with st.chat_message("user"):
            st.markdown(message)
    elif role == "ai":
        with st.chat_message("assistant"):
            st.markdown(message)

# 9. Process User Input
user_query = st.chat_input("Ask a technical specification question...")

if user_query:
    is_valid, validated_query = sanitize_and_check_input(user_query)
    
    if not is_valid:
        st.warning(validated_query)
    else:
        with st.chat_message("user"):
            st.markdown(validated_query)
            
        st.session_state.last_retrieved_context = "No document context called yet (Web Fallback applied)."
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Processing technical prompt..."):
                try:
                    # Truncate memory context to fit Groq API limits
                    recent_history = st.session_state.chat_history[-6:]
                    langchain_history = []
                    for role, text in recent_history:
                        if role == "human":
                            langchain_history.append(HumanMessage(content=text))
                        elif role == "ai":
                            langchain_history.append(AIMessage(content=text))

                    response = agent_engine.invoke({
                        "input": validated_query,
                        "chat_history": langchain_history
                    })
                    
                    output_text = response["output"]
                    message_placeholder.markdown(output_text)
                    
                    if enable_eval:
                        st.markdown("---")
                        st.markdown("**⚖️ Real-time QA Evaluation:**")
                        score_card = run_llm_judge(validated_query, output_text, st.session_state.last_retrieved_context)
                        st.code(score_card, language="text")

                    st.session_state.chat_history.append(("human", validated_query))
                    st.session_state.chat_history.append(("ai", output_text))

                except Exception as e:
                    err_msg = str(e)
                    if "rate_limit_exceeded" in err_msg or "413" in err_msg:
                        st.error("⏳ **Rate Limit Exceeded**: Request payload exceeded Groq limits. Clear chat history or wait 60 seconds.")
                    elif "APIKey" in err_msg or "authentication" in err_msg.lower():
                        st.error("🔑 **Authentication Error**: Groq API Key invalid or missing. Verify environment secrets.")
                    else:
                        st.error(f"❌ **System Error**: {err_msg}")
