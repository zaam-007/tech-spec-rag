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
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# 1. Load security keys
load_dotenv()

# 2. Configure Streamlit Page
st.set_page_config(page_title="Technical Assistant RAG", page_icon="⚙️", layout="wide")

st.title("⚙️ Production Technical Spec RAG Assistant")
st.write("An intelligent engineering agent equipped with layout-aware memory and live evaluation capabilities.")
import re

def sanitize_and_check_input(query: str) -> tuple[bool, str]:
    """
    Validates user input against prompt injection attempts and empty inputs.
    Returns (is_valid, error_message_or_clean_query).
    """
    clean_query = query.strip()
    if not clean_query:
        return False, "Please enter a valid technical question."
    
    # Common prompt injection patterns
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

# 3. Cache the Core Agent and Retrievers
@st.cache_resource
def build_agentic_pipeline(pdf_path):
    import pymupdf4llm
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_community.document_compressors import FlashrankRerank

    # Parse local PDF to structured Markdown
    md_text = pymupdf4llm.to_markdown(pdf_path)
    docs = [Document(page_content=md_text, metadata={"source": pdf_path})]

    # Split using layout boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500,        
        chunk_overlap=400,      
        separators=["\n## ", "\n### ", "\n\n", "\n", " "] 
    )
    splits = text_splitter.split_documents(docs)
    
    # Vector store setup
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
    
    # Initialize Core LLM
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("groq_api_key")
    if not api_key:
        raise ValueError("Groq API Key not found!")

    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

    # Wrap retrievers inside a tool reference layer
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
            "First, ALWAYS use the `search_pdf_specifications` tool to search the provided document for answers.\n"
            "If the document context completely lacks the required information, "
            "seamlessly use the `duckduckgo_search` tool to look up technical concepts online.\n"
            "Analyze the conversation history to handle context-driven follow-up questions accurately.\n"
            "Be descriptive, accurate, and do not make up fake metrics."
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# 4. Helper Function: Run Real-time LLM Evaluation
def run_llm_judge(query, response, context):
    eval_llm = ChatGroq(groq_api_key=st.secrets.get("groq_api_key") or os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")
    
    # Truncate to prevent token limit issues
    safe_context = context[:8000] + "\n...[Context truncated for token limits]..." if len(context) > 8000 else context

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

# 5. Pipeline Initialization
target_pdf = "document.pdf"

if not os.path.exists(target_pdf):
    st.error(f"❌ '{target_pdf}' not found! Please drop your technical manual PDF into your project folder and rename it to '{target_pdf}'.")
    st.stop()

# Initialize persistent memory structures
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_retrieved_context" not in st.session_state:
    st.session_state.last_retrieved_context = "No document context called yet."

try:
    agent_engine = build_agentic_pipeline(target_pdf)
except Exception as e:
    st.error(f"Failed to compile RAG pipeline: {e}")
    st.stop()

# 6. Build Sidebar Controls
with st.sidebar:
    st.markdown("### 📊 Judge Controls")
    enable_eval = st.checkbox("Enable LLM-as-a-Judge", value=True, help="Runs an independent automated evaluation step on the latest response.")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# 7. Execute the request with Guardrails
if user_query:
    is_valid, result = sanitize_and_check_input(user_query)
    
    if not is_valid:
        st.warning(result)
    else:
        with st.spinner("Agent evaluating guardrails and executing tools..."):
            try:
                # Format session state messages with memory pruning [-6:]
                recent_history = st.session_state.chat_history[-6:]
                langchain_history = []
                for role, text in recent_history:
                    if role == "human":
                        langchain_history.append(("human", text))
                    elif role == "ai":
                        langchain_history.append(("ai", text))

                response = agent_engine.invoke({
                    "input": result,
                    "chat_history": langchain_history
                })
                
                output_text = response["output"]
                
                # Update memory
                st.session_state.chat_history.append(("human", result))
                st.session_state.chat_history.append(("ai", output_text))
                
                st.markdown("### 🤖 Engine Output:")
                st.info(output_text)

            except Exception as e:
                err_msg = str(e)
                if "rate_limit_exceeded" in err_msg or "413" in err_msg:
                    st.error("⏳ **Rate Limit / Token Cap Exceeded**: The request payload exceeded Groq's Tokens Per Minute (TPM) limit. Please clear chat history or wait 60 seconds.")
                elif "APIKey" in err_msg or "authentication" in err_msg.lower():
                    st.error("🔑 **Authentication Error**: Groq API Key is invalid or missing. Check your `.env` or Streamlit secrets.")
                else:
                    st.error(f"❌ **System Error**: {err_msg}")
# 8. Chat Input and Execution Lifecycle
user_query = st.chat_input("Ask a technical specification question...")

if user_query:
    # Immediately render user query on screen
    with st.chat_message("user"):
        st.markdown(user_query)
        
    st.session_state.last_retrieved_context = "No document context called yet (Web Fallback applied)."
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                # Format session state messages into LangChain baseline objects for the prompt template
                langchain_history = []
                #keeping only 3 rounds of back and forth dialogue or chat history to stay within groq's 6k token limit
                recent_history = st.session_state.chat_history[-6:]
                for role, text in recent_history:
                    if role == "human":
                        langchain_history.append(HumanMessage(content=text))
                    elif role == "ai":
                        langchain_history.append(AIMessage(content=text))

                # Execute tracking run
                response = agent_engine.invoke({
                    "input": user_query,
                    "chat_history": langchain_history
                })
                
                output_text = response["output"]
                message_placeholder.markdown(output_text)
                
                # Run the Judge metric audit directly inline if checked
                if enable_eval:
                    st.markdown("---")
                    st.markdown("**⚖️ Real-time QA Evaluation:**")
                    score_card = run_llm_judge(user_query, output_text, st.session_state.last_retrieved_context)
                    st.code(score_card, language="text")

                # Save history parameters
                st.session_state.chat_history.append(("human", user_query))
                st.session_state.chat_history.append(("ai", output_text))

            except Exception as e:
                st.error(f"An error occurred: {e}")
