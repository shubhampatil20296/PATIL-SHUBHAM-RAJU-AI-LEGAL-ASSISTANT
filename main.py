import streamlit as st
from dotenv import load_dotenv
import os
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from vector_database import get_embedding_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
logger.info("Loading environment variables from .env file...")
load_dotenv()
logger.info("✓ Environment variables loaded")

custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context
Question: {question} 
Context: {context} 
Answer:
"""


FAISS_DB_PATH="vectorstore/db_faiss"
logger.info(f"FAISS database path: {FAISS_DB_PATH}")

logger.info("Initializing ChatGroq LLM...")
llm_model=ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))
logger.info("✓ ChatGroq LLM initialized")

def retrieve_docs(faiss_db, query, faiss_k=5, top_k=10):
    """FAISS retrieval.
    - faiss_k: number of dense candidates
    - top_k: final number of returned documents
    """
    logger.info(f"FAISS retrieval for query: {query}")

    # FAISS dense candidates with score (if available)
    dense_candidates = []
    try:
        # similarity_search_with_score returns list of (doc, score)
        dense_results = faiss_db.similarity_search_with_score(query, k=faiss_k)
        for doc, score in dense_results:
            dense_candidates.append((doc, float(score)))
    except Exception:
        # fallback to similarity_search without scores
        docs = faiss_db.similarity_search(query, k=faiss_k)
        dense_candidates = [(d, 0.0) for d in docs]

    # Normalize FAISS scores (assume higher is better)
    faiss_scores = [s for (_, s) in dense_candidates]
    if faiss_scores:
        mn = min(faiss_scores)
        mx = max(faiss_scores)
        denom = mx - mn if mx != mn else 1.0
        norm_faiss = [(dense_candidates[i][0], (faiss_scores[i] - mn) / denom) for i in range(len(dense_candidates))]
    else:
        norm_faiss = [(dense_candidates[i][0], 0.0) for i in range(len(dense_candidates))]

    # Sort by normalized score
    norm_faiss.sort(key=lambda x: x[1], reverse=True)
    candidates_docs = [c[0] for c in norm_faiss][:top_k]

    logger.info(f"✓ Retrieved {len(candidates_docs)} documents")
    return candidates_docs

def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

def answer_query(documents, model, query):
    logger.info("Building prompt and invoking LLM...")
    context = get_context(documents)
    logger.info(f"Context: {len(context)} characters from {len(documents)} documents")
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    logger.info("Invoking ChatGroq model...")
    response = chain.invoke({"question": query, "context": context})
    logger.info("✓ Response generated successfully")
    
    # Get the response content
    answer_text = response.content if hasattr(response, 'content') else str(response)
    # Clean up the answer text - remove any prefixes and decode Unicode
    answer_text = answer_text.strip()
    if answer_text.startswith('content='):
        answer_text = answer_text[7:].strip().strip('"').strip("'")
    
    return {
        "answer": answer_text
    }


# Initialize session state for conversation history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Initialize processing state
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Display conversation history
st.title("⚖️ AI Lawyer Chat")
st.markdown("*<span style='color:red'>This system is for informational purposes only and does not constitute official legal advice. Always consult a qualified legal professional for specific legal matters.</span>*", unsafe_allow_html=True)

# Add a clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.session_state.processing = False
    st.rerun()

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Show processing indicator
if st.session_state.processing:
    with st.chat_message("assistant"):
        st.markdown("*Thinking...*")

# Chat input
if prompt := st.chat_input("Ask a legal question...", disabled=st.session_state.processing):
    if prompt.strip() and not st.session_state.processing:
        # Add user message to history immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.processing = True
        st.rerun()
# Process AI response if we're in processing state
if st.session_state.processing and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]
    
    # Load existing vector store and get answer
    try:
        faiss_db = FAISS.load_local(FAISS_DB_PATH, get_embedding_model(), allow_dangerous_deserialization=True)
        retrieved_docs = retrieve_docs(faiss_db, user_prompt)

        if retrieved_docs:
            response = answer_query(documents=retrieved_docs, model=llm_model, query=user_prompt)
            
            # Get and format the answer
            answer_text = response.get("answer", "")
            answer_text = answer_text.replace('\\n', '\n').replace('\\t', '\t')
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": answer_text})
        else:
            error_msg = "No relevant documents found in the knowledge base."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Reset processing state
    st.session_state.processing = False
    st.rerun()

