# project.py

import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader


import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter





# --- App Config ---
st.set_page_config(page_title="AI Textbook Tutor", layout="wide")
st.title("📘Chemistry Textbook Tutor ")

# Load and split textbook
loader = PyPDFLoader(r"C:\Users\Open User\Desktop\AJ_class\chemistry.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
docs_recursive = text_splitter.split_documents(documents)



# --- Config ---
CHROMA_DB_DIR = "chroma_db"  # already embedded DB
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"


# --- Load Vector DB & LLM ---
embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embedding_function
)
retriever = vectorstore.as_retriever()

llm = ChatOllama(model=LLM_MODEL, temperature=0)

# --- Prompt Template ---
template = """
You are a helpful secondary school tutor. You must answer the student's question using **only the information in the context below**.

Follow these steps:
1. Begin by explaining any related or prior knowledge found in the context.
2. Then give a clear answer to the question.
3. Provide **at least 2 simple examples or illustrations**, if they are available in the context.
4. provide a likely WASSCE question from the context

Very Important:
- If the context does **not contain enough information, respond with:  
  "I don't know. The textbook does not provide enough information to answer this."**
- Do not guess or include anything that is not directly in the textbook.
- Keep your explanation simple and clear, suitable for secondary school students.
- Use bullet points or step-by-step explanations where helpful.

Context:
{context}

Question: {question}

Answer:
"""
def get_safe_answer(question):
    # Get documents from retriever
    retrieved_docs = retriever.get_relevant_documents(question)
    
    # Combine content into a single string
    context_text = " ".join([doc.page_content for doc in retrieved_docs])
    
    # If there's not enough context, skip the LLM entirely
    if not context_text.strip() or len(context_text.strip()) < 100:
        return "I don't know. The textbook does not provide enough information to answer this."
    
    # Otherwise, proceed with the RAG chain
    return rag_chain.invoke(question)

prompt = ChatPromptTemplate.from_template(template)

# --- Build Chain ---
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Chat UI ---
st.subheader("Ask a question based on your textbook:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Enter your question:")

if st.button("Submit Question") and question:
    with st.spinner("Thinking..."):
        response = rag_chain.invoke(question)

    st.session_state.chat_history.append((question, response))

# --- Display Chat History ---
for q, a in reversed(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(f"**You:** {q}")
    with st.chat_message("assistant"):
        st.markdown(a)
        col1, col2 = st.columns(2)
        with col1:
            st.button("👍 Helpful", key=f"{q}_yes")
        with col2:
            st.button("👎 Not Helpful", key=f"{q}_no")

