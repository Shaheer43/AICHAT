# app.py
import streamlit as st
from rag_chain import (
    load_faiss_index,
    get_rag_chain_with_memory
)

st.set_page_config(page_title="AlfaBOT")

st.title("💼 Alfabet Document Q&A")
st.markdown("Ask questions about internal Alfabet documents. Relevant document sources will be cited.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Cache the FAISS index
@st.cache_resource
def get_vectorstore():
    return load_faiss_index()

# Cache the RAG chain with memory
@st.cache_resource
def get_chain():
    vectorstore = get_vectorstore()
    return get_rag_chain_with_memory(vectorstore)

# ⏱️ PRELOAD chain ONCE
chain = get_chain()

# --- CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.markdown("**📄 Sources:**")
            for source, page in msg["sources"]:
                st.markdown(f"- **{source}**, Page `{page}`")

# --- CHAT INPUT ---
query = st.chat_input("Ask a question:")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if len(query.strip()) < 5 or not any(char.isalpha() for char in query):
                    answer = "That doesn't look like a real question. Can you rephrase it?"
                    sources = []
                else:
                    result = chain.invoke({"question": query})
                    answer = result.get("answer", "I couldn't generate an answer.")

                    if answer.strip().lower().startswith("[generic]"):
                        answer = answer.removeprefix("[generic]").strip()
                        sources = []
                    else:
                        docs = result.get("source_documents", [])
                        sources = {
                            (doc.metadata.get("source", "Unknown"), doc.metadata.get("page", "?"))
                            for doc in docs if doc.page_content.strip()
                        }
            except Exception as e:
                answer = f"Something went wrong: `{e}`"
                sources = []

        st.markdown(answer)
        if sources:
            st.markdown("**📄 Sources:**")
            for source, page in sorted(sources):
                st.markdown(f"- **{source}**, Page `{page}`")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sorted(sources) if sources else []
        })
