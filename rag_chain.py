# rag_chain.py
import os
import time
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Load env vars
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Config
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
RETRIEVAL_K = 3

def create_faiss_index(docs, index_path="vector_store"):
    print("🔁 Embedding documents and creating FAISS index...")
    start = time.time()
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.from_documents(tqdm(docs, desc="🔗 Embedding"), embedding=embeddings)
    vectorstore.save_local(index_path)
    print(f"✅ Saved FAISS index at {index_path}/index.faiss")
    print(f"⏱️ Indexing time: {time.time() - start:.2f} seconds\n")

def load_faiss_index(index_path="vector_store"):
    print(f"📦 Loading FAISS index from: {index_path}")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

def load_documents(folder_path):
    all_chunks = []
    files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    for filename in tqdm(files, desc="📄 Processing PDFs"):
        loader = PyPDFLoader(os.path.join(folder_path, filename))
        try:
            pages = loader.load_and_split()
        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")
            continue

        for page in pages:
            page.metadata["source"] = filename
            page.metadata["page"] = page.metadata.get("page", 1)

        all_chunks.extend(splitter.split_documents(pages))

    print(f"✅ Loaded and split into {len(all_chunks)} token-based chunks\n")
    return all_chunks

def get_rag_chain_with_memory(vectorstore):
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-8b-8192")

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": RETRIEVAL_K}
    )

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=3
    )

    prompt = PromptTemplate.from_template("""
You are an intelligent assistant for technical consultants using Alfabet (version 10.15), a data-driven enterprise architecture tool.

Answer the user query using context from the retrieved documents **when possible**. If the answer cannot be generated from the documents, or if the question is unrelated or nonsensical, ALWAYS start your answer with the tag [generic].

Guidelines:
- Use relevant detail from documents when available.
- If documents do not mention a term but it is common in enterprise architecture (e.g., TOGAF, ServiceNow, SAP), explain it concisely and do not cite sources here.
- If the question is irrelevant (like "What's the weather today?"), politely say it is outside scope.

Context:
{context}

Question: {question}

Answer:
""")

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=False
    )
