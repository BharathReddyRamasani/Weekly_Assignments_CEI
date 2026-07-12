import os
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Validate API key at startup ────────────────────────────────────────────────
_GROQ_KEY = os.getenv("GROQ_API_KEY", "")
if not _GROQ_KEY or _GROQ_KEY.startswith("your_"):
    raise EnvironmentError(
        "GROQ_API_KEY is not set or still contains the placeholder value. "
        "Please add your real key to the .env file."
    )

# ── Embeddings (loaded once, shared across requests) ───────────────────────────
logger.info("Loading HuggingFace embedding model…")
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
logger.info("Embedding model ready.")


def load_documents(file_path: str):
    """Loads a PDF or TXT file and returns a list of Document objects."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Only .pdf and .txt are supported.")

    docs = loader.load()
    if not docs:
        raise ValueError("The document appears to be empty or could not be parsed.")
    return docs


def process_and_index_document(file_path: str):
    """
    Full pipeline:
      1. Load document
      2. Split into overlapping chunks
      3. Embed and index into an in-memory FAISS vector store
    Returns the FAISS vectorstore.
    """
    logger.info("Loading document: %s", file_path)
    documents = load_documents(file_path)

    logger.info("Splitting into chunks…")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,   # stores byte-offset metadata for future source attribution
    )
    chunks = splitter.split_documents(documents)
    logger.info("Created %d chunks.", len(chunks))

    if not chunks:
        raise ValueError("Document produced zero chunks after splitting. It may be too short or empty.")

    vectorstore = FAISS.from_documents(chunks, EMBEDDINGS)
    logger.info("FAISS index built with %d vectors.", len(chunks))
    return vectorstore


def format_docs(docs) -> str:
    """Joins retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_qa_chain(vectorstore):
    """
    Builds and returns an LCEL RAG chain:
      retriever → prompt → LLM → string output
    """
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3)

    system_prompt = (
        "You are a helpful assistant that answers questions strictly based on the provided document context. "
        "Do NOT use outside knowledge. "
        "If the answer is not found in the context, say: 'I could not find that information in the document.' "
        "Be concise and accurate.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},   # retrieve 4 chunks for better coverage
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("QA chain created.")
    return rag_chain


def answer_question(rag_chain, query: str) -> str:
    """Invokes the RAG chain and returns the string answer."""
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")
    return rag_chain.invoke(query.strip())
