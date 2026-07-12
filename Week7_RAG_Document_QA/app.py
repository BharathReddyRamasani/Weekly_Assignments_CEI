"""
RAG.ai – Streamlit Cloud App
Deploy this file to Streamlit Cloud.
Set your GROQ_API_KEY in: App Settings → Secrets → GROQ_API_KEY = "your_key"
"""

import streamlit as st
import os
import tempfile

# ── Streamlit Cloud secrets → env var bridge ──────────────────────────────────
# On Streamlit Cloud, secrets are in st.secrets.
# Locally, they come from .env via load_dotenv() inside rag_pipeline.py.
# This block bridges both so rag_pipeline.py always finds the key.
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Import AFTER injecting the key so the startup validation in rag_pipeline passes
from rag_pipeline import process_and_index_document, create_qa_chain, answer_question

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG.ai – Document Q&A",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (dark, premium look) ──────────────────────────────────────────
st.markdown("""
<style>
    /* Overall background */
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.03) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Headings */
    h1, h2, h3 { color: #f0f4ff !important; }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 14px !important;
        padding: 4px 8px !important;
    }

    /* Primary buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 8px 20px !important;
        transition: transform 0.2s;
    }
    .stButton > button:hover { transform: translateY(-2px); }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "qa_chain"      not in st.session_state: st.session_state.qa_chain      = None
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "doc_name"      not in st.session_state: st.session_state.doc_name      = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ◈ RAG**.ai**")
    st.caption("Intelligence tailored to your data")
    st.divider()

    st.markdown("### Step 1 — Upload Document")
    uploaded_file = st.file_uploader(
        "Drag or click to upload a PDF / TXT",
        type=["pdf", "txt"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        # Show file size warning
        size_mb = uploaded_file.size / (1024 * 1024)
        if size_mb > 20:
            st.error(f"File too large ({size_mb:.1f} MB). Max 20 MB.")
        else:
            st.caption(f"📄 **{uploaded_file.name}** ({size_mb:.2f} MB)")
            if st.button("Process Document", use_container_width=True):
                temp_path = None
                try:
                    ext = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                        tmp.write(uploaded_file.read())
                        temp_path = tmp.name

                    with st.spinner("Chunking, embedding and indexing…"):
                        vectorstore = process_and_index_document(temp_path)
                        st.session_state.qa_chain     = create_qa_chain(vectorstore)
                        st.session_state.doc_name     = uploaded_file.name
                        st.session_state.chat_history = []   # reset on new doc

                    st.success("✓ Document ready! Ask questions on the right.")

                except Exception as e:
                    st.error(f"Processing failed: {e}")
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)

    st.divider()

    # Active document info
    st.markdown("### Active Document")
    if st.session_state.doc_name:
        st.info(f"📄 {st.session_state.doc_name}")
        if st.button("Clear document", use_container_width=True):
            st.session_state.qa_chain     = None
            st.session_state.chat_history = []
            st.session_state.doc_name     = None
            st.rerun()
    else:
        st.caption("No document loaded yet.")

    st.divider()
    st.caption("Powered by **Groq** + **LangChain** + **FAISS**")

# ── Main chat area ────────────────────────────────────────────────────────────
st.markdown("## Step 2 — Ask Questions")

if st.session_state.qa_chain is None:
    st.info("Upload and process a document in the sidebar to start asking questions.")
else:
    # Render existing chat history
    for role, message in st.session_state.chat_history:
        st.chat_message(role).markdown(message)

    # New query input
    user_query = st.chat_input("Ask anything about the document…")

    if user_query:
        user_query = user_query.strip()
        if len(user_query) > 2000:
            st.error("Message too long (max 2 000 characters).")
        elif user_query:
            st.chat_message("user").markdown(user_query)
            st.session_state.chat_history.append(("user", user_query))

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        answer = answer_question(st.session_state.qa_chain, user_query)
                        st.markdown(answer)
                        st.session_state.chat_history.append(("assistant", answer))
                    except Exception:
                        err = "Sorry, I could not generate an answer. Please try again."
                        st.error(err)
                        st.session_state.chat_history.append(("assistant", err))
