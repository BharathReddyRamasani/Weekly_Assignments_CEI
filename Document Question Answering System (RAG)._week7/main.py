import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import shutil
import tempfile

from rag_pipeline import process_and_index_document, create_qa_chain, answer_question

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)s │ %(message)s")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 20
ALLOWED_EXTENSIONS = {".pdf", ".txt"}

# ── Shared state ───────────────────────────────────────────────────────────────
# Lock prevents a mid-query chain swap corrupting results
import asyncio
_state_lock = asyncio.Lock()
app_state: dict = {"qa_chain": None, "doc_name": None}

# ── Pydantic models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty.")
        if len(v) > 2000:
            raise ValueError("Message too long (max 2 000 characters).")
        return v

class ChatResponse(BaseModel):
    answer: str
    doc_name: str | None = None

class UploadResponse(BaseModel):
    status: str
    message: str

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG.ai – Professional Document Q&A",
    version="1.0.0",
    docs_url="/api/docs",
)

# Allow same-origin and localhost for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Receives a PDF or TXT file, chunks it, embeds it, and stores the QA chain."""

    # 1. Validate file extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Only PDF and TXT are allowed."
        )

    # 2. Read content & enforce size limit
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large ({size_mb:.1f} MB). Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
        )

    logger.info("Upload received: '%s' (%.2f MB)", file.filename, size_mb)

    # 3. Write to temp file safely; ensure cleanup even on error
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            temp_path = tmp.name

        # 4. Process document (CPU-bound; runs synchronously but acceptable for local use)
        vectorstore = process_and_index_document(temp_path)

        # 5. Atomically swap global QA chain
        async with _state_lock:
            app_state["qa_chain"] = create_qa_chain(vectorstore)
            app_state["doc_name"] = file.filename

        logger.info("Document '%s' indexed successfully.", file.filename)
        return UploadResponse(
            status="success",
            message=f"'{file.filename}' processed and ready for questions!"
        )

    except HTTPException:
        raise
    except Exception as exc:
        # Log full traceback server-side; return a safe generic message to client
        logger.exception("Error processing '%s'", file.filename)
        raise HTTPException(status_code=500, detail="Failed to process the document. Please try again.") from exc

    finally:
        # Always clean up the temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
            logger.debug("Temp file '%s' deleted.", temp_path)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Accepts a question and returns a RAG-generated answer."""

    async with _state_lock:
        qa_chain = app_state.get("qa_chain")
        doc_name = app_state.get("doc_name")

    if qa_chain is None:
        raise HTTPException(
            status_code=400,
            detail="No document has been uploaded yet. Please upload a document first."
        )

    try:
        answer = answer_question(qa_chain, request.message)
        return ChatResponse(answer=answer, doc_name=doc_name)
    except Exception as exc:
        logger.exception("Error answering question.")
        raise HTTPException(status_code=500, detail="Could not generate an answer. Please try again.") from exc


@app.get("/api/status")
async def get_status():
    """Returns whether a document is currently loaded."""
    async with _state_lock:
        return {
            "document_loaded": app_state.get("qa_chain") is not None,
            "doc_name": app_state.get("doc_name"),
        }


@app.post("/api/reset")
async def reset():
    """Clears the currently loaded document and QA chain. Useful for testing."""
    async with _state_lock:
        app_state["qa_chain"] = None
        app_state["doc_name"] = None
    logger.info("State reset — document cleared.")
    return {"status": "success", "message": "Document cleared. Please upload a new document."}


# ── Static files (must be last) ────────────────────────────────────────────────
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
