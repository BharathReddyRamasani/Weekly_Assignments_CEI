/**
 * RAG.ai – Frontend JavaScript
 *
 * Security notes:
 *  - All user-supplied text is set via textContent / createTextNode, never innerHTML,
 *    to prevent XSS injection.
 *  - Answers from the server are also set via textContent (safe).
 */

document.addEventListener('DOMContentLoaded', () => {

    /* ── Element refs ──────────────────────────────────────────── */
    const dropZone        = document.getElementById('drop-zone');
    const fileInput       = document.getElementById('file-input');
    const uploadStatus    = document.getElementById('upload-status');
    const connectionBadge = document.getElementById('connection-status');
    const badgeText       = connectionBadge.querySelector('.badge-text');
    const chatInput       = document.getElementById('chat-input');
    const sendBtn         = document.getElementById('send-btn');
    const chatForm        = document.getElementById('chat-form');
    const chatHistory     = document.getElementById('chat-history');
    const charCounter     = document.getElementById('char-counter');
    const docNameDisplay  = document.getElementById('doc-name');
    const dropHint        = document.getElementById('drop-hint');
    const dropIcon        = document.getElementById('drop-icon');

    /* ── State ─────────────────────────────────────────────────── */
    let isUploading  = false;
    let isSending    = false;
    const MAX_CHARS  = 2000;

    /* ── Drag & Drop ───────────────────────────────────────────── */
    dropZone.addEventListener('click', () => { if (!isUploading) fileInput.click(); });

    dropZone.addEventListener('keydown', (e) => {
        if ((e.key === 'Enter' || e.key === ' ') && !isUploading) {
            e.preventDefault();
            fileInput.click();
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        if (!isUploading) dropZone.classList.add('drag-over');
    });

    ['dragleave', 'dragend'].forEach(evt =>
        dropZone.addEventListener(evt, () => dropZone.classList.remove('drag-over'))
    );

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (!isUploading && e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) handleFileUpload(fileInput.files[0]);
        fileInput.value = '';   // reset so same file can be re-uploaded
    });

    /* ── File upload ───────────────────────────────────────────── */
    async function handleFileUpload(file) {
        // Client-side type check
        const ext = file.name.split('.').pop().toLowerCase();
        if (!['pdf', 'txt'].includes(ext)) {
            setUploadStatus('Only PDF and TXT files are supported.', 'error');
            return;
        }

        // Client-side size check (20 MB)
        if (file.size > 20 * 1024 * 1024) {
            setUploadStatus(`File is too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max 20 MB.`, 'error');
            return;
        }

        isUploading = true;
        dropZone.classList.add('uploading');
        setDropZoneContent('⏳', 'Uploading & processing…');
        setUploadStatus('Processing document…', 'pending');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData,
            });
            const result = await response.json();

            if (response.ok) {
                setUploadStatus('✓ ' + result.message, 'success');
                setDropZoneContent('✅', 'Upload another document');
                enableChat();
                updateDocName(file.name);
                addMessage('assistant', `Document loaded: **${escapeText(file.name)}**. What would you like to know?`);
            } else {
                setUploadStatus(result.detail || 'Upload failed.', 'error');
                setDropZoneContent('📄', 'Click or drag a PDF / TXT file here');
            }
        } catch (err) {
            setUploadStatus('Network error. Is the server running?', 'error');
            setDropZoneContent('📄', 'Click or drag a PDF / TXT file here');
        } finally {
            isUploading = false;
            dropZone.classList.remove('uploading');
        }
    }

    function setDropZoneContent(icon, hint) {
        dropIcon.textContent = icon;
        dropHint.textContent = hint;
    }

    function setUploadStatus(message, type) {
        uploadStatus.textContent = message;
        uploadStatus.className = `status-pill ${type}`;
        uploadStatus.classList.remove('hidden');
    }

    function updateDocName(name) {
        docNameDisplay.textContent = name;
    }

    function enableChat() {
        chatInput.disabled    = false;
        sendBtn.disabled      = false;
        chatInput.setAttribute('aria-disabled', 'false');
        connectionBadge.className = 'status-badge online';
        badgeText.textContent     = 'Document ready';
        chatInput.focus();
    }

    function disableChatWhileSending() {
        chatInput.disabled = true;
        sendBtn.disabled   = true;
    }

    function reEnableChat() {
        chatInput.disabled = false;
        sendBtn.disabled   = false;
        chatInput.focus();
    }

    /* ── Character counter ─────────────────────────────────────── */
    chatInput.addEventListener('input', () => {
        const len = chatInput.value.length;
        if (len === 0) {
            charCounter.textContent = '';
            charCounter.className   = 'char-counter';
        } else if (len > MAX_CHARS * 0.9) {
            charCounter.textContent = `${len} / ${MAX_CHARS}`;
            charCounter.className   = len >= MAX_CHARS ? 'char-counter over' : 'char-counter warn';
        } else {
            charCounter.textContent = '';
            charCounter.className   = 'char-counter';
        }
    });

    /* ── Chat form submit ──────────────────────────────────────── */
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const message = chatInput.value.trim();
        if (!message || isSending) return;

        isSending = true;
        disableChatWhileSending();
        chatInput.value       = '';
        charCounter.textContent = '';

        addMessage('user', message);
        const typingId = showTypingIndicator();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message }),
            });

            removeById(typingId);

            const result = await response.json();

            if (response.ok) {
                addMessage('assistant', result.answer);
            } else {
                addMessage('assistant', '⚠️ ' + (result.detail || 'Could not get a response. Please try again.'));
            }
        } catch (err) {
            removeById(typingId);
            addMessage('assistant', '⚠️ Network error. Please check your connection and try again.');
        } finally {
            isSending = false;
            reEnableChat();
        }
    });

    /* ── Message builder (XSS-safe via textContent) ────────────── */
    function addMessage(role, text) {
        const wrap   = document.createElement('div');
        wrap.className = `message ${role === 'user' ? 'user-msg' : 'assistant-msg'}`;

        const avatarEl = document.createElement('div');
        avatarEl.setAttribute('aria-hidden', 'true');

        const bubbleEl = document.createElement('div');
        bubbleEl.className = 'bubble';

        if (role === 'user') {
            avatarEl.className   = 'avatar user-avatar';
            avatarEl.textContent = '👤';
            bubbleEl.textContent = text;   // ✅ Safe – no innerHTML
        } else {
            avatarEl.className   = 'avatar ai-avatar';
            avatarEl.textContent = '◈';
            // For AI responses: allow simple markdown-ish bold via a safe method
            renderSafeMarkdown(bubbleEl, text);
        }

        wrap.appendChild(avatarEl);
        wrap.appendChild(bubbleEl);
        chatHistory.appendChild(wrap);
        scrollToBottom();
    }

    /**
     * Renders a subset of markdown safely (bold **text**, newlines)
     * WITHOUT using innerHTML on untrusted input.
     */
    function renderSafeMarkdown(container, text) {
        // Split on bold markers and newlines
        const parts = text.split(/(\*\*[^*]+\*\*|\n)/g);
        parts.forEach(part => {
            if (part === '\n') {
                container.appendChild(document.createElement('br'));
            } else if (/^\*\*[^*]+\*\*$/.test(part)) {
                const strong = document.createElement('strong');
                strong.textContent = part.slice(2, -2);
                container.appendChild(strong);
            } else {
                container.appendChild(document.createTextNode(part));
            }
        });
    }

    /* ── Typing indicator ──────────────────────────────────────── */
    function showTypingIndicator() {
        const id   = 'typing-' + Date.now();
        const wrap = document.createElement('div');
        wrap.id    = id;
        wrap.className = 'message assistant-msg';
        wrap.setAttribute('aria-label', 'AI is typing');

        const avatar = document.createElement('div');
        avatar.className   = 'avatar ai-avatar';
        avatar.textContent = '◈';
        avatar.setAttribute('aria-hidden', 'true');

        const bubble = document.createElement('div');
        bubble.className = 'bubble';

        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        for (let i = 0; i < 3; i++) indicator.appendChild(document.createElement('span'));

        bubble.appendChild(indicator);
        wrap.appendChild(avatar);
        wrap.appendChild(bubble);
        chatHistory.appendChild(wrap);
        scrollToBottom();
        return id;
    }

    /* ── Helpers ───────────────────────────────────────────────── */
    function removeById(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function escapeText(str) {
        return str.replace(/[&<>"']/g, c => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[c]));
    }
});
