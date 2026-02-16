// Use relative path for API - works both locally and on deployed server
const API_BASE = window.location.hostname === 'localhost' 
    ? 'http://localhost:5000/api' 
    : '/api';

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadDocuments();
    loadHistory();
    checkStatus();
    
    // Check status every 30 seconds
    setInterval(checkStatus, 30000);
    
    // File input change handler
    document.getElementById('fileInput').addEventListener('change', uploadFile);
    
    // Enter key to submit question
    document.getElementById('questionInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            askQuestion();
        }
    });
});

// Show notification
function showNotification(message, type = 'success') {
    const notification = document.getElementById('notification');
    notification.textContent = message;
    notification.className = `notification ${type} show`;
    
    setTimeout(() => {
        notification.classList.remove('show');
    }, 3000);
}

// Check system status
async function checkStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const status = await response.json();
        
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        
        // Determine overall health
        let overallStatus = 'healthy';
        let statusMessage = [];
        
        if (status.backend === 'healthy') {
            statusMessage.push('Backend ✓');
        }
        
        if (status.database.status === 'healthy') {
            statusMessage.push(`DB: ${status.database.documents} docs`);
        } else {
            overallStatus = 'error';
            statusMessage.push('DB ✗');
        }
        
        if (status.llm.status === 'healthy') {
            statusMessage.push('LLM ✓');
        } else if (status.llm.status === 'error') {
            overallStatus = 'error';
            statusMessage.push('LLM ✗');
        }
        
        // Update UI
        statusDot.className = `status-dot ${overallStatus}`;
        statusText.textContent = statusMessage.join(' • ');
        
    } catch (error) {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        statusDot.className = 'status-dot error';
        statusText.textContent = 'Connection error';
    }
}

// Upload file
async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) return;
    
    if (!file.name.endsWith('.txt')) {
        showNotification('Only .txt files are allowed', 'error');
        fileInput.value = '';
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        console.log('Uploading to:', `${API_BASE}/upload`);
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Upload error response:', errorText);
            try {
                const errorData = JSON.parse(errorText);
                showNotification(errorData.error || 'Upload failed', 'error');
            } catch {
                showNotification(`Upload failed: ${response.status} ${response.statusText}`, 'error');
            }
            return;
        }
        
        const data = await response.json();
        console.log('Upload success:', data);
        
        showNotification(`✓ ${file.name} uploaded successfully`);
        loadDocuments();
        fileInput.value = ''; // Reset input
        
    } catch (error) {
        console.error('Upload error:', error);
        showNotification(`Failed to upload: ${error.message}`, 'error');
    }
}

// Load documents
async function loadDocuments() {
    try {
        const response = await fetch(`${API_BASE}/documents`);
        const data = await response.json();
        
        const documentsList = document.getElementById('documentsList');
        
        if (data.documents.length === 0) {
            documentsList.innerHTML = '<p class="empty-state">No documents uploaded yet</p>';
            return;
        }
        
        documentsList.innerHTML = data.documents.map(doc => `
            <div class="document-item">
                <div class="document-info">
                    <div class="document-name">${escapeHtml(doc.name)}</div>
                    <div class="document-meta">
                        ${formatBytes(doc.size)} • ${doc.uploaded}
                    </div>
                </div>
                <div class="document-actions">
                    <button class="btn btn-small btn-danger" onclick="deleteDocument('${escapeHtml(doc.name)}')">
                        🗑️ Delete
                    </button>
                </div>
            </div>
        `).join('');
    } catch (error) {
        showNotification('Failed to load documents', 'error');
    }
}

// Delete document
async function deleteDocument(filename) {
    if (!confirm(`Delete "${filename}"?`)) return;
    
    try {
        const response = await fetch(`${API_BASE}/documents/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            showNotification(`✓ ${filename} deleted`);
            loadDocuments();
        } else {
            showNotification('Delete failed', 'error');
        }
    } catch (error) {
        showNotification('Delete failed: ' + error.message, 'error');
    }
}

// Ask question with streaming
async function askQuestion() {
    const questionInput = document.getElementById('questionInput');
    const question = questionInput.value.trim();
    
    if (!question) {
        showNotification('Please enter a question', 'error');
        return;
    }
    
    const answerSection = document.getElementById('answerSection');
    const answerText = document.getElementById('answerText');
    const sourcesSection = document.getElementById('sourcesSection');
    const sourcesList = document.getElementById('sourcesList');
    
    // Show loading state
    answerSection.style.display = 'block';
    answerText.textContent = '🔍 Searching documents...';
    sourcesSection.style.display = 'none';
    
    try {
        // Use streaming endpoint
        const response = await fetch(`${API_BASE}/ask-stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });
        
        if (!response.ok) {
            throw new Error('Failed to get answer');
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullAnswer = '';
        
        // Clear answer text for streaming
        answerText.textContent = '';
        
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop(); // Keep incomplete line in buffer
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const jsonData = JSON.parse(line.substring(6));
                    
                    if (jsonData.type === 'answer') {
                        // Append content letter by letter
                        fullAnswer += jsonData.content;
                        answerText.textContent = fullAnswer;
                        
                        // Auto-scroll to bottom
                        answerText.scrollTop = answerText.scrollHeight;
                    } else if (jsonData.type === 'done') {
                        // Show sources
                        if (jsonData.sources && jsonData.sources.length > 0) {
                            sourcesSection.style.display = 'block';
                            sourcesList.innerHTML = jsonData.sources.map(source => `
                                <div class="source-item">
                                    <div class="source-header">
                                        <span class="source-document">📄 ${escapeHtml(source.document)}</span>
                                        <span class="source-relevance">${source.relevance}% relevant</span>
                                    </div>
                                    <div class="source-snippet">${escapeHtml(source.snippet)}</div>
                                </div>
                            `).join('');
                        } else {
                            sourcesSection.style.display = 'none';
                        }
                        
                        // Reload history
                        loadHistory();
                    } else if (jsonData.type === 'error') {
                        answerText.textContent = jsonData.content;
                        sourcesSection.style.display = 'none';
                    }
                }
            }
        }
        
        // Clear question input
        questionInput.value = '';
        
    } catch (error) {
        answerText.textContent = 'Failed to get answer: ' + error.message;
        sourcesSection.style.display = 'none';
    }
}

// Load history
async function loadHistory() {
    try {
        const response = await fetch(`${API_BASE}/history`);
        const data = await response.json();
        
        const historyList = document.getElementById('historyList');
        
        if (data.history.length === 0) {
            historyList.innerHTML = '<p class="empty-state">No queries yet</p>';
            return;
        }
        
        historyList.innerHTML = data.history.map(item => `
            <div class="history-item" onclick="showHistoryItem(${escapeHtml(JSON.stringify(item))})">
                <div class="history-question">Q: ${escapeHtml(item.question)}</div>
                <div class="history-answer">${escapeHtml(item.answer)}</div>
                <div class="history-timestamp">${item.timestamp}</div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Failed to load history:', error);
    }
}

// Show history item
function showHistoryItem(item) {
    const answerSection = document.getElementById('answerSection');
    const answerText = document.getElementById('answerText');
    const sourcesSection = document.getElementById('sourcesSection');
    const sourcesList = document.getElementById('sourcesList');
    const questionInput = document.getElementById('questionInput');
    
    // Show the question and answer
    questionInput.value = item.question;
    answerSection.style.display = 'block';
    answerText.textContent = item.answer;
    
    if (item.sources && item.sources.length > 0) {
        sourcesSection.style.display = 'block';
        sourcesList.innerHTML = item.sources.map(source => `
            <div class="source-item">
                <div class="source-header">
                    <span class="source-document">📄 ${escapeHtml(source.document)}</span>
                    <span class="source-relevance">${source.relevance}% relevant</span>
                </div>
                <div class="source-snippet">${escapeHtml(source.snippet)}</div>
            </div>
        `).join('');
    } else {
        sourcesSection.style.display = 'none';
    }
    
    // Scroll to answer
    answerSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Clear history
async function clearHistory() {
    if (!confirm('Clear all history?')) return;
    
    try {
        const response = await fetch(`${API_BASE}/history`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            showNotification('✓ History cleared');
            loadHistory();
        }
    } catch (error) {
        showNotification('Failed to clear history', 'error');
    }
}

// Utility functions
function escapeHtml(text) {
    if (typeof text !== 'string') {
        text = JSON.stringify(text);
    }
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}
