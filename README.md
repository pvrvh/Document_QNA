# Private Knowledge Q&A System

A web application that allows you to upload documents and ask questions about them using RAG (Retrieval-Augmented Generation) with AI-powered answers.

## Features

✅ **Completed:**
- Upload and manage text documents (.txt files)
- Ask questions and get AI-generated answers with source attribution
- View query history (last 5 questions)
- Real-time streaming responses (letter-by-letter generation)
- Dark theme UI with charcoal background
- System health status indicator
- Vector-based semantic search for document retrieval
- Source relevance scoring

❌ **Not Implemented:**
- Multi-format document support (PDF, DOCX, etc.) - only .txt supported
- User authentication/multi-user support
- Persistent storage on free hosting (documents lost on restart)
- Advanced chunking strategies
- Citation with page numbers
- Document preprocessing (OCR, cleaning)

## Technology Stack

- **Backend:** Flask 3.0.0 (Python)
- **AI/LLM:** Groq API with Llama 3.3 70B model
- **Vector Search:** Custom TF-IDF implementation (lightweight)
- **Frontend:** Vanilla JavaScript, HTML5, CSS3
- **Deployment:** Render (or any Python hosting platform)

## Prerequisites

- Python 3.11 or higher
- Groq API key (get free at https://console.groq.com)
- Git (for cloning)

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/pvrvh/Document_QNA.git
cd Document_QNA
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

**Get your Groq API key:**
1. Go to https://console.groq.com
2. Sign up/login
3. Navigate to API Keys section
4. Create a new API key
5. Copy and paste it into the `.env` file

### 5. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## Usage

1. **Upload Documents:**
   - Click "➕ Upload Document"
   - Select a `.txt` file
   - Wait for confirmation

2. **Ask Questions:**
   - Type your question in the text area
   - Click "🔍 Ask Question" or press Ctrl+Enter
   - Watch the AI-generated answer stream in real-time
   - View sources and relevance scores

3. **Manage Documents:**
   - See all uploaded documents in the left panel
   - Delete documents using the 🗑️ button

4. **View History:**
   - Last 5 questions and answers shown at the bottom
   - Click "Clear History" to reset

5. **Check System Status:**
   - Status indicator in the header shows:
     - Backend health
     - Database status (number of documents)
     - LLM connection status

## Project Structure

```
qna/
├── app.py                  # Flask backend with RAG implementation
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (not in git)
├── .gitignore            # Git ignore rules
├── README.md             # This file
├── AI_NOTES.md          # AI usage documentation
├── PROMPTS_USED.md      # Development prompts log
├── render.yaml          # Render deployment config
├── static/
│   ├── index.html       # Frontend UI
│   ├── styles.css       # Dark theme styling
│   └── script.js        # Frontend logic
├── documents/           # Uploaded documents (not in git)
├── vector_index.json    # Vector database (not in git)
└── history.json         # Query history (not in git)
```

## Deployment

### Deploy on Render

1. Push code to GitHub
2. Go to https://render.com
3. Create new Web Service
4. Connect your GitHub repository
5. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
6. Add environment variable: `GROQ_API_KEY`
7. Deploy!

**Note:** Free tier has ephemeral storage - uploaded documents are lost on restart.

## How It Works

### RAG Process

1. **Document Upload:**
   - Text is split into 500-character chunks with 50-word overlap
   - Each chunk is converted to a TF-IDF vector embedding
   - Chunks stored in JSON-based vector database

2. **Question Answering:**
   - User question is converted to vector embedding
   - Top 5 most similar chunks retrieved using cosine similarity
   - Retrieved chunks sent as context to Groq's Llama 3.3 70B model
   - AI generates answer based only on provided context
   - Sources displayed with relevance scores

3. **Streaming:**
   - Uses Server-Sent Events (SSE) for real-time streaming
   - Answer appears letter-by-letter like ChatGPT

## Input Validation

- ✅ File type validation (only .txt allowed)
- ✅ Empty file check
- ✅ Empty question validation
- ✅ API key presence check
- ✅ Error handling for network failures
- ✅ Graceful degradation on service errors

## Troubleshooting

### "Upload failed"
- Check console (F12) for detailed errors
- Ensure file is .txt format
- Check file is not empty

### "No documents available"
- Upload at least one .txt file first
- Check if documents were saved (ephemeral storage issue)

### "LLM connection error"
- Verify GROQ_API_KEY in .env file
- Check API key is valid at https://console.groq.com
- Ensure internet connection

### Status shows "DB ✗"
- Vector database initialization failed
- Check write permissions in app directory

## API Endpoints

- `GET /` - Serve main page
- `GET /api/status` - System health check
- `POST /api/upload` - Upload document
- `GET /api/documents` - List all documents
- `DELETE /api/documents/<filename>` - Delete document
- `POST /api/ask-stream` - Ask question (streaming)
- `GET /api/history` - Get query history
- `DELETE /api/history` - Clear history

## Performance

- **Latency:** ~1-3s for answer generation
- **Chunk retrieval:** <100ms
- **Maximum documents:** Limited by available memory
- **Concurrent users:** Handled by gunicorn workers

## Security Notes

⚠️ **Important:**
- Never commit `.env` file to git
- API key is server-side only (not exposed to frontend)
- No authentication - suitable for personal use only
- CORS enabled - restrict in production

## License

MIT License - feel free to use and modify

## Author

**pvrvh**
- GitHub: https://github.com/pvrvh
- Repository: https://github.com/pvrvh/Document_QNA

## Acknowledgments

- **Groq** for fast LLM inference
- **Llama 3.3 70B** by Meta for the language model
- Built with assistance from GitHub Copilot
