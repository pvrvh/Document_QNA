# Private Knowledge Q&A

A web application for uploading documents and asking questions about them with source attribution.

## Features

- 📤 Upload text documents (.txt files)
- 📋 View list of uploaded documents
- ❓ Ask questions about your documents
- 💡 Get answers with source attribution
- 🕐 View history of last 5 queries
- 🎯 TF-IDF based relevance scoring

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. **Upload Documents**: Click "Upload Document" to add .txt files
2. **Ask Questions**: Type your question in the text area and click "Ask Question"
3. **View Sources**: See which documents were used and their relevance scores
4. **Check History**: View your last 5 questions and answers

## Tech Stack

- **Backend**: Python Flask
- **Frontend**: HTML, CSS, JavaScript (Vanilla)
- **Search**: TF-IDF algorithm for document relevance
- **Storage**: File system for documents, JSON for history

## Project Structure

```
qna/
├── app.py              # Flask backend
├── requirements.txt    # Python dependencies
├── static/
│   ├── index.html     # Main HTML page
│   ├── styles.css     # Styling
│   └── script.js      # Frontend logic
├── documents/         # Uploaded documents (created automatically)
└── history.json       # Query history (created automatically)
```
