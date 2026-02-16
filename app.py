from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
import re
from collections import Counter
import math

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'documents'
HISTORY_FILE = 'history.json'
VECTOR_INDEX_FILE = 'vector_index.json'
MAX_HISTORY = 5
CHUNK_SIZE = 500

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Groq client
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# Simple in-memory vector store (replaces ChromaDB)
vector_store = {'chunks': [], 'embeddings': [], 'metadata': []}

# Initialize history and vector store files
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w') as f:
        json.dump([], f)

if not os.path.exists(VECTOR_INDEX_FILE):
    with open(VECTOR_INDEX_FILE, 'w') as f:
        json.dump(vector_store, f)


def load_history():
    """Load query history from file"""
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except:
        return []


def save_history(history):
    """Save query history to file"""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history[-MAX_HISTORY:], f, indent=2)


def load_vector_store():
    """Load vector store from file"""
    global vector_store
    try:
        with open(VECTOR_INDEX_FILE, 'r') as f:
            vector_store = json.load(f)
    except:
        vector_store = {'chunks': [], 'embeddings': [], 'metadata': []}


def save_vector_store():
    """Save vector store to file"""
    with open(VECTOR_INDEX_FILE, 'w') as f:
        json.dump(vector_store, f)


def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Split text into overlapping chunks"""
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        
        if current_length >= chunk_size:
            chunks.append(' '.join(current_chunk))
            # Overlap: keep last 50 words
            current_chunk = current_chunk[-50:]
            current_length = sum(len(w) + 1 for w in current_chunk)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def simple_embedding(text):
    """Create simple TF-IDF style embedding (no external dependencies)"""
    words = re.findall(r'\w+', text.lower())
    word_freq = Counter(words)
    
    # Create normalized vector
    total = sum(word_freq.values())
    vector = {word: freq/total for word, freq in word_freq.items()}
    
    return vector


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two sparse vectors"""
    # Get common words
    common_words = set(vec1.keys()) & set(vec2.keys())
    
    if not common_words:
        return 0.0
    
    # Calculate dot product
    dot_product = sum(vec1[word] * vec2[word] for word in common_words)
    
    # Calculate magnitudes
    mag1 = math.sqrt(sum(val**2 for val in vec1.values()))
    mag2 = math.sqrt(sum(val**2 for val in vec2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (mag1 * mag2)


def add_document_to_vectordb(filename, content):
    """Add document chunks to vector store"""
    load_vector_store()
    
    # Remove existing chunks for this document
    indices_to_remove = []
    for i, meta in enumerate(vector_store['metadata']):
        if meta['filename'] == filename:
            indices_to_remove.append(i)
    
    for i in sorted(indices_to_remove, reverse=True):
        del vector_store['chunks'][i]
        del vector_store['embeddings'][i]
        del vector_store['metadata'][i]
    
    chunks = chunk_text(content)
    
    if not chunks:
        return
    
    for i, chunk in enumerate(chunks):
        embedding = simple_embedding(chunk)
        chunk_id = f"{filename}_chunk_{i}"
        
        vector_store['chunks'].append(chunk)
        vector_store['embeddings'].append(embedding)
        vector_store['metadata'].append({
            'filename': filename,
            'chunk_id': i,
            'id': chunk_id
        })
    
    save_vector_store()


def search_similar_chunks(query, n_results=5):
    """Search for similar chunks using cosine similarity"""
    load_vector_store()
    
    if not vector_store['chunks']:
        return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
    
    query_embedding = simple_embedding(query)
    
    # Calculate similarities
    similarities = []
    for i, embedding in enumerate(vector_store['embeddings']):
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((similarity, i))
    
    # Sort by similarity (highest first)
    similarities.sort(reverse=True)
    
    # Get top results
    top_results = similarities[:n_results]
    
    documents = [[vector_store['chunks'][idx] for _, idx in top_results]]
    metadatas = [[vector_store['metadata'][idx] for _, idx in top_results]]
    distances = [[1 - sim for sim, _ in top_results]]  # Convert similarity to distance
    
    return {
        'documents': documents,
        'metadatas': metadatas,
        'distances': distances
    }


def generate_answer_with_groq(question, context_chunks, stream=False):
    """Generate answer using Groq LLM"""
    # Prepare context from retrieved chunks
    context = "\n\n".join([
        f"[Source: {meta['filename']}]\n{doc}" 
        for doc, meta in zip(context_chunks['documents'][0], context_chunks['metadatas'][0])
    ])
    
    # Create prompt
    prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the context provided
- Be clear, concise, and easy to understand
- If the context doesn't contain enough information, say so
- Cite which source document you're using in your answer
- Use simple language that anyone can understand

Answer:"""

    try:
        # Call Groq API with streaming
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",  # Fast and accurate (updated model)
            temperature=0.3,  # Lower = more focused answers
            max_tokens=800,
            stream=stream  # Enable streaming
        )
        
        if stream:
            return chat_completion  # Return generator for streaming
        else:
            return chat_completion.choices[0].message.content
    
    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}\n\nPlease check your GROQ_API_KEY in the .env file."
        if stream:
            return iter([error_msg])  # Return as iterable for streaming
        return error_msg


@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('static', 'index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)


@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Upload a document and add to vector database"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.txt'):
            return jsonify({'error': 'Only .txt files are allowed'}), 400
        
        # Read content directly from uploaded file
        content = file.read().decode('utf-8')
        
        if not content.strip():
            return jsonify({'error': 'File is empty'}), 400
        
        # Try to save file (optional - may fail on ephemeral filesystems)
        try:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"⚠️ Could not save file to disk: {e}")
        
        # Add to vector database (this is what matters)
        add_document_to_vectordb(file.filename, content)
        print(f"✅ Indexed: {file.filename}")
        
        return jsonify({
            'message': 'File uploaded and indexed successfully',
            'filename': file.filename
        })
    
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/api/documents', methods=['GET'])
def list_documents():
    """Get list of all documents"""
    try:
        load_vector_store()
        
        # Get unique filenames from vector store
        seen_files = {}
        for meta in vector_store['metadata']:
            filename = meta['filename']
            if filename not in seen_files:
                seen_files[filename] = {
                    'name': filename,
                    'chunks': 1,
                    'uploaded': 'Recently'
                }
            else:
                seen_files[filename]['chunks'] += 1
        
        # Try to get file stats from filesystem if available
        documents = []
        for filename, info in seen_files.items():
            try:
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.exists(filepath):
                    stat = os.stat(filepath)
                    documents.append({
                        'name': filename,
                        'size': stat.st_size,
                        'uploaded': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    })
                else:
                    # File not on disk, use vector store info
                    documents.append({
                        'name': filename,
                        'size': info['chunks'] * 500,  # Estimate based on chunks
                        'uploaded': info['uploaded']
                    })
            except:
                # Fallback to vector store info
                documents.append({
                    'name': filename,
                    'size': info['chunks'] * 500,
                    'uploaded': info['uploaded']
                })
        
        return jsonify({'documents': documents})
    
    except Exception as e:
        print(f"❌ Error listing documents: {e}")
        return jsonify({'documents': []})


@app.route('/api/documents/<filename>', methods=['DELETE'])
def delete_document(filename):
    """Delete a document from filesystem and vector store"""
    try:
        # Remove from filesystem if it exists
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"✅ Deleted file: {filename}")
        
        # Remove from vector store (this is what matters)
        load_vector_store()
        indices_to_remove = []
        for i, meta in enumerate(vector_store['metadata']):
            if meta['filename'] == filename:
                indices_to_remove.append(i)
        
        if not indices_to_remove:
            return jsonify({'error': 'Document not found'}), 404
        
        for i in sorted(indices_to_remove, reverse=True):
            del vector_store['chunks'][i]
            del vector_store['embeddings'][i]
            del vector_store['metadata'][i]
        
        save_vector_store()
        print(f"✅ Removed {filename} from vector store")
        
        return jsonify({'message': 'Document deleted'})
    
    except Exception as e:
        print(f"❌ Error deleting document: {e}")
        return jsonify({'error': f'Delete failed: {str(e)}'}), 500


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Answer a question using RAG (Retrieval-Augmented Generation)"""
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    # Check if we have documents
    load_vector_store()
    if not vector_store['chunks']:
        return jsonify({
            'answer': '❌ No documents available. Please upload some documents first.',
            'sources': [],
            'question': question
        })
    
    try:
        # 1. RETRIEVAL: Search for relevant chunks
        search_results = search_similar_chunks(question, n_results=5)
        
        if not search_results['documents'][0]:
            return jsonify({
                'answer': '❌ No relevant information found. Try uploading more documents or rephrasing your question.',
                'sources': [],
                'question': question
            })
        
        # 2. GENERATION: Use Groq to generate answer
        answer = generate_answer_with_groq(question, search_results)
        
        # 3. Prepare sources with relevance scores
        sources = []
        seen_files = set()
        for i, (doc, meta, distance) in enumerate(zip(
            search_results['documents'][0],
            search_results['metadatas'][0],
            search_results['distances'][0]
        )):
            filename = meta['filename']
            # Convert distance to relevance percentage (lower distance = higher relevance)
            relevance = max(0, (1 - distance) * 100)
            
            if filename not in seen_files:
                seen_files.add(filename)
                sources.append({
                    'document': filename,
                    'snippet': doc[:300] + ('...' if len(doc) > 300 else ''),
                    'relevance': round(relevance, 1)
                })
        
        response = {
            'answer': f"🤖 AI-Generated Answer:\n\n{answer}",
            'sources': sources[:3],  # Top 3 sources
            'question': question
        }
        
        # Save to history
        history = load_history()
        history.append({
            'question': question,
            'answer': response['answer'],
            'sources': response['sources'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        save_history(history)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'answer': f'❌ Error: {str(e)}\n\nPlease make sure your GROQ_API_KEY is set correctly in the .env file.',
            'sources': [],
            'question': question
        })


@app.route('/api/ask-stream', methods=['POST'])
def ask_question_stream():
    """Answer a question using RAG with streaming response"""
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    # Check if we have documents
    load_vector_store()
    if not vector_store['chunks']:
        def error_stream():
            yield f"data: {json.dumps({'type': 'answer', 'content': '❌ No documents available. Please upload some documents first.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream')
    
    def generate():
        try:
            # 1. RETRIEVAL: Search for relevant chunks
            search_results = search_similar_chunks(question, n_results=5)
            
            if not search_results['documents'][0]:
                yield f"data: {json.dumps({'type': 'answer', 'content': '❌ No relevant information found.'})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
                return
            
            # Send initial message
            yield f"data: {json.dumps({'type': 'answer', 'content': '🤖 AI-Generated Answer:\\n\\n'})}\n\n"
            
            # 2. GENERATION: Use Groq to generate streaming answer
            stream = generate_answer_with_groq(question, search_results, stream=True)
            
            full_answer = "🤖 AI-Generated Answer:\n\n"
            
            # Stream chunks as they arrive
            for chunk in stream:
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_answer += content
                    yield f"data: {json.dumps({'type': 'answer', 'content': content})}\n\n"
            
            # 3. Prepare sources
            sources = []
            seen_files = set()
            for i, (doc, meta, distance) in enumerate(zip(
                search_results['documents'][0],
                search_results['metadatas'][0],
                search_results['distances'][0]
            )):
                filename = meta['filename']
                relevance = max(0, (1 - distance) * 100)
                
                if filename not in seen_files:
                    seen_files.add(filename)
                    sources.append({
                        'document': filename,
                        'snippet': doc[:300] + ('...' if len(doc) > 300 else ''),
                        'relevance': round(relevance, 1)
                    })
            
            # Send sources and completion
            yield f"data: {json.dumps({'type': 'done', 'sources': sources[:3]})}\n\n"
            
            # Save to history
            history = load_history()
            history.append({
                'question': question,
                'answer': full_answer,
                'sources': sources[:3],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            save_history(history)
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': f'❌ Error: {str(e)}'})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get query history"""
    history = load_history()
    return jsonify({'history': list(reversed(history))})


@app.route('/api/history', methods=['DELETE'])
def clear_history():
    """Clear query history"""
    save_history([])
    return jsonify({'message': 'History cleared'})


if __name__ == '__main__':
    # Re-index existing documents on startup
    print("🔄 Re-indexing existing documents...")
    load_vector_store()
    
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.endswith('.txt'):
            # Check if already indexed
            already_indexed = any(meta['filename'] == filename for meta in vector_store['metadata'])
            if not already_indexed:
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    add_document_to_vectordb(filename, f.read())
                    print(f"✅ Indexed: {filename}")
    
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting server on port {port}...")
    print("💡 Using Groq (Llama 3) for AI-generated answers\n")
    app.run(host='0.0.0.0', port=port, debug=False)
