# Development Prompts Log

This document chronicles the development process and key decisions made while building this RAG-based Q&A system. AI assistance was used for boilerplate code generation, while architecture, debugging, and optimization were handled manually.

## Phase 1: Initial Architecture & Planning

### Prompt 1: Feature Requirements
```
Build a web app where I can: add a few documents (text files are enough), see the list of uploaded documents, ask a question, get an answer, see 'where the answer came from' (show which document and the part of it that helped), see a simple history of the last 5 run
```

**Manual Work:**
- Researched Flask vs FastAPI for backend
- Decided on RAG architecture with vector embeddings
- Chose TF-IDF for initial implementation (later upgraded)
- Planned document chunking strategy
- Designed REST API structure

---

## Phase 2: UI/UX Design

### Prompt 2: Theme Implementation
```
make the background charcoal black and the dialogue boxes something which goes with the background
```

**Manual Work:**
- Created color palette (#1a1a1a, #2a2a2a, #4a9eff)
- Designed responsive grid layout
- Implemented CSS animations and transitions
- Tested contrast ratios for accessibility
- Added gradient effects on buttons
- Optimized for mobile responsiveness

---

## Phase 3: RAG System Integration

### Prompt 3: LLM Integration
```
how can i make it answer using the rag
```

**Manual Work:**
- Researched LLM providers (Groq, OpenAI, Claude)
- Evaluated Groq for cost and speed
- Implemented document chunking (500 chars, 50-word overlap)
- Built vector similarity search
- Created prompt engineering for context injection
- Tested and tuned relevance thresholds
- Implemented source attribution logic

---

## Phase 4: Real-Time Streaming

### Prompt 4: Streaming Feature
```
can i get output text continuously like i get here, each letter generating one by one
```

**Manual Work:**
- Implemented Server-Sent Events (SSE)
- Built streaming response handler with buffers
- Debugged CORS issues with streaming endpoints
- Added error handling for interrupted streams
- Tested on slow connections
- Optimized buffer size for smooth rendering

---

## Phase 5: Production Debugging

### Issue 1: Model Compatibility
**Error encountered:** llama-3.1-70b-versatile deprecated

**Manual Resolution:**
- Checked Groq API documentation
- Updated to llama-3.3-70b-versatile
- Tested performance comparison
- Verified prompt compatibility
- Updated error handling

---

## Phase 6: Deployment Strategy

### Decision: Platform Selection
**Initial attempt:** Vercel
**Issue:** Stateless architecture incompatible with document storage

**Manual Analysis:**
- Evaluated Railway, Render, Heroku, DigitalOcean
- Compared pricing and storage options
- Tested Railway (failed: 4GB+ image size)
- Researched ephemeral filesystem limitations
- Chose Render for free tier with disk

---

## Phase 7: Dependency Optimization

### Issue 1: Environment Variables
**Error:** `GROQ_API_KEY not found`

**Manual Debugging:**
- Checked .env file encoding (UTF-8 BOM issue)
- Verified python-dotenv version
- Tested environment variable loading
- Created proper .gitignore rules
- Implemented fallback error messages

### Issue 2: Package Compatibility
**Error:** `TypeError: Client.__init__() got unexpected keyword argument 'proxies'`

**Manual Resolution:**
- Debugged httpx version conflict (0.28.1 incompatible)
- Checked groq package dependencies
- Tested multiple httpx versions
- Pinned httpx<0.28 in requirements.txt
- Verified on both local and production

---

## Phase 8: Image Size Optimization

### Challenge: Deployment Image >8GB

**Manual Optimization:**
- Removed ChromaDB (saved ~2GB)
- Replaced sentence-transformers with TF-IDF (saved ~6GB)
- Implemented custom cosine similarity
- Built lightweight JSON-based vector store
- Tested search quality vs original
- Final image: ~500MB

**Technical Decisions:**
- Accepted slight accuracy tradeoff for deployability
- Maintained semantic search capability
- Optimized for startup time

---

## Phase 9: Production File Handling

### Issue: Upload Failures on Render

**Root Cause Analysis:**
- Render free tier uses ephemeral filesystem
- Files deleted on dyno restart
- API endpoint expecting file persistence

**Manual Fixes:**
- Modified upload to read file content directly
- Implemented in-memory vector storage
- Added fallback to vector DB for document listing
- Created graceful degradation for missing files
- Added detailed error logging
- Tested with various file sizes

---

## Phase 10: Error Handling & Validation

**Implemented Manually:**
- File type validation (.txt only)
- Empty file detection
- Empty question validation
- API key presence checks
- Network timeout handling
- Detailed console logging for debugging
- User-friendly error messages

**Testing Coverage:**
- Invalid file formats
- Network interruptions
- Missing API keys
- Empty documents
- Concurrent uploads
- Browser compatibility (Chrome, Firefox, Edge)

---

## Phase 11: Monitoring & Observability

### Feature: Status Page

**Manual Implementation:**
- Designed health check endpoint
- Implemented backend status verification
- Added vector DB connection testing
- Created LLM connectivity check
- Built real-time UI status indicator
- Added auto-refresh every 30 seconds
- Styled status dots with CSS animations

---

## Development Statistics

### Time Investment
- **Research & Planning:** ~8 hours
- **Core Development:** ~15 hours
- **Debugging & Testing:** ~12 hours
- **Deployment Optimization:** ~6 hours
- **Documentation:** ~3 hours
- **Total:** ~44 hours

### Code Written
- **Total Lines:** ~1,500
- **Manual Modifications:** ~600 lines (40%)
- **AI-Generated Base:** ~900 lines (60%)
- **Bug Fixes:** ~200 lines

### Iterations
- **Deployment Attempts:** 5 (Vercel → Railway → Render)
- **Dependency Fixes:** 8
- **Architecture Changes:** 3 major revisions
- **Performance Optimizations:** 7

## Technical Decisions Made

### 1. LLM Provider: Groq
**Rationale:**
- 300+ tokens/sec inference speed
- Generous free tier
- Streaming support
- No credit card required
- Good model quality (Llama 3.3 70B)

### 2. Vector Search: Custom TF-IDF
**Rationale:**
- Lightweight (<1MB vs 6GB)
- Fast deployment
- Sufficient accuracy for small datasets
- Easy to debug and modify

### 3. Storage: JSON-based
**Rationale:**
- Works with ephemeral filesystems
- Easy to inspect/debug
- No external dependencies
- Fast for <1000 documents

### 4. Frontend: Vanilla JS
**Rationale:**
- No build step required
- Faster initial load
- Easier to deploy
- Sufficient for simple UI

## Lessons Learned

### What Worked Well
✅ Starting with simple architecture
✅ Testing locally before deployment
✅ Incremental feature additions
✅ Comprehensive error logging
✅ Reading documentation first

### What Could Be Improved
⚠️ Should have researched hosting limitations earlier
⚠️ Could have added unit tests
⚠️ Performance monitoring would help
⚠️ Could implement caching layer

## Future Enhancements

**Planned:**
- PDF/DOCX support
- User authentication
- Advanced chunking strategies
- Caching layer
- Rate limiting
- Analytics dashboard

**Technical Debt:**
- Add unit tests
- Implement proper logging framework
- Add API rate limiting
- Optimize chunk overlap algorithm
- Add document versioning

---

## Summary

This project demonstrates practical application of RAG technology with emphasis on:
- Real-world deployment challenges
- Performance vs functionality tradeoffs
- Debugging complex dependency issues
- Optimizing for constrained environments
- Building production-ready error handling

**AI Assistance:** Used for generating boilerplate, standard patterns, and documentation structure.

**Manual Work:** Architecture design, debugging, optimization, deployment strategy, testing, and all technical decisions.
