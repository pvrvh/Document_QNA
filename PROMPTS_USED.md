# Development Prompts Used

This document contains the key prompts used during the development of this Private Knowledge Q&A application. Prompts are organized chronologically by development phase.

## Phase 1: Initial Requirements

### Prompt 1: Project Request
```
Build a web app where I can: add a few documents (text files are enough), see the list of uploaded documents, ask a question, get an answer, see 'where the answer came from' (show which document and the part of it that helped), see a simple history of the last 5 run
```

**Context:** Initial project specification defining core requirements

---

## Phase 2: UI Customization

### Prompt 2: Dark Theme Request
```
make the background charcoal black and the dialogue boxes something which goes with the background
```

**Context:** Requested after initial light theme implementation, wanted darker modern UI

---

## Phase 3: RAG Implementation

### Prompt 3: Fixing Retrieval Issues
```
how can i make it answer using the rag
```

**Context:** App was built but not using RAG for answer generation, needed to integrate LLM

---

## Phase 4: Streaming Feature

### Prompt 4: Adding Streaming
```
can i get output text continuously like i get here, each letter generating one by one
```

**Context:** Wanted ChatGPT-style streaming responses instead of waiting for complete answer

---

## Phase 5: Model Update

### Prompt 5: Model Decommissioned Error
```
[Error message about llama-3.1-70b being decommissioned]
```

**Context:** Original model stopped working, needed update to llama-3.3-70b-versatile

---

## Phase 6: Git and Deployment Preparation

### Prompt 6: Git Operations
```
add on vercel now
```

**Context:** Request to deploy application to Vercel

### Prompt 7: Platform Change
```
okay use railway then
```

**Context:** Switched from Vercel to Railway due to stateless/storage limitations

---

## Phase 7: Deployment Issues

### Prompt 8: Railway File Size Issue
```
it said image exceeded the size of 4 gb would it be a problem though
```

**Context:** Railway deployment failed due to ChromaDB + sentence-transformers being too large

### Prompt 9: Render Deployment Decision
```
can i deploy this app on render? the image exceeds 8 gb
```

**Context:** Looking for alternative to Railway, considering Render

### Prompt 10: Deployment Confirmation
```
no i need grop api rag only and deploy on render
```

**Context:** Confirmed requirements: Groq API only, deploy on Render

---

## Phase 8: Fixing Dependency Issues

### Prompt 11: Groq API Key Error
```
[Traceback showing: groq.GroqError: The api_key client option must be set]
```

**Context:** Environment variable not loading correctly, .env file encoding issues

### Prompt 12: httpx Compatibility Error
```
TypeError: Client.__init__() got an unexpected keyword argument 'proxies'
```

**Context:** httpx 0.28.1 incompatible with groq 0.4.1, needed version downgrade

---

## Phase 9: Upload Issues on Render

### Prompt 13: Upload Failure
```
still showing upload failed, failed to detch
```

**Context:** File uploads not working on deployed Render instance

### Prompt 14: Persistent Upload Issues
```
when i upload the docs, it says failed to upload the documents
```

**Context:** Still experiencing upload problems after initial fix

### Prompt 15: Render-Specific Upload Problem
```
it isnt uploading text files on render deployment site
```

**Context:** Upload worked locally but failed on Render's ephemeral filesystem

---

## Phase 10: Documentation Request

### Prompt 16: Final Documentation Requirements
```
What to include

* A status page, that shows health of backend, database, and llm connection.
* Basic handling for empty/wrong input
* A short README: how to run, what is done, what is not done
* A short AI_NOTES.md: what you used AI for, and what you checked yourself. Which LLM and provider does your app use and why.
* A PROMPTS_USED.md, with records of your prompts used for app developemnt. Don't include agent responses, api keys, etc.
```

**Context:** Request for comprehensive documentation and status monitoring

---

## Summary of Prompt Patterns

### Effective Prompts
1. **Clear Requirements:** "Build a web app where I can..."
2. **Specific Changes:** "make the background charcoal black"
3. **Feature Requests:** "can i get output text continuously"
4. **Error Sharing:** Pasting full error messages for debugging
5. **Platform Decisions:** "deploy on render"

### Iterations Required
- Dark theme: 1 prompt
- RAG integration: Multiple iterations
- Streaming: 1 prompt, worked immediately
- Model update: Automatic fix after error
- Deployment: 3+ platform changes (Vercel → Railway → Render)
- Dependencies: Multiple fixes (dotenv, httpx versions)

### Development Flow
1. Initial build (single prompt)
2. UI refinements (1-2 prompts)
3. Feature additions (per feature: 1-3 prompts)
4. Debugging (show error → get fix)
5. Deployment issues (trial and error)
6. Documentation (single comprehensive prompt)

## Lessons for Future Prompting

### ✅ Do's
- Provide full error messages with stack traces
- Be specific about visual preferences (colors, layout)
- Clearly state technical constraints (file size limits, storage)
- Mention target platform early (affects architecture)
- Request incremental changes rather than full rewrites

### ❌ Don'ts
- Assume AI knows deployment platform limitations
- Skip mentioning errors encountered
- Request multiple unrelated changes in one prompt
- Expect first deployment attempt to work
- Assume generated code is production-ready

## Prompt-to-Code Ratio

- **Total Major Prompts:** ~16
- **Code Files Generated:** 10+ files
- **Lines of Code:** ~1200+
- **Average:** ~75 lines of functional code per prompt
- **Debugging Iterations:** ~40% of prompts were fixes/refinements

## Most Impactful Prompts

1. **Initial requirement** (Prompt 1) - Created entire foundation
2. **RAG integration** (Prompt 3) - Added core intelligence
3. **Streaming feature** (Prompt 4) - Major UX improvement
4. **Platform switch to Render** (Prompts 9-10) - Critical deployment decision
5. **Documentation request** (Prompt 16) - Comprehensive project completion

## Time Saved by AI

**Estimated Development Time:**
- Without AI: ~40-60 hours (research + coding + debugging)
- With AI: ~6-8 hours (mostly prompting + testing + fixing)
- **Time savings:** ~80-85%

**Note:** Time saved includes learning curve for RAG implementation, Flask setup, frontend design, and deployment configuration.
