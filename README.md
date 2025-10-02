# PDF RAG Chatbot (English Texts only)

A Retrieval-Augmented Generation (RAG) chatbot that processes English PDF documents and answers user questions in either English or Hindi. The system uses intelligent query translation and language enforcement to provide accurate, grounded responses regardless of the query language.

## 🚀 Features

- **Multilingual Query Support**: Ask questions in English or Hindi
- **Language-Specific Responses**: Choose your preferred response language (English/Hindi)
- **PDF Document Ingestion**: Upload and process multiple PDF files
- **Smart Query Translation**: Hindi queries are automatically translated to English for better retrieval
- **Accurate Retrieval**: Uses dense embeddings + cross-encoder reranking for precise context matching
- **Source Citations**: Every answer includes document references (filename, page number)
- **Grounded Responses**: System abstains when answers aren't found in the documents
- **Persistent Index**: Uploaded documents are indexed once and reused across sessions

## 🏗️ Architecture Overview
User Query (EN/HI) → Language Detection → Query Translation (if needed) →
Vector Retrieval → Reranking → LLM Generation → Language Enforcement → Response


### Key Components:

1. **PDF Processing Pipeline** (`rag_utils.py`)
   - Text extraction using `pypdf`
   - Semantic chunking with overlap for context preservation
   - Metadata tracking (filename, page, chunk index)

2. **Multilingual Query Handling**
   - Language detection using `langdetect`
   - Hindi→English translation using `deep-translator`
   - English embeddings for consistent retrieval

3. **Retrieval System**
   - Dense embeddings: `sentence-transformers/all-MiniLM-L6-v2`
   - Vector similarity search with FAISS
   - Cross-encoder reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`

4. **Generation & Language Control**
   - OpenAI GPT-4o-mini for answer generation
   - Strict prompt engineering for language enforcement
   - Post-generation language verification and correction

## 🛠️ Technology Stack & Rationale

### Core Technologies

| Component | Technology | Why Chosen |
|-----------|------------|------------|
| **UI Framework** | Streamlit | Rapid prototyping, built-in file upload, minimal code |
| **PDF Processing** | pypdf + langchain | Reliable text extraction, metadata preservation |
| **Text Chunking** | RecursiveCharacterTextSplitter | Semantic-aware splitting, configurable overlap |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | Fast, lightweight, good English performance |
| **Vector Store** | FAISS | High-performance similarity search, local deployment |
| **Reranking** | cross-encoder/ms-marco-MiniLM-L-6-v2 | Improved precision over pure embedding similarity |
| **Translation** | deep-translator | Reliable Google Translate API, no async issues |
| **Language Detection** | langdetect | Fast, accurate language identification |
| **LLM** | OpenAI GPT-4o-mini | Cost-effective, good multilingual capabilities |

### Design Decisions

#### 1. Query Translation Approach
**Decision**: Translate Hindi queries to English before retrieval

**Rationale**: 
- English PDFs work best with English embeddings
- Avoids multilingual embedding complexity
- Maintains retrieval accuracy
- Simple implementation

#### 2. English-Only Embeddings
**Decision**: Use `all-MiniLM-L6-v2` instead of multilingual models

**Rationale**:
- Smaller model size (80MB vs 400MB+)
- Faster inference
- Better English performance
- Translation handles cross-language queries

#### 3. Two-Stage Retrieval
**Decision**: Dense retrieval + cross-encoder reranking

**Rationale**:
- Dense retrieval: Fast, semantic understanding
- Reranking: Improved precision, better relevance scoring
- Best of both worlds for accuracy

#### 4. Streamlit UI
**Decision**: Streamlit over FastAPI/React

**Rationale**:
- Rapid development
- Built-in components (file upload, chat interface)
- Easy deployment
- Focus on functionality over UI polish

## 📋 Prerequisites

- **Python**: 3.10 or higher
- **Internet Connection**: Required for model downloads and translation
- **OpenAI API Key**: For answer generation (get from [OpenAI Platform](https://platform.openai.com/))
- **Memory**: ~2GB RAM for model loading
- **Storage**: ~500MB for models and index files

## 🔧 Installation & Setup

### Step 1: Clone and Setup Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows Command Prompt:
.venv\Scripts\activate.bat
# macOS/Linux:
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**What gets installed:**
- `streamlit`: Web UI framework
- `sentence-transformers`: Embedding models
- `faiss-cpu`: Vector similarity search
- `deep-translator`: Hindi↔English translation
- `langdetect`: Language identification
- `openai`: GPT API client
- `langchain-community`: PDF processing utilities
- `pypdf`: PDF text extraction

### Step 3: Configure Environment
```bash

Create `.env` file:


OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
RAG_TOP_K=12
RAG_RERANK_K=6
SIMILARITY_ABSTAIN_THRESHOLD=0.25
```

**Configuration Parameters:**
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: GPT model to use (gpt-4o-mini recommended for cost/performance)
- `RAG_TOP_K`: Number of chunks to retrieve initially (12 = good recall)
- `RAG_RERANK_K`: Number of chunks after reranking (6 = focused context)
- `SIMILARITY_ABSTAIN_THRESHOLD`: Minimum similarity to provide answer (0.25 = conservative)

### Step 4: Test Installation
```bash
python test_models.py
```

**This verifies:**
- Embedding model downloads correctly
- Reranker model loads
- FAISS vector operations work
- Translation functionality works
- All dependencies are properly installed

### Expected output:
Testing model loading...

Loading embedding model...

✓ Embedding model loaded successfully

Loading reranker model...

✓ Reranker model loaded successfully

Testing FAISS...

✓ FAISS working correctly

Testing translation...

✓ Translation working: 'मशीन लर्निंग क्या है?' -> 'What is machine learning?'

🎉 All models loaded successfully! You can now run 'streamlit run app.py'


### Step 5: Launch Application
```bash
streamlit run app.py
```

Open the displayed URL (typically `http://localhost:8501`)

## 🎯 Usage Guide

### Basic Workflow

1. **Upload PDFs**: Click "Upload English PDFs" and select your documents
2. **Wait for Indexing**: System processes and indexes your documents (one-time per file)
3. **Select Response Language**: Choose English or Hindi for responses
4. **Ask Questions**: Type questions in English or Hindi
5. **Review Answers**: Get grounded responses with source citations

### Example Queries

**English Queries:**
- "What is machine learning?"
- "How does neural network training work?"
- "What are the benefits of cloud computing?"

**Hindi Queries:**
- "मशीन लर्निंग क्या है?" (What is machine learning?)
- "न्यूरल नेटवर्क कैसे काम करता है?" (How do neural networks work?)
- "क्लाउड कंप्यूटिंग के फायदे क्या हैं?" (What are the benefits of cloud computing?)

### Understanding the Interface

- **📚 Indexed files**: Shows which documents are currently processed
- **🔍 Detected Hindi query**: Appears when system translates your query
- **Sources**: Expandable section showing document references
- **Clear**: Removes current answer
- **Clear Index**: Removes all indexed documents (forces re-upload)

## 🔍 How It Works

### 1. PDF Ingestion Process
```python
PDF → Text Extraction → Chunking (2000 chars, 250 overlap) → Metadata Tagging
```

**Why this approach:**
- **2000 character chunks**: Balance between context and precision
- **250 character overlap**: Prevents information loss at boundaries
- **Metadata tracking**: Enables accurate source citations

### 2. Query Processing Pipeline
```python
User Query → Language Detection → Translation (if Hindi) → Embedding → Vector Search
```

**Process details:**
- Detects query language using statistical models
- Translates Hindi to English for consistent retrieval
- Converts query to 384-dimensional vector
- Searches against indexed document vectors

### 3. Retrieval & Ranking
```python
Vector Search (top 12) → Cross-Encoder Reranking (top 6) → Context Assembly
```

**Two-stage approach:**
- **Stage 1**: Fast vector similarity (semantic matching)
- **Stage 2**: Precise cross-encoder scoring (relevance ranking)
- **Result**: Most relevant 6 chunks for answer generation

### 4. Answer Generation
```python
Context + Query → GPT Prompt → Raw Answer → Language Enforcement → Final Response
```

**Language control:**
- System prompt specifies target language
- Post-generation language verification
- Translation if language enforcement fails

## 🎛️ Configuration & Tuning

### Performance Tuning

**For better recall (find more relevant content):**
```env
RAG_TOP_K=20
RAG_RERANK_K=8
SIMILARITY_ABSTAIN_THRESHOLD=0.20
```

**For better precision (more focused answers):**
```env
RAG_TOP_K=8
RAG_RERANK_K=4
SIMILARITY_ABSTAIN_THRESHOLD=0.35
```

### Model Alternatives

**For better multilingual support:**
Replace in `rag_utils.py`:
```python
self.embed_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

**For better English performance:**
```python
self.embed_model_name = "sentence-transformers/all-mpnet-base-v2"
```

## 📁 Project Structure

    QA RAG/

    ├── app.py # Main Streamlit application

    ├── rag_utils.py # RAG pipeline implementation

    ├── test_models.py # Installation verification script

    ├── requirements.txt # Python dependencies

    ├── .env   # Environment template

    ├── README.md # This file

    ├── uploads/ # Uploaded PDF storage

    └── data/

      └── index/ # Vector index and metadata storage


## 🔧 Troubleshooting

### Common Issues

**Models fail to download:**
```bash
pip install --upgrade sentence-transformers transformers torch
pip install --no-cache-dir sentence-transformers
```

**Translation errors:**
- Ensure internet connection for Google Translate API
- Check if query is properly detected as Hindi

**No answers found:**
- Lower `SIMILARITY_ABSTAIN_THRESHOLD` in `.env`
- Try rephrasing your question
- Ensure PDFs contain relevant content

**Out of memory:**
- Reduce `RAG_TOP_K` and `RAG_RERANK_K`
- Use smaller embedding model
- Process fewer documents at once

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Docker Deployment
```dockerfile
FROM python:3.10-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```



### Performance Benchmarks
- **Query processing**: ~2-3 seconds
- **Document indexing**: ~1 second per page
- **Memory usage**: ~1.5GB with models loaded
- **Storage**: ~10MB per 100-page document

## 🤝 Contributing

### Areas for Enhancement
- Support for more languages (Spanish, French, etc.)
- Integration with local LLMs (Llama, Mistral)
- Advanced chunking strategies
- Conversation memory
- Batch query processing

