# AI Search Pipeline

## Overview
The **AI Search Pipeline** is an enterprise-grade solution designed to empower businesses with advanced semantic search and AI-driven response generation from a FAISS vector database. Leveraging ensemble retrieval techniques (FAISS, MMR, BM25, cross-encoder reranking) and LLM integration (Ollama with Llama3.2:1b), it delivers high-quality, context-aware answers for enterprise knowledge management and AI applications. The pipeline features a Flask-based API, a modern, responsive front-end, and a robust evaluation framework (BLEU, ROUGE, METEOR) for reliability and performance monitoring. Deployable locally or on IIS, it’s ideal for enterprise search, analytics, and Retrieval-Augmented Generation (RAG) systems.

## Key Features
- **Ensemble Retrieval**: Combines FAISS vector search, Maximum Marginal Relevance (MMR), BM25, similarity search, and cross-encoder reranking for superior document retrieval.
- **LLM Response Generation**: Uses Ollama (Llama3.2:1b) to generate concise, context-aware answers from retrieved documents.
- **Prompt Engineering**: Supports zero-shot, few-shot, chain-of-thought, and self-consistency prompting for flexible LLM interactions.
- **Evaluation Framework**: Validates LLM outputs and computes quantitative metrics (BLEU, ROUGE, METEOR) via a LangGraph workflow.
- **Responsive Front-End**: Modern JavaScript/HTML/CSS UI for intuitive query submission and result display with source attribution.
- **Flask API**: RESTful `/response` endpoint for programmatic query processing.
- **IIS Deployment**: Configuration support for Windows server environments via `web.config`.
- **Robust Logging**: Centralized logging for monitoring, debugging, and performance tracking.
- **GPU Support**: Leverages CUDA for embedding generation when available, optimizing for large-scale datasets.

## Use Cases
- **Enterprise Search**: Enable semantic search over indexed documents and web content for internal knowledge bases.
- **AI-Driven Insights**: Generate accurate, context-aware responses for analytics and decision-making.
- **Knowledge Management**: Retrieve and present relevant information from large-scale data repositories.
- **RAG Applications**: Power Retrieval-Augmented Generation systems with high-quality document retrieval.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/ai-search-pipeline.git
   cd ai-search-pipeline
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Requires Python 3.8+ and dependencies like Flask, LangChain, Sentence Transformers, FAISS, Ollama, NLTK, and others (see `requirements.txt`).
3. **Configure Settings**:
   - Update `config/config.toml` with paths for FAISS storage, LLM, and embedding models:
     ```toml
     [faiss]
     path = "./database/faiss"
     [LLMmodel]
     model = "llama3.2:1b"
     [embeddings]
     model_path = "./models/sentence_tranformer_model"
     encoder_model = "./models/ms-marco-MiniLM-L-6-v2"
     [logging]
     log_dir = "logs"
     log_level = "INFO"
     log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
     ```
   - Ensure a pre-built FAISS index (from a separate ingestion pipeline) is accessible at `faiss.path`.
   - Place embedding models in `models/` (e.g., `ms-marco-MiniLM-L-6-v2`, `sentence_tranformer_model`).
4. **Run the Flask Application**:
   ```bash
   python main.py
   ```
   Access the UI at `http://localhost:5008` or use the `/response` API endpoint.

## Directory Structure
- `main.py`: Flask API and application entry point for query processing and front-end serving.
- `src/`:
  - `agent_llm.py`: Handles Ollama LLM calls for answer evaluation.
  - `embeddings.py`: Generates SentenceTransformer embeddings with GPU support.
  - `ensemble_retrieval.py`: Implements ensemble retrieval with FAISS, MMR, BM25, and cross-encoder reranking.
  - `langraph.py`: LangGraph workflow for evaluating LLM outputs with validation and metrics.
  - `llm_response.py`: Orchestrates document retrieval and LLM response generation.
  - `prompt_engg.py`: Generates prompts for zero-shot, few-shot, chain-of-thought, and self-consistency.
  - `utils.py`: Utility functions (assumed, e.g., `get_logger`).
- `templates/index.html`: Responsive front-end UI for query submission and result display.
- `config/config.toml`: Configuration for FAISS, LLM, embeddings, and logging.
- `models/`:
  - `ms-marco-MiniLM-L-6-v2/`: Cross-encoder model for reranking.
  - `sentence_tranformer_model/`: SentenceTransformer model for embeddings.
- `logs/`: Directory for log files.
- `logger_setup.py`: Centralized logging configuration (assumed).

## Deployment
### Local Deployment
- Run `python main.py` to start the Flask server on port 5008.
- Access the UI at `http://localhost:5008` or send POST requests to `/response` with JSON:
  ```json
  "<query_text>"
  ```

## API Endpoints
- **GET /**: Serves the front-end UI (`templates/index.html`).
- **POST /response**: Processes a query and returns an answer with sources.
  - **Request**: JSON string (e.g., `"What is Pydantic?"`).
  - **Response**:
    ```json
    {
      "answer": "Pydantic is a Python library for data validation...",
      "sources": [
        {"source": "file.pdf", "page_number": 1, "relevance": 0.85},
        {"url": "https://example.com", "relevance": 0.78}
      ]
    }
    ```

## Requirements
- **Python**: 3.8 or higher.
- **Dependencies**: Flask, Flask-CORS, LangChain, LangChain-Community, Sentence Transformers, FAISS, Ollama, NLTK, rouge-score, evaluate, torch, pydantic, langgraph, numpy, wfastcgi (see `requirements.txt`).
- **Hardware**: GPU recommended for embedding generation (CUDA via PyTorch).
- **FAISS Index**: Requires a pre-built FAISS index from a separate ingestion pipeline.

## Contributing
Contributions are welcome! Fork the repository, create a feature branch, and submit a pull request with tests and documentation. See `CONTRIBUTING.md` (to be added) for guidelines.

## License
MIT License

## Contact
For enterprise inquiries or support, open an issue on GitHub or contact [your-email@example.com].