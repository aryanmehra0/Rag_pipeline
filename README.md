🧬 Biomedical RAG Pipeline (Gemini 2.5 Pro + Hybrid Retrieval)Overview: A production-grade Retrieval-Augmented Generation (RAG) pipeline optimized for complex biomedical PDFs. It leverages Gemini 2.5 Pro, Parent–Child Chunking, and Hybrid Retrieval (FAISS + BM25 + CrossEncoder) to handle scientific prose, tables, and citations with high precision.🎯 Project GoalThe challenge in biomedical RAG is not just retrieving text, but maintaining the scientific context often lost during standard chunking. This pipeline is designed to:Ingest layout-complex PDFs (multi-column, headers, figures).Preserve Context using a Parent-Child chunking strategy.Retrieve Accurately using a hybrid approach (Semantic + Keyword) re-ranked for relevance.Generate Answers with strict grounding and explicit citations using Gemini 2.5 Pro.This implementation is contained within a single, modular Google Colab Notebook for ease of deployment and demonstration.🏗️ ArchitectureThe pipeline follows a Retriever-Reranker-Generator pattern.Code snippetflowchart LR
    subgraph Ingestion
    A[PDF Upload] --> B[pdfplumber Parsing]
    B --> C[Structure-Aware Cleaning]
    end

    subgraph Indexing
    C --> D[Parent-Child Chunking]
    D --> E[MinHash Deduplication]
    E --> F[(FAISS Vector Store)]
    E --> G[(BM25 Sparse Index)]
    end

    subgraph Retrieval
    H[User Query] --> I{Hybrid Retrieval}
    F --> I
    G --> I
    I --> J[CrossEncoder Reranker]
    J --> K[Fetch Parent Context]
    end

    subgraph Generation
    K --> L[Gemini 2.5 Pro]
    L --> M[Answer w/ Citations]
    end
Key Technical Decisions1. Parent–Child ChunkingThe Problem: Standard chunks (e.g., 512 tokens) often cut off scientific reasoning or experimental conditions.The Solution: We index small Child Chunks (256-512 tokens) for precise vector search match, but we feed the Parent Chunk (full section/context) to the LLM.Result: High retrieval precision without losing the broader context required for reasoning.2. Hybrid Retrieval (Dense + Sparse)Dense (FAISS + Gemini Embeddings): Captures semantic meaning (e.g., "cell death" $\approx$ "apoptosis").Sparse (BM25): Captures exact keyword matches, crucial for specific gene names (e.g., "TAF6δ") or protein IDs that semantic models might gloss over.Fusion: Results are merged to ensure neither specific keywords nor general concepts are missed.3. Cross-Encoder RerankingModel: BAAI/bge-reranker-baseFunction: Takes the top candidates from the hybrid step and re-scores them based on true relevance to the query. This filters out "nearest neighbor" noise before the data hits the LLM context window.🚀 Getting StartedThis project is designed to run entirely in Google Colab.PrerequisitesA Google Account.A Google Gemini API Key (Get it from Google AI Studio).Installation & ExecutionOpen the Notebook: Load the .ipynb file in Google Colab.Cell 1: Dependencies: Installs google-generativeai, faiss-cpu, rank-bm25, pdfplumber, etc.Cell 2: Core Pipeline: Defines the BiomedicalRAGApp class and all utility logic.Cell 3: Upload & Build:Enter your API Key when prompted.Upload your PDF files via the file picker.The pipeline will ingest, chunk, and index the data automatically.Cell 4: Inference:Use the ask("Your Question") function to query your documents.Example UsagePython# Initialize the app (after uploading PDFs)
app = BiomedicalRAGApp.from_pdf_paths(pdf_paths)

# Ask a question
ask("What is the role of TAF6 delta in apoptosis regulation?")
Output:TAF6δ is a pro-apoptotic splice variant... It drives apoptosis by...Citations:[Source: nature_paper_2024, Results Section][Source: supp_data, Figure 3 Caption]📂 Code StructureModuleDescriptionPDFIngestionPipelineHandles pdfplumber logic, regex cleaning, and section extraction.RecursiveStructureAwareChunkingImplements the Parent-Child logic with overlap handling.ChunkDeduplicatorUses MinHash (Jaccard similarity) to remove duplicate paragraphs across papers.FAISSVectorStoreManages text-embedding-004 generation and vector similarity search.HybridRetrieverCombines FAISS scores with BM25Okapi scores.RAGGeneratorConstructs the prompt and manages the Gemini 2.5 Pro interface.🛠️ Future ImprovementsGraphRAG Integration: Implement a Knowledge Graph to better handle multi-hop reasoning between proteins and pathways.Vision Support: Upgrade the ingestion pipeline to use Gemini 1.5 Pro Vision to natively read charts and graphs rather than relying on text extraction.Evaluation: Add RAGAS (RAG Assessment) metrics to quantitatively score retrieval accuracy.📄 LicenseMIT License
