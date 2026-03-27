# Repo Structure

```
.
├── inference_eval_pipeline.ipynb   # Main notebook: full retrieve → generate → evaluate pipeline
├── output_payload.json             # Output of the latest end-to-end run (with generated responses)
├── requirements.txt                # Python dependency list
├── README.md                       # Project overview, usage guide, and dataset documentation
│
├── retrieval/                      # Retrieval and reranking modules
│   ├── bm25_retriever.py           # BM25 sparse retriever (rank-bm25); doc_ids aligned with FAISS
│   ├── dense_retriever.py          # Dense retriever: loads FAISS index, lazy-loads query encoder
│   ├── fusion.py                   # Reciprocal Rank Fusion (RRF) for merging ranked lists
│   ├── reranker.py                 # Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
│   ├── pipeline.py                 # End-to-end RetrievalPipeline class (BM25 + Dense → RRF → Rerank)
│   └── ablation.ipynb              # Ablation notebook: BM25-only vs Dense-only vs Full pipeline
│
├── generation/                     # Response generation module
│   └── generator.py                # RAGGenerator class: wraps Qwen2.5-0.5B-Instruct for RAG generation
│
├── evaluation/                     # Evaluation modules
│   ├── metrics.py                  # Retrieval metrics (Recall@K, MRR, NDCG@K) and benchmark loader
│   ├── gen_metrics.py              # Generation metrics (Token F1, BERTScore)
│   └── __init__.py
│
├── embedding/                      # Chunk embedding module
│   └── embed_chunks.py             # Encodes chunks with natural_language_prefix strategy; writes FAISS index
│
├── chunking/                       # Text chunking module
│   ├── chunker.py                  # Step 1: structure-aware fixed-size chunking → corpus_fixed.jsonl
│   ├── semantic_chunker_tfidf.py   # Step 2a: merge adjacent chunks by TF-IDF similarity
│   └── semantic_chunker_embedding.py  # Step 2b: merge adjacent chunks by embedding similarity
│
├── corpus/                         # Raw corpus files
│   ├── corpus.jsonl                # Merged corpus (Wikipedia + Wikibooks + 80Cuisine, 1003 records)
│   ├── wikipedia.jsonl             # Wikipedia East-Asian cuisine articles (605 records)
│   ├── wikibook.jsonl              # Wikibooks recipe entries (370 records)
│   ├── 80Cuisine.jsonl             # 80 Cuisines blog posts (28 records)
│   ├── merge_corpus.py             # Script to merge source JSONL files into corpus.jsonl
│   └── east-asian-corpus-builder/  # Corpus crawling and construction toolkit
│       ├── build_east_asian_corpus_v9_singlefile.py  # Main crawling script
│       ├── seed_wikipedia_big.json     # Seed URL list for Wikipedia crawling
│       ├── seed_wikibooks_big.json     # Seed URL list for Wikibooks crawling
│       ├── requirements.txt            # Dependencies for the crawler
│       └── README.md                   # Crawling pipeline instructions
│
├── benchmark/                      # Evaluation benchmark
│   └── benchmark.json              # 50 factoid QA items with gold chunk_ids for retrieval and generation eval
│
└── artifacts/                      # Pre-computed artifacts (fully reproducible)
    ├── chunks/                     # Chunked corpus files
    │   ├── corpus_fixed.jsonl      # Fixed-size chunking output (Step 1)
    │   ├── chunks_tfidf.jsonl      # TF-IDF semantic merging output (Step 2a)
    │   └── chunks_embedding.jsonl  # Embedding semantic merging output (Step 2b); used for indexing
    └── faiss/                      # FAISS vector index and metadata
        ├── *.index                 # Binary FAISS index (1404 vectors, dim=1024)
        ├── *.meta.jsonl            # Per-vector chunk metadata (text, title, sections, etc.)
        └── *.manifest.json         # Index build config (model, strategy, paths, parameters)
```
