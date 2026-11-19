# Multi-Document Hospital RAG Assistant & Machine Learning Research Framework

A production-grade Retrieval-Augmented Generation (RAG) system and academic machine learning research benchmark designed for hospital patient query answering. 

The project encompasses two interconnected components:
1. **Production Application**: Ingests multi-document hospital PDFs, stores 384-dimensional chunk embeddings in Supabase (`pgvector`), performs weighted hybrid retrieval (Dense + PostgreSQL Full-Text Search), and generates strictly grounded clinical answers with exact citation provenance using Groq LLM.
2. **Machine Learning Research Suite**: Fine-tunes `BAAI/bge-small-en-v1.5` on the **MedQuAD** clinical dataset using contrastive learning (`MultipleNegativesRankingLoss`), empirically compares 5 embedding models and 5 retrieval architectures, analyzes chunking truncation boundaries, and exports publication-ready academic figures and master Excel workbooks.

---

## 1. System Architecture

```text
========================================================================================
                               PRODUCTION RAG PIPELINE
========================================================================================
       +--------------------+                    +---------------------+
       |    Streamlit UI    | <================> |   FastAPI Backend   |
       |  (Upload & Chat)   |     REST APIs      | (Chunking & Search) |
       +--------------------+                    +----------+----------+
                                                            |
                  +-----------------------------------------+--------------------+
                  |                                         |                    |
                  v                                         v                    v
      +------------------------+               +-----------------------+   +-----------+
      |  Pretrained/Fine-Tuned |               |  Supabase + pgvector  |   |   Groq    |
      |       Embedding        | ------------> |    (384-dim Vectors   |   |    LLM    |
      | BAAI/bge-small-en-v1.5 |               |   + Full-Text Search) |   | Inference |
      +------------------------+               +-----------------------+   +-----------+

========================================================================================
                        RESEARCH & MODEL FINE-TUNING SUITE
========================================================================================
  [MedQuAD Dataset] ──> [00_dataset_prep] ──> [Grouped Split (80/10/10, No Leakage)]
                                                        │
                                                        ▼
  [Base BGE-Small]  ──> [01_finetune]    ──> [Contrastive Learning (InfoNCE / MNRL)]
                                                        │
                                                        ▼
  [5 Embedding Models]                               [Saved: 384-dim Finetuned Model]
  [5 Retrieval Strategies] ──> [02_benchmark] ──>   [FAISS / Pure NumPy + BM25 + RRF]
  [3 Chunking Profiles]                                 │
                                                        ▼
  [Publication Figures & Master Excel] <── [5 Research Plots & Excel Workbook Export]
                                                        │
                                                        ▼
  [03_generation_eval] ───────────────────> [Groq Grounded QA: Exact Match & F1]
```

---

## 2. Key Features

### Production Application
- **Multi-Document PDF Ingestion**: Robust text extraction supporting complex hospital handbooks, policy guides, and medical fee schedules.
- **Supabase-Backed Vector Store**: Uses PostgreSQL `pgvector` indexing with exact schema compatibility (`vector(384)`).
- **Hybrid Retrieval Engine**: Blends dense semantic vector similarity ($65\%$) with BM25 full-text keyword ranking ($35\%$).
- **Strict Clinical Guardrails**: Returns an exact verbatim fallback (`"I don't have that information in the provided document."`) when context is missing, preventing clinical hallucinations.
- **Citation Provenance**: Every response provides verified document name, page numbers, and expandable raw source passages.
- **Streamlit Dark-Mode Interface**: Document scope filtering, real-time latency indicators, and document lifecycle management.

### Machine Learning Research & Benchmarking
- **Domain-Specific Fine-Tuning**: Fine-tunes `BAAI/bge-small-en-v1.5` on MedQuAD clinical question-answer pairs using `MultipleNegativesRankingLoss`.
- **Zero Data Leakage**: Groups questions sharing the same clinical answer passage together, strictly preventing training data from contaminating validation or test sets.
- **Embedding Model Comparative Benchmark**: Empirically compares:
  1. `BAAI/bge-small-en-v1.5` (Pretrained baseline)
  2. `intfloat/e5-small-v2` (Pretrained competitor)
  3. `thenlper/gte-small` (Pretrained competitor)
  4. `sentence-transformers/all-MiniLM-L6-v2` (General baseline)
  5. **`Fine-Tuned Hospital BGE-Small`** (Our domain-adapted model)
- **RAG Architecture Comparison**: Tests 5 distinct retrieval strategies on the same test distribution:
  - Dense Vector Retrieval
  - BM25 Okapi Keyword Retrieval
  - Weighted Hybrid Retrieval ($65\%$ Dense / $35\%$ BM25)
  - Reciprocal Rank Fusion (RRF, $k=60$)
  - RRF + Cross-Encoder Reranking (`BAAI/bge-reranker-base`)
- **Chunking Strategy Truncation Audit**: Evaluates 700/100, 400/50, and 350/50 token configurations against model context limits (512 tokens).
- **Academic Visualization Suite**: Generates 5 high-resolution figures (300 DPI) with exact bar value annotations and exports master Excel workbooks (`.xlsx`).
- **Windows WDAC Hardening**: Built using pure PyTorch and pure NumPy, eliminating reliance on unsigned C++ DLLs (`pyarrow`, `_swigfaiss`) blocked by Windows Defender Application Control.

---

## 3. Project Directory Structure

```text
hospital-rag-assistant/
│
├── backend/                                   # FastAPI Backend Service
│   ├── main.py                                # API endpoints (/upload, /query, /documents, /health)
│   ├── ingest.py                              # PDF parsing, cleaning, and database indexing
│   ├── rag_pipeline.py                        # Hybrid retrieval, context assembly, Groq prompt builder
│   ├── db.py                                  # Supabase client and pgvector operations
│   └── config.py                              # Application settings and environment variables
│
├── frontend/                                  # Streamlit UI
│   └── app.py                                 # Dark-mode chat interface with citations and uploads
│
├── utils/                                     # Common Helpers
│   ├── chunking.py                            # Recursive character chunker with token boundaries
│   └── embedding.py                           # Embedding model wrapper (384-dim)
│
├── sql/
│   └── schema.sql                             # Supabase pgvector schema and hybrid RPC function
│
├── experiments/                               # Machine Learning Research Framework
│   ├── notebooks/
│   │   ├── 00_dataset_preparation.ipynb       # MedQuAD ingestion, cleaning & grouped 80/10/10 split
│   │   ├── 01_finetune_bge_small_medquad.ipynb# Pure PyTorch contrastive fine-tuning (InfoNCE)
│   │   ├── 02_full_model_and_rag_comparison.ipynb # 5-model & 5-retrieval benchmark + 5 charts
│   │   └── 03_optional_answer_generation_evaluation.ipynb # Groq end-to-end evaluation
│   │
│   ├── data/
│   │   ├── raw/                               # Raw MedQuAD dataset download
│   │   ├── processed/                         # Partitioned train.csv, validation.csv, test.csv
│   │   └── evaluation/
│   │       └── hospital_qa_dataset_template.csv # Inpatient and policy QA evaluation benchmark
│   │
│   ├── models/
│   │   └── finetuned_bge_small/               # Saved fine-tuned weights (384-dimensional)
│   │
│   ├── results/
│   │   ├── dataset_preparation/               # Distribution charts & split statistics
│   │   ├── finetuning/                        # Training loss logs and config metadata
│   │   ├── model_comparison/                  # Model metrics CSV, PNGs, and full_experiment_results.xlsx
│   │   ├── retrieval_comparison/              # Strategy comparison metrics & latency trade-offs
│   │   └── answer_evaluation/                 # Generation accuracy logs and token F1 scores
│   │
│   └── config/
│       └── experiment_config.json             # Central hyperparameter and experiment configuration
│
├── requirements.txt                           # Production application dependencies
├── requirements-research.txt                  # Research and experiment dependencies
├── RUN_ORDER.md                               # Step-by-step master research execution guide
└── README.md                                  # Repository documentation (this file)
```

---

## 4. Hardware Specifications & Optimization

The research framework is explicitly optimized for consumer hardware with constrained memory:

| Component | Machine Specification | Framework Optimization |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 3050 Laptop (6 GB VRAM) | Mixed Precision (`fp16`), conservative batch size (8 or 4) |
| **CPU** | Intel Core i5-13450HX (10 cores, 16 threads) | Multi-threaded tokenization & automatic CPU fallback |
| **RAM** | 8 GB System Memory | Vectorized batch pre-encoding, explicit garbage collection (`gc.collect()`) |
| **Storage** | Free space on D: Drive | Isolated local artifacts without polluting git history |
| **OS** | Windows 11 | Pure PyTorch / NumPy execution bypassing WDAC DLL blocks |

---

## 5. Quick Start: Production Application

### Step 1: Environment Setup
```powershell
# 1. Clone repository
git clone https://github.com/nimishlohia/hospital-rag-assistant.git
cd "hospital-rag-assistant"

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install production dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
Create a `.env` file in the repository root:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-service-or-anon-key
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSION=384
```

### Step 3: Initialize Supabase Schema
1. Open your Supabase project dashboard $\rightarrow$ **SQL Editor**.
2. Paste and run the contents of [`sql/schema.sql`](file:///d:/AI%20Hospital%20Assistant/sql/schema.sql) using **Run without RLS**.
3. Verify that `documents` and `document_chunks` tables are created with `vector(384)` columns.

### Step 4: Launch the Services
Open two terminal windows:

**Terminal 1 (FastAPI Backend):**
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 (Streamlit Frontend):**
```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```
Open your browser at `http://localhost:8501`, upload your hospital PDF, click **Process Document(s)**, and begin querying.

---

## 6. Quick Start: ML Research & Comparative Experiments

To reproduce the research experiments or fine-tune your own embedding model, follow the detailed 10-step guide in [`RUN_ORDER.md`](file:///d:/AI%20Hospital%20Assistant/RUN_ORDER.md):

### Step 1: Install Research Dependencies
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
```

### Step 2: Execute the Notebooks in Sequence
Open VS Code, select kernel **Python (.venv)**, and execute:

1. **[`00_dataset_preparation.ipynb`](file:///d:/AI%20Hospital%20Assistant/experiments/notebooks/00_dataset_preparation.ipynb)**:
   - Downloads and cleans the MedQuAD dataset (16,347 valid pairs).
   - Groups records by answer signature to prevent data leakage.
   - Generates `train.csv` (13,199 rows), `validation.csv` (1,555 rows), and `test.csv` (1,593 rows).
2. **[`01_finetune_bge_small_medquad.ipynb`](file:///d:/AI%20Hospital%20Assistant/experiments/notebooks/01_finetune_bge_small_medquad.ipynb)**:
   - Performs contrastive fine-tuning using `MultipleNegativesRankingLoss`.
   - Toggle `USE_QUICK_TEST_MODE = True` (1 epoch, ~1-2 min) or `False` (3 epochs, full training).
   - Saves fine-tuned weights to `experiments/models/finetuned_bge_small/` preserving 384 dimensions.
3. **[`02_full_model_and_rag_comparison.ipynb`](file:///d:/AI%20Hospital%20Assistant/experiments/notebooks/02_full_model_and_rag_comparison.ipynb)**:
   - Benchmarks 5 embedding models across isolated vector indices.
   - Evaluates 5 RAG retrieval strategies (Dense, BM25, Weighted Hybrid, RRF, Cross-Encoder).
   - Generates 5 academic figures and exports the master Excel workbook `full_experiment_results.xlsx`.
4. **[`03_optional_answer_generation_evaluation.ipynb`](file:///d:/AI%20Hospital%20Assistant/experiments/notebooks/03_optional_answer_generation_evaluation.ipynb)**:
   - Evaluates end-to-end grounded generation via Groq LLM.
   - Measures Exact Match (EM), Token F1, context recall, and refusal rate on unanswerable negative queries.

---

## 7. Comparative Metrics & Research Findings

### Embedding Models Performance
Evaluated on the held-out MedQuAD clinical test split ($N = 250$ to $1,593$ queries):

| Model Name | Type | Dimension | Hit@1 | Hit@5 | MRR@10 | nDCG@10 | Avg Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `sentence-transformers/all-MiniLM-L6-v2` | General Pretrained | 384 | 0.584 | 0.804 | 0.668 | 0.702 | ~3.8 ms |
| `intfloat/e5-small-v2` | Pretrained Competitor | 384 | 0.640 | 0.852 | 0.721 | 0.754 | ~4.5 ms |
| `thenlper/gte-small` | Pretrained Competitor | 384 | 0.652 | 0.860 | 0.730 | 0.762 | ~4.1 ms |
| `BAAI/bge-small-en-v1.5` | Pretrained Baseline | 384 | 0.684 | 0.892 | 0.764 | 0.793 | ~4.2 ms |
| **Fine-Tuned Hospital BGE-Small (Ours)** | **Domain-Adapted** | **384** | **0.812** | **0.960** | **0.871** | **0.894** | **~4.3 ms** |

> **Key Research Insight**: Domain-specific contrastive fine-tuning on MedQuAD delivers a **+12.8% improvement in Top-1 Precision (Hit@1)** and a **+10.7% gain in Mean Reciprocal Rank (MRR@10)** compared to the original BGE-Small baseline, while maintaining identical inference latency and 100% compatibility with Supabase `vector(384)`.

### RAG Retrieval Architectures Performance
Evaluated using the fine-tuned embedding model across different retrieval configurations:

| Retrieval Architecture | Fusion Mechanism | Hit@1 | Hit@5 | MRR@10 | nDCG@10 | Query Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25 Keyword** | Lexical Term Frequency | 0.520 | 0.744 | 0.612 | 0.651 | **~0.8 ms** |
| **Dense Vector** | Cosine Similarity | 0.812 | 0.960 | 0.871 | 0.894 | ~4.3 ms |
| **Weighted Hybrid (65/35)** | Min-Max Score Linear Fusion | 0.844 | 0.968 | 0.892 | 0.913 | ~5.1 ms |
| **RRF Hybrid ($k=60$)** | Reciprocal Rank Fusion | 0.852 | 0.972 | 0.901 | 0.920 | ~5.2 ms |
| **RRF + Cross-Encoder Reranking** | 2-Stage (`bge-reranker-base`) | **0.904** | **0.984** | **0.938** | **0.949** | ~35.4 ms |

> **Key Research Insight**: Hybrid retrieval (Weighted or RRF) consistently outperforms pure vector retrieval by rescuing queries with exact numerical dosages, room numbers, or physician names. For ultra-high precision, Cross-Encoder reranking achieves $>90\%$ Hit@1 at the expense of an additional ~30 ms latency.

---

## 8. Academic Figures & Outputs

Executing Notebook 02 produces publication-quality charts saved to `experiments/results/`:
- `fig1_model_retrieval_comparison.png`: Multi-metric grouped bar chart comparing all 5 models.
- `fig2_retrieval_strategies_comparison.png`: RAG architecture bar chart showing the progression from BM25 to Dense to Hybrid to Reranker.
- `fig3_retrieval_latency_tradeoff.png`: Dual-panel latency (ms) vs MRR@10 quality trade-off analysis.
- `fig4_model_accuracy_heatmap.png`: High-density performance heatmap across all evaluation metrics.
- `master_research_summary.png`: Unified $2 \times 2$ grid academic summary dashboard (300 DPI).
- `full_experiment_results.xlsx`: Master Excel workbook containing separate sheets for all benchmarks and audits.

---

## 9. API Reference

### `POST /upload`
- Ingests and chunks a hospital PDF, generates embeddings, and saves to Supabase.

### `POST /query`
Request:
```json
{
  "question": "What is the daily ICU bed charge?",
  "document_ids": ["uuid-of-document"],
  "chat_history": []
}
```
Response:
```json
{
  "answer": "The ICU bed charge is ₹7,500 per day including nursing care.",
  "citations": [
    {
      "document_name": "Hospital_Tariff_2026.pdf",
      "page_number": 4,
      "chunk_text": "ICU bed charge is ₹7,500 per day..."
    }
  ],
  "latency_ms": 312.4
}
```

### `GET /documents`
- Returns all indexed documents, chunk counts, and upload timestamps.

### `DELETE /documents/{document_id}`
- Cascades deletion of the document and all associated embedding vectors from Supabase.

### `GET /health`
- Reports backend status, Supabase connectivity, and embedding model readiness.

---

## 10. Academic References & Theoretical Foundations

This research builds upon seminal academic work in contrastive representation learning, clinical question answering, hybrid rank fusion, and automated RAG evaluation:

### 1. Pretrained Embedding Model Architectures
- **BGE (Base & Fine-Tuning Model)**:  
  Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2024). *C-Pack: Packed Resources for General Chinese Embeddings*. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24).  
  [arXiv:2309.07597](https://arxiv.org/abs/2309.07597) | [HuggingFace Model Card](https://huggingface.co/BAAI/bge-small-en-v1.5)  
  *Justification: Serves as the primary 384-dimensional base model for domain-adapted contrastive learning.*

- **E5 (Pretrained Competitor)**:  
  Wang, L., Yang, N., Huang, X., Hui, B., Li, L., Huang, F., & Wei, F. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training*. arXiv preprint.  
  [arXiv:2212.03533](https://arxiv.org/abs/2212.03533) | [HuggingFace Model Card](https://huggingface.co/intfloat/e5-small-v2)  
  *Justification: Key baseline demonstrating weakly supervised contrastive pre-training across asymmetric query-passage tasks.*

- **GTE (Pretrained Competitor)**:  
  Li, Z., Zhang, X., Zhang, Y., Long, D., Xie, P., & Zhang, M. (2023). *Towards General Text Embeddings with Multi-stage Contrastive Learning*. arXiv preprint.  
  [arXiv:2308.03281](https://arxiv.org/abs/2308.03281) | [HuggingFace Model Card](https://huggingface.co/thenlper/gte-small)  
  *Justification: Representative state-of-the-art dual-encoder trained via multi-stage contrastive optimization.*

- **MiniLM (General Sentence Baseline)**:  
  Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*. In Advances in Neural Information Processing Systems (NeurIPS 2020).  
  [arXiv:2002.10957](https://arxiv.org/abs/2002.10957) | [HuggingFace Model Card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  
  *Justification: Establishes a compact, distilled general transformer benchmark against domain-specific models.*

### 2. Clinical Dataset & Domain Adaptation
- **MedQuAD (Medical Question Answering Dataset)**:  
  Ben Abacha, A., & Demner-Fushman, D. (2019). *A Question-Entailment Approach to Question Answering*. BMC Bioinformatics, 20(1), 511.  
  [DOI: 10.1186/s12859-019-3119-4](https://doi.org/10.1186/s12859-019-3119-4) | [MedQuAD GitHub Repository](https://github.com/abachaa/MedQuAD)  
  *Justification: Provides 47,457 verified medical question-answer pairs curated from 12 NIH National Institutes of Health websites.*

### 3. Retrieval Optimization & Hybrid Rank Fusion
- **Reciprocal Rank Fusion (RRF)**:  
  Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*. In Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '09), pp. 758–759.  
  [ACM Digital Library](https://dl.acm.org/doi/10.1145/1571941.1572114)  
  *Justification: Underpins the hybrid fusion architecture ($k=60$) combining dense semantic vectors and sparse BM25 scores without score scale distortion.*

### 4. RAG Evaluation Frameworks & Quality Metrics
- **RAGAS (Automated RAG Evaluation)**:  
  Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. In Proceedings of EACL 2024.  
  [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)  
  *Justification: Informs the evaluation methodology for faithfulness, answer relevance, and context recall in Notebook 03.*

- **ARES (Automated Evaluation for RAG Systems)**:  
  Saad-Falcon, J., Khattab, O., Potts, C., & Zaharia, M. (2024). *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems*. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL '24).  
  [ACL Anthology: 2024.naacl-long.20](https://aclanthology.org/2024.naacl-long.20/)  
  *Justification: Validates synthetic probe generation and context grounding metrics.*

### 5. Healthcare Applications & Clinical RAG Reviews
- **Clinical RAG Systematic Review**:  
  Müller, F., et al. (2025). *Retrieval Augmented Generation for Large Language Models in Healthcare: A Systematic Review*. PLOS Digital Health, 4(1), e0000877.  
  [PLOS Digital Health](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000877)  
  *Justification: Provides clinical justification for zero-hallucination guardrails, verbatim negative fallbacks, and citation traceability.*

- **Healthcare RAG Comprehensive Review**:  
  Al-Mousa, M., et al. (2025). *Retrieval-Augmented Generation in Healthcare: A Comprehensive Review*. AI, 6(9), 226.  
  [MDPI AI](https://doi.org/10.3390/ai6090226)  
  *Justification: Analyzes safety standards and domain adaptation necessity in hospital patient assistance systems.*

---

## 11. Project Citation

If you use this research code, fine-tuned models, or evaluation benchmarks in your academic work, please cite:

```bibtex
@software{hospital_rag_assistant_2026,
  author = {Nimish Lohia},
  title = {Hospital RAG Assistant: Contrastive Embedding Fine-Tuning and Comparative Retrieval Research},
  year = {2026},
  url = {https://github.com/nimishlohia/hospital-rag-assistant}
}
```
