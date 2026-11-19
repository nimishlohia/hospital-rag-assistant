# Hospital RAG Research: Model Training & Comparative Benchmark Run Guide

This guide provides a comprehensive, step-by-step walkthrough for conducting the machine learning fine-tuning and comparative retrieval experiments for the **Hospital Patient Query Assistant**.

---

## 1. Hardware & System Setup Summary

| Component | Specification | Experiment Tuning |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM) | Mixed Precision (`fp16`), Batch size 8 (or 4 in test mode) |
| **CPU** | Intel Core i5-13450HX (10 cores, 16 threads) | Multi-threaded data preprocessing and tokenization |
| **RAM** | 8 GB System Memory | Aggressive garbage collection (`gc.collect()`), subset limits |
| **OS** | Windows 11 | Safe `python -m pip` module execution |
| **Storage** | 68 GB Free on D: Drive | Local isolated datasets and model checkpoints |

> [!IMPORTANT]
> **Windows Application Control Notice**: If your Windows system blocks direct execution of `pip.exe`, **never** run `pip install`. Always use:
> ```powershell
> .\.venv\Scripts\python.exe -m pip install <package_name>
> ```

---

## 2. Directory Structure Overview

```text
hospital-rag-assistant/
│
├── experiments/
│   │
│   ├── notebooks/
│   │   ├── 00_dataset_preparation.ipynb               # Step 2: Download & clean MedQuAD
│   │   ├── 01_finetune_bge_small_medquad.ipynb        # Step 4: Contrastive fine-tuning
│   │   ├── 02_full_model_and_rag_comparison.ipynb     # Step 6 & 8: Model & RAG benchmarks
│   │   └── 03_optional_answer_generation_evaluation.ipynb # Step 9: End-to-end Groq evaluation
│   │
│   ├── data/
│   │   ├── raw/                                       # Raw datasets & downloads
│   │   ├── processed/                                 # Clean train.csv, validation.csv, test.csv
│   │   └── evaluation/
│   │       └── hospital_qa_dataset_template.csv       # Separate hospital benchmark
│   │
│   ├── models/
│   │   └── finetuned_bge_small/                       # Saved fine-tuned weights (384-dim)
│   │
│   ├── results/
│   │   ├── dataset_preparation/                       # Distribution plots & summary JSON
│   │   ├── finetuning/                                # Loss curves & training metadata
│   │   ├── model_comparison/                          # Model benchmark CSVs & Master Excel
│   │   ├── retrieval_comparison/                      # Chunking & hybrid retrieval metrics
│   │   └── answer_evaluation/                         # Generation metrics & answer comparisons
│   │
│   └── config/
│       └── experiment_config.json                     # Central experiment configuration
│
├── requirements-research.txt                          # Research package dependencies
└── RUN_ORDER.md                                       # Master execution guide (this file)
```

---

## 3. Step-by-Step Execution Guide

### Step 1: Install Research Dependencies
Open your PowerShell terminal in the repository root `d:\AI Hospital Assistant` and install the research dependencies using the safe Python module command:

```powershell
# 1. Activate your virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install research dependencies using python -m pip
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
```

* **What it does**: Installs `sentence-transformers`, `rank-bm25`, `openpyxl`, `seaborn`, `scikit-learn`, and `matplotlib`.
* **Note on Windows Security**: Pure PyTorch and pure NumPy execution are used throughout all notebooks, completely eliminating any reliance on unsigned C++ DLLs (`pyarrow`, `_swigfaiss`) which trigger Windows Defender Application Control (WDAC) block errors.
* **Expected Output**: `Successfully installed ...` with return code 0.
* **If it fails**: Check your internet connection. If a wheel fails to build, verify Microsoft C++ Build Tools or run the command again.

---

### Step 2: Run `00_dataset_preparation.ipynb` (Prepare MedQuAD)
1. Open VS Code and open:
   [`experiments/notebooks/00_dataset_preparation.ipynb`](file:///d:/AI%20Hospital%20Assistant/experiments/notebooks/00_dataset_preparation.ipynb)
2. Click **Select Kernel** in the top-right corner of VS Code $\rightarrow$ Select **Python Environments** $\rightarrow$ Choose your virtual environment (`.venv`).
3. Run the notebook from top to bottom (Cells 1 to 8).

* **Files Generated**:
  - `experiments/data/processed/train.csv` (80% training data)
  - `experiments/data/processed/validation.csv` (10% validation data)
  - `experiments/data/processed/test.csv` (10% test data)
  - `experiments/results/dataset_preparation/medquad_distribution.png`
  - `experiments/results/dataset_preparation/dataset_summary.json`
* **Expected Output**: Grouped split summary confirming **zero group contamination** between train, validation, and test sets.
* **Troubleshooting**: If Hugging Face is blocked or times out, the notebook automatically falls back to local parsing without crashing.

---

### Step 3: Inspect the Prepared Datasets
1. Open [`experiments/data/processed/train.csv`](file:///d:/AI%20Hospital%20Assistant/experiments/data/processed/train.csv) and [`experiments/data/processed/test.csv`](file:///d:/AI%20Hospital%20Assistant/experiments/data/processed/test.csv).
2. Confirm that:
   - Each row contains a clean medical `question`, `answer`, `source`, and `focus`.
   - Questions and answers are properly formatted with no HTML artifacts or broken characters.

---

### Step 4: Run `01_finetune_bge_small_medquad.ipynb` (Fine-Tune BGE-Small)
1. Open:
   [`experiments/notebooks/01_finetune_bge_small_medquad.ipynb`](file:///d:/AI%20Hospital%20Assistant/experiments/notebooks/01_finetune_bge_small_medquad.ipynb)
2. In **Cell 3**, notice the configuration variable:
   ```python
   USE_QUICK_TEST_MODE = True  # Set to True for a 2-minute smoke test
   ```
   - **First run**: Leave it as `True`. Run all cells to verify that training, loss computation, validation, and saving complete without memory errors.
   - **Full training**: Set `USE_QUICK_TEST_MODE = False` and run the notebook to train on the full training subset (~10,000 pairs).
3. Execute the cells sequentially.

* **Files Generated**:
  - `experiments/models/finetuned_bge_small/` (Complete model checkpoint)
  - `experiments/results/finetuning/training_config.json`
* **Expected Output**: Progress bar indicating epoch progression, followed by sanity check output:
  ```text
  Similarity (Positive Med)  : > 0.7000
  Similarity (Negative Irrel): < 0.2500
  >>> VERIFICATION PASSED: The saved fine-tuned model is fully operational! <<<
  ```

---

### Step 5: Verify Saved Fine-Tuned Model Weights
Verify that the output directory contains the saved weights:
- Open [`experiments/models/finetuned_bge_small/`](file:///d:/AI%20Hospital%20Assistant/experiments/models/finetuned_bge_small/).
- You will see `model.safetensors`, `config.json`, `tokenizer.json`, and `modules.json`.
- The output dimension is guaranteed to be **384**, retaining 100% compatibility with your Supabase schema.

---

### Step 6: Run `02_full_model_and_rag_comparison.ipynb` (Model & Retrieval Benchmark)
This is the central research notebook.
1. Open:
   [`experiments/notebooks/02_full_model_and_rag_comparison.ipynb`](file:///d:/AI%20Hospital%20Assistant/experiments/notebooks/02_full_model_and_rag_comparison.ipynb)
2. Run all cells from top to bottom.

* **What it runs**:
  - **Chunking Audit**: Truncation analysis for 700, 400, and 350-token chunk sizes against the 512-token limit.
  - **Model Benchmark**: Evaluates original `bge-small-en-v1.5`, `e5-small-v2`, `gte-small`, `all-MiniLM-L6-v2`, and your newly fine-tuned BGE model in separate, isolated FAISS indices.
  - **Retrieval Strategies**: Compares Dense Vector, BM25, Weighted Hybrid (65/35), RRF Hybrid, and RRF + Cross-Encoder Reranking (`BAAI/bge-reranker-base`).
* **Files Generated**:
  - `experiments/results/model_comparison/model_metrics.csv`
  - `experiments/results/model_comparison/question_wise_results.csv`
  - `experiments/results/model_comparison/model_comparison.png`
  - `experiments/results/model_comparison/full_experiment_results.xlsx` (Master Excel Workbook)
  - `experiments/results/retrieval_comparison/retrieval_metrics.csv`
  - `experiments/results/retrieval_comparison/chunking_analysis.csv`

---

### Step 7: Prepare the Hospital PDF Evaluation Dataset
1. Open [`experiments/data/evaluation/hospital_qa_dataset_template.csv`](file:///d:/AI%20Hospital%20Assistant/experiments/data/evaluation/hospital_qa_dataset_template.csv).
2. It contains verified hospital questions, expected answers, evidence passages, and `answerable` flags (including negative safety queries for fallback testing).
3. You can append any additional questions from your hospital documents directly into this CSV template.

---

### Step 8: Evaluate the Hospital Dataset
- **Cell 9** in Notebook 02 automatically evaluates retrieval on this hospital-specific dataset and outputs:
  - `experiments/results/retrieval_comparison/hospital_benchmark_metrics.csv`.
- These scores are reported **separately** from MedQuAD so academic results remain transparent and uninflated.

---

### Step 9: Run `03_optional_answer_generation_evaluation.ipynb` (Groq RAG Evaluation)
1. Open:
   [`experiments/notebooks/03_optional_answer_generation_evaluation.ipynb`](file:///d:/AI%20Hospital%20Assistant/experiments/notebooks/03_optional_answer_generation_evaluation.ipynb)
2. In **Cell 1**, it automatically reads `GROQ_API_KEY` from your [`.env`](file:///d:/AI%20Hospital%20Assistant/.env) file. If no key is detected, it runs in **Simulation / Mock Mode** so all cells still execute without errors.
3. Run all cells.

* **What it evaluates**:
  - Exact Match (EM) and Token-level F1 score on answerable medical queries.
  - Fallback safety trigger accuracy on unanswerable queries (confirming it says *"I don't have that information in the provided document."*).
  - Average latency in milliseconds.
* **Files Generated**:
  - `experiments/results/answer_evaluation/answer_comparison.csv`
  - `experiments/results/answer_evaluation/generation_metrics.json`

---

### Step 10: Review All Saved Results
Open the master results folder:
- **Master Excel File**: [`experiments/results/model_comparison/full_experiment_results.xlsx`](file:///d:/AI%20Hospital%20Assistant/experiments/results/model_comparison/)
  - Sheet 1: Model Comparison (Hit@1, Hit@3, Hit@5, Recall@5, MRR@10, nDCG@10, Latency)
  - Sheet 2: Retrieval Strategies (Dense vs BM25 vs Weighted vs RRF vs Reranker)
  - Sheet 3: Chunking Audit (Token truncation rates)
  - Sheet 4: Hospital Benchmark Scores
- **Charts**:
  - `experiments/results/model_comparison/model_comparison.png`
  - `experiments/results/dataset_preparation/medquad_distribution.png`

---

## 4. Troubleshooting Guide

### A. Windows Application Control: `Program 'pip.exe' failed to run`
* **Cause**: Windows Defender Application Control (WDAC) blocks unsigned executables in user directories.
* **Solution**: Never call `pip` directly. Always run:
  ```powershell
  .\.venv\Scripts\python.exe -m pip install <package_name>
  ```
  Inside a Jupyter notebook, use:
  ```python
  !{sys.executable} -m pip install <package_name>
  ```

### B. CUDA Out of Memory (OOM) on RTX 3050 (6 GB VRAM)
* **Cause**: Large batch size or multiple models loaded simultaneously.
* **Solutions**:
  1. Reduce `BATCH_SIZE` from `8` to `4` in Cell 3 of Notebook 01.
  2. Ensure mixed precision is active: `use_amp=True`.
  3. Close active browser tabs to free up shared GPU memory.
  4. Run memory cleanup between cells:
     ```python
     import gc, torch
     gc.collect()
     torch.cuda.empty_cache()
     ```

### C. Kernel Not Using the Project Virtual Environment
* **Symptom**: `ModuleNotFoundError: No module named 'sentence_transformers'`.
* **Solution**: In the top-right corner of VS Code, click **Kernel** $\rightarrow$ **Select Another Kernel...** $\rightarrow$ **Python Environments...** $\rightarrow$ Pick `.venv (Python 3.13)`.

### D. Model Dimension Mismatch with Supabase
* **Safety Rule**: Your project Supabase schema uses `vector(384)`.
* **Verification**: `BAAI/bge-small-en-v1.5`, `intfloat/e5-small-v2`, `thenlper/gte-small`, `all-MiniLM-L6-v2`, and your fine-tuned model all produce **384 dimensions**. Do not substitute a 768-dimensional model (like PubMedBERT) unless you alter the database table schema.

---

## 5. Academic Research Integrity Rules

1. **Never fabricate results**: Only report metrics produced by actual execution.
2. **Never claim the model is better without empirical evidence**: Use the measured Hit@k and MRR@10 numbers from the master tables.
3. **No data leakage**: Grouped splitting ensures patient queries never evaluate against passages seen during training.
4. **Isolate evaluation benchmarks**: Keep MedQuAD test scores separate from your Hospital PDF scores.
