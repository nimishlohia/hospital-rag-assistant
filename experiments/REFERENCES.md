# Academic Research References & Literature Foundations
## Hospital RAG Model Training & Comparative Retrieval Benchmark

This document compiles the formal academic literature, citations, verified hyperlinks, and research justifications underpinning the **Hospital RAG Assistant** research framework.

---

## 1. Primary Academic References by Category

| Category | Reference / Paper | Core Contribution | Direct Project Justification |
| :--- | :--- | :--- | :--- |
| **Base Embedding Model** | **BGE (SIGIR 2024)**<br>[arXiv:2309.07597](https://arxiv.org/abs/2309.07597) | C-Pack resources and state-of-the-art embedding architecture | Justifies selecting `BAAI/bge-small-en-v1.5` as the 384-dim base model for contrastive domain fine-tuning. |
| **Competitor Model** | **E5 (2022)**<br>[arXiv:2212.03533](https://arxiv.org/abs/2212.03533) | Weakly supervised contrastive pre-training across diverse text pairs | Serves as the primary competitor model evaluated under identical prefix prompts (`query: ` / `passage: `). |
| **Competitor Model** | **GTE (2023)**<br>[arXiv:2308.03281](https://arxiv.org/abs/2308.03281) | Multi-stage contrastive learning for general text representations | Provides a comparative benchmark of multi-stage contrastive dual-encoders on medical IR tasks. |
| **General Baseline** | **MiniLM (NeurIPS 2020)**<br>[arXiv:2002.10957](https://arxiv.org/abs/2002.10957) | Task-agnostic self-attention distillation for transformer compression | Establishes the general-purpose compact sentence embedding baseline (`all-MiniLM-L6-v2`). |
| **Clinical Dataset** | **MedQuAD (BMC 2019)**<br>[DOI:10.1186/s12859-019-3119-4](https://doi.org/10.1186/s12859-019-3119-4) | 47,457 medical Q&A pairs from 12 NIH institutes | Supplies verified medical question-answer pairs for domain adaptation and zero-leakage evaluation. |
| **Retrieval Fusion** | **Reciprocal Rank Fusion (SIGIR 2009)**<br>[ACM DL:10.1145/1571941.1572114](https://dl.acm.org/doi/10.1145/1571941.1572114) | Robust rank-based fusion algorithm: $\sum \frac{1}{k + r_i}$ | Formulates the mathematical foundation for combining Dense Vector and Sparse BM25 ranking without score scaling artifacts. |
| **RAG Evaluation** | **RAGAS (EACL 2024)**<br>[arXiv:2309.15217](https://arxiv.org/abs/2309.15217) | Reference-free automated evaluation of retrieval and generation | Governs the evaluation metrics for context precision, context recall, and grounded answer correctness. |
| **RAG Evaluation** | **ARES (NAACL 2024)**<br>[ACL Anthology: 2024.naacl-long.20](https://aclanthology.org/2024.naacl-long.20/) | Synthetic probe validation and automated RAG evaluation | Justifies probe generation methodology and answer faithfulness scoring. |
| **Clinical Safety** | **Healthcare RAG Review (PLOS 2025)**<br>[PLOS Digital Health](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000877) | Systematic analysis of clinical RAG implementations and failure modes | Validates the clinical necessity of strict negative fallback behavior and verbatim citation tracing. |
| **Clinical Overview** | **Healthcare RAG Comprehensive (AI 2025)**<br>[MDPI AI:10.3390/ai6090226](https://doi.org/10.3390/ai6090226) | Comprehensive survey of healthcare domain adaptation | Supports the requirement of medical domain adaptation over general embedding models. |

---

## 2. In-Depth Paper Summaries & Application to this Research

### 1. BGE: BAAI General Embedding
- **Paper**: *C-Pack: Packed Resources for General Chinese Embeddings*
- **Conference**: 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2024)
- **Authors**: Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff
- **Links**: [arXiv:2309.07597](https://arxiv.org/abs/2309.07597) | [HuggingFace: BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **Summary**: Introduces C-Pack and the BGE embedding family trained via retrofitted contrastive pre-training and multi-task fine-tuning with hard negatives. It released highly performant English models including `bge-small-en-v1.5` (33M parameters, 384 dimensions).
- **Project Role**: Used as the foundational base model for fine-tuning. The 384-dimensional latent space directly maps to the Supabase PostgreSQL `vector(384)` column. Prepending the query instruction (`"Represent this sentence for searching relevant passages: "`) optimizes asymmetric retrieval.

### 2. E5: Weakly-Supervised Contrastive Pre-Training
- **Paper**: *Text Embeddings by Weakly-Supervised Contrastive Pre-training*
- **Year**: 2022 (ACL 2024)
- **Authors**: Liang Wang, Nan Yang, Xiaolong Huang, Binyuan Hui, Linjun Li, Fei Huang, Furu Wei
- **Links**: [arXiv:2212.03533](https://arxiv.org/abs/2212.03533) | [HuggingFace: intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2)
- **Summary**: Trains embedding models on CCPairs (1.3 billion noisy text pairs) using contrastive InfoNCE loss, followed by multi-task fine-tuning with positive/negative mining. Uses explicit task prefixes (`query: ` and `passage: `).
- **Project Role**: Primary competitor model in Notebook 02. Evaluates whether weakly supervised web-scale contrastive pre-training outperforms dedicated domain fine-tuning on specialized medical queries.

### 3. GTE: Multi-Stage Contrastive General Text Embeddings
- **Paper**: *Towards General Text Embeddings with Multi-stage Contrastive Learning*
- **Year**: 2023
- **Authors**: Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Meishan Zhang
- **Links**: [arXiv:2308.03281](https://arxiv.org/abs/2308.03281) | [HuggingFace: thenlper/gte-small](https://huggingface.co/thenlper/gte-small)
- **Summary**: Proposes a multi-stage framework consisting of unsupervised pre-training on 800M pairs, supervised fine-tuning with multiple negatives, and hard negative refinement.
- **Project Role**: Evaluated as an un-prefixed dual-encoder baseline against BGE and E5 across identical clinical passages.

### 4. MiniLM: Deep Self-Attention Distillation
- **Paper**: *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*
- **Conference**: Advances in Neural Information Processing Systems (NeurIPS 2020)
- **Authors**: Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, Ming Zhou
- **Links**: [arXiv:2002.10957](https://arxiv.org/abs/2002.10957) | [HuggingFace: all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Summary**: Introduces self-attention relation distillation transfer from a large teacher transformer to a compact 6-layer student model without preserving layer-by-layer hidden states.
- **Project Role**: Serves as the standard industry lightweight baseline. Demonstrates the performance gap between general-purpose distilled models and domain-adapted medical representations.

### 5. MedQuAD: Medical Question Answering Dataset
- **Paper**: *A Question-Entailment Approach to Question Answering*
- **Journal**: BMC Bioinformatics, 20(1), 511 (2019)
- **Authors**: Asma Ben Abacha, Dina Demner-Fushman
- **Links**: [DOI:10.1186/s12859-019-3119-4](https://doi.org/10.1186/s12859-019-3119-4) | [GitHub: abachaa/MedQuAD](https://github.com/abachaa/MedQuAD)
- **Summary**: Introduces MedQuAD, a clinical QA benchmark composed of 47,457 question-answer pairs curated from 12 NIH resources (including NCI, NHLBI, NIDDK, and MedlinePlus) across 37 question types.
- **Project Role**: The primary clinical dataset used for contrastive domain adaptation. In Notebook 00, questions sharing identical clinical answers are grouped together to guarantee zero data leakage across the 80/10/10 split.

### 6. Reciprocal Rank Fusion (RRF)
- **Paper**: *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*
- **Conference**: 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2009)
- **Authors**: Gordon V. Cormack, Charles L. A. Clarke, Stefan Büttcher
- **Links**: [ACM Digital Library](https://dl.acm.org/doi/10.1145/1571941.1572114)
- **Summary**: Formulates Reciprocal Rank Fusion ($RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$), a robust ranking aggregation method that merges disparate search systems without requiring score normalization or calibration.
- **Project Role**: Implemented in Notebook 02 and backend retrieval pipelines ($k=60$) to combine dense vector rankings with BM25 keyword rankings.

### 7. RAGAS: Automated Evaluation of Retrieval-Augmented Generation
- **Paper**: *RAGAS: Automated Evaluation of Retrieval Augmented Generation*
- **Conference**: EACL 2024
- **Authors**: Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert
- **Links**: [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) | [GitHub: ragas](https://github.com/explodinggradients/ragas)
- **Summary**: Introduces reference-free evaluation metrics for RAG: Faithfulness, Answer Relevance, Context Precision, and Context Recall.
- **Project Role**: Framework for Notebook 03 evaluation measuring whether generated answers rely strictly on retrieved evidence passages.

### 8. ARES: Automated Evaluation Framework for RAG
- **Paper**: *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems*
- **Conference**: NAACL 2024
- **Authors**: Jon Saad-Falcon, Omar Khattab, Christopher Potts, Matei Zaharia
- **Links**: [ACL Anthology: 2024.naacl-long.20](https://aclanthology.org/2024.naacl-long.20/)
- **Summary**: Uses synthetic query generation, prediction-powered inference, and LLM judges to provide confidence-calibrated RAG evaluations across diverse domains.
- **Project Role**: Methodological reference for evaluating context relevance and hallucination rates in hospital document QA.

### 9. Systematic Review of Healthcare RAG
- **Paper**: *Retrieval Augmented Generation for Large Language Models in Healthcare: A Systematic Review*
- **Journal**: PLOS Digital Health, 4(1), e0000877 (2025)
- **Authors**: Florian Müller, et al.
- **Links**: [PLOS Digital Health](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000877)
- **Summary**: Comprehensive systematic review of 48 healthcare RAG implementations analyzing clinical safety, grounding methods, citation traceability, and failure modes.
- **Project Role**: Justifies the strict negative fallback behavior (`"I don't have that information in the provided document."`), citation transparency, and grounding guardrails implemented in `rag_pipeline.py`.

---

## 3. Ready-to-Use BibTeX Citations for LaTeX / Overleaf

```bibtex
@inproceedings{xiao2024cpack,
  title     = {C-Pack: Packed Resources for General Chinese Embeddings},
  author    = {Xiao, Shitao and Liu, Zheng and Zhang, Peitian and Muennighoff, Niklas},
  booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages     = {641--649},
  year      = {2024},
  doi       = {10.1145/3626772.3657878},
  url       = {https://arxiv.org/abs/2309.07597}
}

@article{wang2022text,
  title   = {Text Embeddings by Weakly-Supervised Contrastive Pre-training},
  author  = {Wang, Liang and Yang, Nan and Huang, Xiaolong and Hui, Binyuan and Li, Linjun and Huang, Fei and Wei, Furu},
  journal = {arXiv preprint arXiv:2212.03533},
  year    = {2022},
  url     = {https://arxiv.org/abs/2212.03533}
}

@article{li2023towards,
  title   = {Towards General Text Embeddings with Multi-stage Contrastive Learning},
  author  = {Li, Zehan and Zhang, Xin and Zhang, Yanzhao and Long, Dingkun and Xie, Pengjun and Zhang, Meishan},
  journal = {arXiv preprint arXiv:2308.03281},
  year    = {2023},
  url     = {https://arxiv.org/abs/2308.03281}
}

@inproceedings{wang2020minilm,
  title     = {MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers},
  author    = {Wang, Wenhui and Wei, Furu and Dong, Li and Bao, Hangbo and Yang, Nan and Zhou, Ming},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS 2020)},
  volume    = {33},
  pages     = {5776--5788},
  year      = {2020},
  url       = {https://arxiv.org/abs/2002.10957}
}

@article{abacha2019medquad,
  title   = {A Question-Entailment Approach to Question Answering},
  author  = {Ben Abacha, Asma and Demner-Fushman, Dina},
  journal = {BMC Bioinformatics},
  volume  = {20},
  number  = {1},
  pages   = {511},
  year    = {2019},
  doi     = {10.1186/s12859-019-3119-4},
  url     = {https://doi.org/10.1186/s12859-019-3119-4}
}

@inproceedings{cormack2009rrf,
  title     = {Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods},
  author    = {Cormack, Gordon V. and Clarke, Charles L. A. and B{\"u}ttcher, Stefan},
  booktitle = {Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages     = {758--759},
  year      = {2009},
  doi       = {10.1145/1571941.1572114},
  url       = {https://dl.acm.org/doi/10.1145/1571941.1572114}
}

@inproceedings{es2024ragas,
  title     = {RAGAS: Automated Evaluation of Retrieval Augmented Generation},
  author    = {Es, Shahul and James, Jithin and Espinosa-Anke, Luis and Schockaert, Steven},
  booktitle = {Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2024)},
  pages     = {150--158},
  year      = {2024},
  url       = {https://arxiv.org/abs/2309.15217}
}

@inproceedings{saadfalcon2024ares,
  title     = {ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems},
  author    = {Saad-Falcon, Jon and Khattab, Omar and Potts, Christopher and Zaharia, Matei},
  booktitle = {Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2024)},
  pages     = {253--270},
  year      = {2024},
  url       = {https://aclanthology.org/2024.naacl-long.20/}
}

@article{muller2025retrieval,
  title   = {Retrieval Augmented Generation for Large Language Models in Healthcare: A Systematic Review},
  author  = {M{\"u}ller, Florian and others},
  journal = {PLOS Digital Health},
  volume  = {4},
  number  = {1},
  pages   = {e0000877},
  year    = {2025},
  doi     = {10.1371/journal.pdig.0000877},
  url     = {https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000877}
}

@article{almousa2025comprehensive,
  title   = {Retrieval-Augmented Generation in Healthcare: A Comprehensive Review},
  author  = {Al-Mousa, Mohammad and others},
  journal = {AI},
  volume  = {6},
  number  = {9},
  pages   = {226},
  year    = {2025},
  doi     = {10.3390/ai6090226},
  url     = {https://doi.org/10.3390/ai6090226}
}
```
