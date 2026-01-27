# Technical Approach: Grounded Text Summarization of Research Papers

## Project Feasibility Assessment

This project is **highly feasible** within a 10-week academic timeline. The proposed system—generating reference-grounded summaries with traceable evidence—aligns well with established NLP pipelines for summarization and retrieval. Key feasibility factors:

| Aspect | Assessment |
|--------|------------|
| **Computational Resources** | ✅ CPU-friendly models (BART-base, T5-base) are sufficient |
| **Data Availability** | ✅ S2ORC provides pre-structured data; PDF parsing tools exist for uploads |
| **Technical Complexity** | ✅ Modular pipeline with well-documented components |
| **Evaluation Methods** | ✅ NLI-based hallucination detection is established practice |
| **Timeline** | ✅ Achievable with clear milestone breakdown |

---

## Corpus Comparison: S2ORC vs. User-Uploaded Papers

### Option A: S2ORC (Semantic Scholar Open Research Corpus)

**Overview:** S2ORC contains 81.1M English-language academic papers as structured JSON objects with pre-segmented sections, paragraphs, bibliographic references, and metadata.

| Advantages | Disadvantages |
|------------|---------------|
| Pre-processed and consistently structured | Large storage footprint (~500GB+ for full corpus) |
| Section-level segmentation already complete | Requires data use agreement and API access |
| Rich metadata (field of study, citation links) | Static dataset—no papers after corpus freeze date |
| Ideal for systematic evaluation at scale | Limited to papers included in the corpus |
| Enables reproducible experiments | Initial setup overhead for download/indexing |

**Best For:** Systematic evaluation, benchmarking against other methods, reproducible research experiments.

### Option B: User-Uploaded Papers (Real-Time Processing)

**Overview:** Users upload PDF papers directly, which are then parsed, segmented, summarized, and grounded in real-time.

| Advantages | Disadvantages |
|------------|---------------|
| Maximum flexibility—any paper can be processed | Requires robust PDF parsing pipeline |
| Real-time access to latest publications | Inconsistent PDF formatting across publishers |
| Practical end-user product potential | Tables, figures, equations may parse poorly |
| Minimal storage requirements | Quality depends on extraction accuracy |
| No licensing restrictions on user's own papers | More preprocessing code to maintain |

**Best For:** Interactive demo, user-facing product, handling papers outside S2ORC.

### Recommendation: Hybrid Approach

We recommend a **hybrid architecture** that leverages both approaches:

1. **Development & Evaluation Phase:** Use a subset of S2ORC (e.g., 10K-50K papers from Computer Science) for controlled experiments and benchmarking
2. **Demo & Production Phase:** Implement PDF upload capability using GROBID or PaperMage for user-submitted papers
3. **Fallback:** If PDF parsing fails, prompt users to paste plain text or use abstract-only mode

---

## System Architecture

### Overall Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INPUT                                      │
│         (Select paper from S2ORC  OR  Upload PDF  OR  Paste text)           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING MODULE                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  S2ORC Parser   │    │  PDF Extractor  │    │  Text Cleaner   │         │
│  │  (JSON → text)  │    │  (GROBID/PyMuPDF)│   │  (normalize)    │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                      │                                       │
│                                      ▼                                       │
│                    ┌─────────────────────────────────┐                      │
│                    │    CHUNKING & SEGMENTATION      │                      │
│                    │  (paragraphs, sections, ~512    │                      │
│                    │   token sliding windows)        │                      │
│                    └─────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SUMMARIZATION MODULE                                   │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │  BASELINE: TextRank (Extractive)                                  │     │
│   │  - Graph-based sentence ranking                                   │     │
│   │  - Selects top-k most "central" sentences                        │     │
│   │  - No hallucination risk (copies verbatim)                       │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │  ADVANCED: BART / T5 (Abstractive)                               │     │
│   │  - Encoder-decoder transformer architecture                       │     │
│   │  - Generates novel, fluent summaries                             │     │
│   │  - Higher hallucination risk → requires grounding                │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│   OUTPUT: Generated summary (3-7 sentences)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL MODULE                                      │
│         (For each sentence in the generated summary)                         │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │  BASELINE: BM25 (Sparse Retrieval)                               │     │
│   │  - Term frequency-based scoring                                   │     │
│   │  - Fast, interpretable, no training required                     │     │
│   │  - Struggles with paraphrasing/synonyms                          │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │  ADVANCED: Sentence-BERT (Dense Retrieval)                       │     │
│   │  - Semantic embedding similarity (cosine)                        │     │
│   │  - Handles paraphrasing and synonyms well                        │     │
│   │  - Pre-trained: all-MiniLM-L6-v2 or all-mpnet-base-v2           │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│   OUTPUT: Top-k evidence chunks per summary sentence                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     GROUNDING & PRESENTATION                                 │
│                                                                              │
│   - Link each summary sentence to retrieved evidence                        │
│   - Highlight matching spans in source text                                 │
│   - Display confidence scores for each link                                 │
│   - Interactive UI: click sentence → show evidence                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Model Selection

### Baseline: TextRank (Extractive Summarization)

**Why TextRank?**
- **Zero hallucination risk:** Only selects existing sentences
- **No training required:** Unsupervised graph-based algorithm
- **Interpretable:** Clear ranking mechanism
- **Fast inference:** No neural network overhead

**Implementation:** Use `sumy` library or `gensim.summarization`

**Limitations:** Summaries can be choppy; no novel phrasing; may miss key information if not in high-centrality sentences.

### Advanced: BART and T5 (Abstractive Summarization)

#### Facebook's BART (Bidirectional and Auto-Regressive Transformer)

| Model | Parameters | Context Length | Notes |
|-------|------------|----------------|-------|
| `facebook/bart-base` | 139M | 1024 tokens | Faster, lighter |
| `facebook/bart-large-cnn` | 406M | 1024 tokens | Fine-tuned for summarization |

**Why BART?**
- Pre-trained denoising autoencoder—excellent for text generation
- `bart-large-cnn` specifically fine-tuned on CNN/DailyMail news summarization
- Strong abstractive capabilities with reasonable faithfulness

#### Google's T5 (Text-to-Text Transfer Transformer)

| Model | Parameters | Context Length | Notes |
|-------|------------|----------------|-------|
| `t5-small` | 60M | 512 tokens | Minimal resources |
| `t5-base` | 220M | 512 tokens | Good balance |
| `google/flan-t5-base` | 250M | 512 tokens | Instruction-tuned variant |

**Why T5?**
- Unified text-to-text framework ("summarize: [text]")
- Multiple sizes available for resource constraints
- FLAN-T5 variant offers improved instruction-following

### Long Document Handling (Exploration Area)

For papers exceeding model context limits (1024 tokens ≈ 750 words):

| Approach | Description |
|----------|-------------|
| **Truncation** | Use introduction + conclusion only |
| **Hierarchical** | Summarize sections → summarize summaries |
| **Sliding Window** | Generate summaries per chunk, then merge |
| **Long-context Models** | Longformer, LED, LongT5 (exploration) |

---

## Retrieval Methods

### Baseline: BM25 (Best Match 25)

**Algorithm:** TF-IDF variant that scores documents based on query term frequency, inverse document frequency, and document length normalization.

```python
# Pseudocode for BM25 retrieval
from rank_bm25 import BM25Okapi

# Index source document chunks
tokenized_chunks = [chunk.split() for chunk in source_chunks]
bm25 = BM25Okapi(tokenized_chunks)

# For each summary sentence, retrieve top-k evidence
for sentence in summary_sentences:
    scores = bm25.get_scores(sentence.split())
    top_k_indices = scores.argsort()[-k:][::-1]
    evidence = [source_chunks[i] for i in top_k_indices]
```

**Hyperparameters to tune:** k1 (term saturation), b (length normalization)

### Advanced: Sentence Transformers (Dense Retrieval)

**Algorithm:** Encode both summary sentences and source chunks into dense vectors; retrieve by cosine similarity.

```python
# Pseudocode for dense retrieval
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode source chunks (do once)
chunk_embeddings = model.encode(source_chunks)

# For each summary sentence
for sentence in summary_sentences:
    query_embedding = model.encode(sentence)
    similarities = np.dot(chunk_embeddings, query_embedding)
    top_k_indices = similarities.argsort()[-k:][::-1]
    evidence = [source_chunks[i] for i in top_k_indices]
```

**Recommended Models:**
- `all-MiniLM-L6-v2` — Fast, 384-dim embeddings, good quality
- `all-mpnet-base-v2` — Higher quality, 768-dim, slower
- `SPECTER` — Specifically trained on scientific papers (exploration)

### Hybrid Retrieval (Exploration Area)

Combine BM25 and dense retrieval for best of both:
1. BM25 for initial candidate selection (top-100)
2. Re-rank candidates using Sentence-BERT
3. Return top-k after re-ranking

---

## Evaluation Framework

### Factual Consistency (Primary Metric)

**Method:** NLI-based claim verification using DeBERTa-v3

```
For each (claim, evidence) pair:
    If NLI predicts "contradiction" → hallucination
    If NLI predicts "entailment" → grounded
    If NLI predicts "neutral" → unverifiable

Hallucination Rate = # contradictions / # total claims
```

**Model:** `microsoft/deberta-v3-large-nli` or `facebook/bart-large-mnli`

### Reference Alignment (Coverage Metric)

**Method:** Measure distribution of evidence across source document

```
coverage_score = 1 - std([evidence_positions]) / max_position
```

Penalize summaries that only cite the introduction or conclusion.

### Additional Metrics (Exploration)

| Metric | Tool | Purpose |
|--------|------|---------|
| ROUGE-1/2/L | `rouge-score` | N-gram overlap with reference |
| BERTScore | `bert-score` | Semantic similarity to reference |
| Grammarly API | External | Fluency and readability |
| Human Evaluation | Manual | Ground truth for subset |

---

## Implementation Stack

| Component | Library/Tool |
|-----------|--------------|
| **UI Framework** | Streamlit |
| **PDF Parsing** | PyMuPDF, pdfplumber, or GROBID |
| **Extractive Summarization** | sumy, gensim |
| **Abstractive Summarization** | Hugging Face Transformers (BART, T5) |
| **Sparse Retrieval** | rank-bm25 |
| **Dense Retrieval** | sentence-transformers |
| **NLI Evaluation** | Hugging Face Transformers (DeBERTa) |
| **Text Processing** | spaCy, nltk |
| **Vector Storage** | FAISS (optional, for larger scale) |

---

## Areas for Exploration and Adjustment

### Model Variations
- [ ] Compare BART vs. T5 vs. FLAN-T5 on same dataset
- [ ] Experiment with Pegasus (specifically designed for abstractive summarization)
- [ ] Try SciBERT embeddings for scientific domain retrieval
- [ ] Evaluate LongT5 or LED for full-paper processing without truncation

### Retrieval Enhancements
- [ ] Implement hybrid BM25 + dense retrieval pipeline
- [ ] Experiment with cross-encoder re-ranking (slower but more accurate)
- [ ] Tune chunk size (256, 512, 1024 tokens) for optimal retrieval
- [ ] Add section-aware retrieval (weight methodology vs. results differently)

### Grounding Strategies
- [ ] Post-generation filtering: Remove summary sentences with no strong evidence
- [ ] Constrained decoding: Guide generation to stay faithful to source
- [ ] Iterative refinement: Regenerate low-grounding sentences

### Data Scope
- [ ] Start with CS papers from S2ORC (familiar domain)
- [ ] Expand to biomedical papers if successful
- [ ] Test on user-uploaded papers for generalization

### UI/UX Exploration
- [ ] Sentence-level highlighting with confidence gradients
- [ ] Side-by-side view: summary ↔ evidence
- [ ] Evidence quality indicators (entailment score display)
- [ ] Export functionality (summary + citations)

---

## References

1. **S2ORC: The Semantic Scholar Open Research Corpus**  
   Lo, K., Wang, L.L., Neumann, M., Kinney, R., & Weld, D.S. (2020). ACL 2020.  
   https://github.com/allenai/s2orc

2. **BART: Denoising Sequence-to-Sequence Pre-training**  
   Lewis, M., et al. (2020). ACL 2020.  
   https://arxiv.org/abs/1910.13461

3. **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)**  
   Raffel, C., et al. (2020). JMLR.  
   https://arxiv.org/abs/1910.10683

4. **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**  
   Reimers, N. & Gurevych, I. (2019). EMNLP 2019.  
   https://arxiv.org/abs/1908.10084

5. **TextRank: Bringing Order into Texts**  
   Mihalcea, R. & Tarau, P. (2004). EMNLP 2004.  
   https://aclanthology.org/W04-3252/

6. **BM25: The Probabilistic Relevance Framework**  
   Robertson, S. & Zaragoza, H. (2009). Foundations and Trends in IR.

7. **Evaluating Factual Consistency in Abstractive Summarization**  
   Kryscinski, W., et al. (2020). EMNLP 2020.  
   https://arxiv.org/abs/1910.12840

8. **DeBERTa: Decoding-enhanced BERT with Disentangled Attention**  
   He, P., et al. (2021). ICLR 2021.  
   https://arxiv.org/abs/2006.03654

9. **OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs**  
   Ajith, A., et al. (2024). arXiv:2411.14199.  
   https://arxiv.org/abs/2411.14199

10. **GROBID: A Machine Learning Software for Extracting Information from Scholarly Documents**  
    Lopez, P. (2009). https://github.com/kermitt2/grobid

---

## Next Steps

1. **Week 4:** Set up development environment; download S2ORC CS subset; implement basic preprocessing pipeline
2. **Week 5:** Implement TextRank baseline; integrate BART/T5; generate initial summaries
3. **Week 6:** Implement BM25 and Sentence-BERT retrieval; link summaries to evidence
4. **Week 7:** Build Streamlit UI with interactive grounding display
5. **Week 8:** Implement NLI-based evaluation; run systematic experiments
6. **Week 9:** Add PDF upload capability; conduct error analysis
7. **Week 10:** Finalize UI; write report; prepare demo

---

*This document is intended to be a living reference. Update sections as exploration reveals new insights or constraints change.*
