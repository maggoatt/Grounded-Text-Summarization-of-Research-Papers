# Technical Approach: Grounded Text Summarization of Research Papers

## Data Source: S2ORC Corpus

### Overview

S2ORC contains 81.1M English-language academic papers as structured JSON objects with pre-segmented sections, paragraphs, bibliographic references, and metadata.

| Property | Value |
|----------|-------|
| Total Papers | 81.1M (8.1M with full text) |
| Format | JSON (one object per paper) |
| Pre-segmented | ✅ Paragraphs, sections labeled |
| Metadata | Field of study, authors, citations, year |
| Access | Data use agreement required |

### Recommended Subset Strategy

For this project, we will use a **filtered subset** rather than the full corpus:

| Filter | Rationale |
|--------|-----------|
| **Field of Study:** Computer Science | Familiar domain, manageable vocabulary |
| **Has Full Text:** True | Need body text for grounding |
| **Year:** 2018-2023 | Recent, relevant papers |
| **Target Size:** 5,000-10,000 papers | Sufficient for evaluation, manageable storage |

Approximately **2-5 GB** of JSON data.

---

## Complete S2ORC Workflow

### Phase 1: Data Acquisition & Preprocessing

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Download S2ORC Subset                                  │
│  - Request access via Semantic Scholar API                      │
│  - Download relevant shards (CS papers with full text)          │
│  - Storage: ~2-5 GB for filtered subset                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Filter & Extract                                       │
│  - Parse JSON files                                             │
│  - Filter by: field_of_study, has_pdf_parse, year              │
│  - Extract: paper_id, title, abstract, body_text, sections     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Chunk Body Text                                        │
│  - Split body_text into chunks (~256-512 tokens each)          │
│  - Preserve section labels (intro, methods, results, etc.)     │
│  - Store: paper_id → [chunk_1, chunk_2, ..., chunk_n]          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Build Retrieval Index                                  │
│  - Compute embeddings for all chunks (Sentence-BERT)           │
│  - Store in FAISS index for fast similarity search             │
│  - Also build BM25 index for baseline comparison               │
└─────────────────────────────────────────────────────────────────┘

OUTPUT: Preprocessed dataset ready for summarization & retrieval
```

### Phase 2: Summarization Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: User selects paper (by ID, title search, or filter)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Prepare Input for Summarization                        │
│  - Retrieve full text from preprocessed data                    │
│  - Use sliding window to chunk paper into ~1024 token segments │
│  - Select key sections: [intro] [methods] [conclusion]         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: Generate Summary                                       │
│  - BASELINE: TextRank extracts top-k central sentences         │
│  - ADVANCED: BART generates abstractive summary                │
│  - Split summary into individual sentences for retrieval       │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 3: Evidence Retrieval & Grounding

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7: Retrieve Evidence for Each Summary Sentence            │
│  - For each sentence in generated summary:                      │
│    - Query BM25 index → get top-5 chunk candidates             │
│    - Query FAISS index → get top-5 chunk candidates            │
│    - Merge and re-rank results                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 8: Verify Grounding (NLI)                                 │
│  - For each (summary_sentence, evidence_chunk) pair:           │
│    - Run through DeBERTa NLI model                             │
│    - Label: entailed / neutral / contradicted                  │
│  - Flag sentences with no entailed evidence as "ungrounded"    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: Summary with linked evidence + grounding scores        │
│  - Each sentence → list of supporting chunks                   │
│  - Each link → entailment confidence score                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## System Architecture

### Overall Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INPUT                                      │
│              (Select paper from S2ORC subset via Streamlit UI)              │
│           Future potential: query specific topics within a paper            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING MODULE                                  │
│  ┌─────────────────┐              ┌─────────────────┐                       │
│  │  S2ORC Parser   │──────────────│  Text Cleaner   │                       │
│  │  (JSON → text)  │              │  (normalize)    │                       │
│  └─────────────────┘              └─────────────────┘                       │
│                                      │                                       │
│                                      ▼                                       │
│                    ┌─────────────────────────────────┐                      │
│                    │    CHUNKING & SEGMENTATION      │                      │
│                    │  - Sliding window (~512 tokens) │                      │
│                    │  - Preserve section labels      │                      │
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
│   │  ADVANCED: Facebook's BART (Abstractive)                         │     │
│   │  - Encoder-decoder transformer architecture                       │     │
│   │  - Generates novel, fluent summaries                             │     │
│   │  - Higher hallucination risk → requires grounding                │     │
│   │  - Uses sliding window to chunk paper into processable tokens    │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│   OUTPUT: Generated summary (displayed in final UI)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL MODULE                                      │
│         (User clicks on a summary sentence → retrieve 1-3 evidence)         │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │  BASELINE: BM25 (Sparse Retrieval)                               │     │
│   │  - TF-IDF based scoring                                          │     │
│   │  - Top-k chunks based on exact term match                        │     │
│   │  - Fast, interpretable, no training required                     │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │  ADVANCED: Sentence-BERT + FAISS (Dense Retrieval)               │     │
│   │  - Sentence-BERT (all-MiniLM-L6-v2) creates embeddings          │     │
│   │  - FAISS compares embeddings via cosine similarity              │     │
│   │  - Semantic-based: handles paraphrasing well                     │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│   OUTPUT: 1-3 evidence sentences + section title for location context      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     GROUNDING & PRESENTATION (Streamlit)                     │
│                                                                              │
│   - Display summary with clickable sentences                                │
│   - Show retrieved evidence from original text                              │
│   - Display section title to help user locate original chunk                │
│   - Interactive UI: click sentence → show supporting evidence               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Model Selection & Setup

### Important Clarification: Pre-trained Models (No Training Required)

**You are NOT training BERT/BART/T5 from scratch.** These are large pre-trained models that you load and use directly. The "setup" involves:

1. **Installing libraries** (`pip install transformers sentence-transformers`)
2. **Loading pre-trained weights** (automatic download from HuggingFace)
3. **Running inference** (feed text in, get summary/embeddings out)

```python
# This is ALL you need to "set up" BART for summarization
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer(paper_text, max_length=150, min_length=50)
```

**Optional fine-tuning** (if time permits): You could fine-tune on scientific papers, but this is NOT required for a working system.

---

### Baseline: TextRank (Extractive Summarization)

**Why TextRank?**
- **Zero hallucination risk:** Only selects existing sentences
- **No training required:** Unsupervised graph-based algorithm
- **Interpretable:** Clear ranking mechanism
- **Fast inference:** No neural network overhead

**Implementation:** Use `sumy` library

```python
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

parser = PlaintextParser.from_string(paper_text, Tokenizer("english"))
summarizer = TextRankSummarizer()
summary_sentences = summarizer(parser.document, sentences_count=5)
```

**Limitations:** Summaries can be choppy; no novel phrasing; may miss key information if not in high-centrality sentences.

---

### Advanced: Facebook's BART (Abstractive Summarization)

| Model | Parameters | Context Length | Notes |
|-------|------------|----------------|-------|
| `facebook/bart-large-cnn` | 406M | 1024 tokens | Fine-tuned for summarization on CNN/DailyMail |

**Why BART?**
- Pre-trained denoising autoencoder—excellent for text generation
- Specifically fine-tuned on summarization tasks
- 1024 token context is larger than T5-base (512), better for papers

**Implementation:**

```python
from transformers import BartForConditionalGeneration, BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Paper text needs to be chunked via sliding window for long documents
inputs = tokenizer(paper_text, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=50)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

**Note:** For papers longer than 1024 tokens (~750 words), use a sliding window approach to chunk the paper, then either concatenate chunk summaries or summarize the most important sections (intro + methods + conclusion).

---

### Sentence-BERT + FAISS for Retrieval

**Clarification:** BERT itself is an encoder-only model—it produces embeddings, not text. You use **Sentence-BERT** for creating embeddings and **FAISS** for fast similarity search.

| Component | Role |
|-----------|------|
| **Sentence-BERT** (`all-MiniLM-L6-v2`) | Converts text → 384-dim vector embeddings |
| **FAISS** | Stores vectors & performs fast cosine similarity search |

**Why `all-MiniLM-L6-v2`?**
- Fast: ~14,000 sentences/sec on GPU, ~1,000 on CPU
- Good quality semantic similarity
- 80 MB model size—lightweight for deployment

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- INDEXING (done once during preprocessing) ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode all chunks from the paper
chunk_embeddings = model.encode(paper_chunks)  # Shape: (num_chunks, 384)

# Build FAISS index
dimension = 384
index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
faiss.normalize_L2(chunk_embeddings)   # Normalize for cosine
index.add(chunk_embeddings)

# Save index
faiss.write_index(index, "paper_chunks.index")

# --- RETRIEVAL (at query time) ---
query = "The model achieves state-of-the-art results."
query_vec = model.encode([query])
faiss.normalize_L2(query_vec)

k = 3  # Retrieve top 3 evidence chunks
distances, indices = index.search(query_vec, k)
evidence_chunks = [paper_chunks[i] for i in indices[0]]
```

---

### Long Document Handling

For papers exceeding model context limits (1024 tokens ≈ 750 words):

| Approach | Description | Complexity |
|----------|-------------|------------|
| **Truncation** | Use introduction + conclusion only | Easy |
| **Hierarchical** | Summarize sections → summarize summaries | Medium |
| **Sliding Window** | Generate summaries per chunk, then merge | Medium |
| **Long-context Models** | LongT5, LED | Advanced (exploration) |

**Recommended starting point:** Truncation (intro + methods + conclusion)

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

### Advanced: Sentence-BERT + FAISS (Dense Retrieval)

**Algorithm:** Encode both summary sentences and source chunks into dense vectors using Sentence-BERT; store and search using FAISS for speed.

| Property | Value |
|----------|-------|
| **Embedding Model** | `all-MiniLM-L6-v2` |
| **Embedding Dimension** | 384 |
| **Index Type** | FAISS IndexFlatIP (exact search) |
| **Similarity Metric** | Cosine similarity (via normalized inner product) |

```python
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode source chunks (done once, stored in FAISS index)
chunk_embeddings = model.encode(source_chunks)
faiss.normalize_L2(chunk_embeddings)

# Build FAISS index
index = faiss.IndexFlatIP(384)
index.add(chunk_embeddings)

# For each summary sentence, retrieve evidence
for sentence in summary_sentences:
    query_vec = model.encode([sentence])
    faiss.normalize_L2(query_vec)
    distances, indices = index.search(query_vec, k=3)
    evidence = [source_chunks[i] for i in indices[0]]
    # Also retrieve section_titles[i] to display location context
```

**Key advantage over BM25:** Handles paraphrasing and synonyms. If the summary says "outperforms" and the source says "achieves improvements", Sentence-BERT will still find the match.

---

## Evaluation Framework

### Summarization Evaluation

Compare TextRank (baseline) vs. BART (advanced) on:

| Metric | Description | How to Measure |
|--------|-------------|----------------|
| **Coherence** | Is the summary logically organized and readable? | Human evaluation (1-5 scale) |
| **Informativeness** | Does the summary capture key information? | ROUGE scores, human evaluation |
| **Hallucination** | Does the summary contain fabricated content? | NLI-based verification |

#### Hallucination Detection (NLI-based)

```
For each (summary_sentence, evidence_chunk) pair:
    If NLI predicts "contradiction" → hallucination
    If NLI predicts "entailment" → grounded
    If NLI predicts "neutral" → unverifiable

Hallucination Rate = # contradictions / # total claims
```

**Model:** `microsoft/deberta-v3-large-nli` or `facebook/bart-large-mnli`

**Expected Result:** TextRank should have ~0% hallucination (extractive), while BART may have some hallucinations that our grounding system helps identify.

---

### Retrieval Evaluation

Compare BM25 (baseline) vs. Sentence-BERT + FAISS (advanced) on:

| Metric | Description | How to Measure |
|--------|-------------|----------------|
| **True Relevancy** | Does retrieved evidence actually support the summary sentence? | Human evaluation, NLI entailment score |
| **Hallucination Detection** | Can the retrieval system identify unsupported claims? | % of summary sentences with no entailed evidence |

#### Relevancy Scoring

```python
# For each (summary_sentence, retrieved_evidence) pair:
# Run NLI to check if evidence entails the claim

relevancy_score = entailment_probability  # 0.0 to 1.0
```

**Expected Result:** Sentence-BERT should retrieve more semantically relevant evidence than BM25, especially when summary uses different words than source.

---

### Additional Metrics (Optional)

| Metric | Tool | Purpose |
|--------|------|---------|
| ROUGE-1/2/L | `rouge-score` | N-gram overlap with original abstract |
| BERTScore | `bert-score` | Semantic similarity to original abstract |
| Coverage Score | Custom | Evidence distribution across document sections |

---

## Implementation Stack

| Component | Library/Tool | Install |
|-----------|--------------|---------|
| **UI Framework** | Streamlit | `pip install streamlit` |
| **Extractive Summarization** | sumy | `pip install sumy` |
| **Abstractive Summarization** | Hugging Face Transformers (BART) | `pip install transformers` |
| **Sparse Retrieval** | rank-bm25 | `pip install rank-bm25` |
| **Dense Retrieval** | sentence-transformers | `pip install sentence-transformers` |
| **Vector Index** | FAISS | `pip install faiss-cpu` |
| **NLI Evaluation** | Hugging Face Transformers (DeBERTa) | (included with transformers) |
| **Text Processing** | nltk | `pip install nltk` |

### requirements.txt

```
streamlit>=1.28.0
transformers>=4.35.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
rank-bm25>=0.2.2
sumy>=0.11.0
nltk>=3.8.0
torch>=2.0.0
```

---

## Coding Scope Breakdown

**Yes, there is substantial coding work.** Here's a realistic breakdown:

### Module 1: Data Pipeline (~300-400 lines)

| File | Purpose | Estimated Lines |
|------|---------|-----------------|
| `data/download_s2orc.py` | Download and filter S2ORC subset | 80-100 |
| `data/preprocess.py` | Parse JSON, extract fields, clean text | 100-150 |
| `data/chunker.py` | Split papers into retrievable chunks | 80-100 |
| `data/index_builder.py` | Build BM25 and FAISS indices | 80-100 |

**Key tasks:**
- [ ] Write S2ORC JSON parser
- [ ] Implement text cleaning (remove LaTeX, special chars)
- [ ] Implement chunking with overlap
- [ ] Build and save retrieval indices

### Module 2: Summarization (~150-200 lines)

| File | Purpose | Estimated Lines |
|------|---------|-----------------|
| `models/textrank_summarizer.py` | Extractive baseline wrapper | 40-50 |
| `models/bart_summarizer.py` | BART wrapper with sliding window handling | 80-100 |
| `models/summarizer_factory.py` | Unified interface for both methods | 30-50 |

**Key tasks:**
- [ ] Wrap TextRank in consistent API
- [ ] Wrap BART with sliding window chunking for long papers
- [ ] Handle long documents (section selection: intro + methods + conclusion)

### Module 3: Retrieval (~200-250 lines)

| File | Purpose | Estimated Lines |
|------|---------|-----------------|
| `retrieval/bm25_retriever.py` | BM25 search over chunks | 60-80 |
| `retrieval/dense_retriever.py` | Sentence-BERT similarity search | 80-100 |
| `retrieval/hybrid_retriever.py` | Combine BM25 + dense (optional) | 60-80 |

**Key tasks:**
- [ ] Implement BM25 query interface
- [ ] Implement FAISS query interface
- [ ] Return top-k chunks with scores

### Module 4: Evaluation (~150-200 lines)

| File | Purpose | Estimated Lines |
|------|---------|-----------------|
| `evaluation/nli_checker.py` | DeBERTa entailment scoring | 80-100 |
| `evaluation/metrics.py` | Hallucination rate, coverage score | 50-80 |
| `evaluation/run_eval.py` | Batch evaluation over test set | 50-80 |

**Key tasks:**
- [ ] Load DeBERTa NLI model
- [ ] Score (summary, evidence) pairs
- [ ] Compute aggregate metrics

### Module 5: UI (~200-300 lines)

| File | Purpose | Estimated Lines |
|------|---------|-----------------|
| `app/streamlit_app.py` | Main Streamlit application | 150-200 |
| `app/components.py` | Reusable UI components | 50-100 |

**Key tasks:**
- [ ] Paper selection interface (search/filter)
- [ ] Summary display with clickable sentences
- [ ] Evidence panel showing source chunks
- [ ] Grounding score visualization

### Module 6: Utilities & Config (~100 lines)

| File | Purpose | Estimated Lines |
|------|---------|-----------------|
| `config.py` | Paths, model names, hyperparameters | 30-40 |
| `utils.py` | Text processing helpers | 50-60 |

---

### Total Coding Estimate

| Module | Lines | Complexity |
|--------|-------|------------|
| Data Pipeline | 300-400 | Medium |
| Summarization | 150-200 | Low (using libraries) |
| Retrieval | 200-250 | Medium |
| Evaluation | 150-200 | Medium |
| UI | 200-300 | Medium |
| Utilities | ~100 | Low |
| **TOTAL** | **~1,100-1,450 lines** | |

**Note:** This estimate assumes clean, well-documented code. The actual work also includes:
- Debugging and testing
- Hyperparameter tuning
- Writing evaluation scripts
- Documentation

---

### Suggested File Structure

```
project/
├── data/
│   ├── download_s2orc.py       # Download and filter S2ORC subset
│   ├── preprocess.py           # Parse JSON, extract fields, clean text
│   ├── chunker.py              # Sliding window chunking with section labels
│   └── index_builder.py        # Build BM25 and FAISS indices
├── models/
│   ├── textrank_summarizer.py  # Extractive baseline
│   ├── bart_summarizer.py      # BART abstractive summarization
│   └── summarizer_factory.py   # Unified interface
├── retrieval/
│   ├── bm25_retriever.py       # BM25 baseline retrieval
│   └── dense_retriever.py      # Sentence-BERT + FAISS retrieval
├── evaluation/
│   ├── nli_checker.py          # DeBERTa entailment scoring
│   ├── metrics.py              # Hallucination rate, relevancy metrics
│   └── run_eval.py             # Batch evaluation script
├── app/
│   ├── streamlit_app.py        # Main UI application
│   └── components.py           # Reusable UI components
├── config.py                   # Paths, model names, hyperparameters
├── utils.py                    # Text processing helpers
├── requirements.txt
└── README.md
```

---

## Areas for Exploration and Adjustment

### Summarization Variations
- [ ] Tune BART generation parameters (max_length, min_length, num_beams)
- [ ] Experiment with different section combinations for input (intro+methods+conclusion vs. full text)
- [ ] Try Pegasus as alternative abstractive model (specifically designed for summarization)

### Retrieval Enhancements
- [ ] Compare `all-MiniLM-L6-v2` vs. `SPECTER` (scientific domain embeddings)
- [ ] Implement hybrid BM25 + Sentence-BERT retrieval pipeline
- [ ] Tune chunk size (256, 512, 1024 tokens) for optimal retrieval
- [ ] Experiment with retrieving more evidence (1-3 vs. 3-5 chunks)

### Grounding Strategies
- [ ] Post-generation filtering: Flag summary sentences with no strong evidence
- [ ] Display confidence scores for each retrieved evidence chunk
- [ ] Color-code summary sentences by grounding strength

### Data Scope
- [ ] Start with small CS subset (1,000-5,000 papers)
- [ ] Scale to larger subset (10,000+) if pipeline is stable

### UI/UX Exploration (Future Potential)
- [ ] Allow user to query specific topics within a paper
- [ ] Side-by-side view: summary ↔ evidence with section context
- [ ] Evidence quality indicators (entailment score display)
- [ ] Paper browsing/filtering interface

---

## References

1. **S2ORC: The Semantic Scholar Open Research Corpus**  
   Lo, K., Wang, L.L., Neumann, M., Kinney, R., & Weld, D.S. (2020). ACL 2020.  
   https://github.com/allenai/s2orc

2. **BART: Denoising Sequence-to-Sequence Pre-training**  
   Lewis, M., et al. (2020). ACL 2020.  
   https://arxiv.org/abs/1910.13461

3. **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**  
   Reimers, N. & Gurevych, I. (2019). EMNLP 2019.  
   https://arxiv.org/abs/1908.10084

4. **TextRank: Bringing Order into Texts**  
   Mihalcea, R. & Tarau, P. (2004). EMNLP 2004.  
   https://aclanthology.org/W04-3252/

5. **BM25: The Probabilistic Relevance Framework**  
   Robertson, S. & Zaragoza, H. (2009). Foundations and Trends in IR.

6. **FAISS: A Library for Efficient Similarity Search**  
   Douze, M., et al. (2024). Meta AI.  
   https://github.com/facebookresearch/faiss

7. **Evaluating Factual Consistency in Abstractive Summarization**  
   Kryscinski, W., et al. (2020). EMNLP 2020.  
   https://arxiv.org/abs/1910.12840

8. **DeBERTa: Decoding-enhanced BERT with Disentangled Attention**  
   He, P., et al. (2021). ICLR 2021.  
   https://arxiv.org/abs/2006.03654

9. **OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs**  
   Ajith, A., et al. (2024). arXiv:2411.14199.  
   https://arxiv.org/abs/2411.14199

---

## Next Steps (Aligned with Milestones)

### Weeks 4-5: Data & Summarization Foundation
| Task | Owner | Module |
|------|-------|--------|
| Request S2ORC access, download CS subset | Richard | Data Pipeline |
| Write JSON parser and text cleaner | Richard | Data Pipeline |
| Implement sliding window chunking logic | Richard | Data Pipeline |
| Set up TextRank baseline | Lawrence | Summarization |
| Set up BART summarizer with truncation handling | Lawrence | Summarization |
| Initial summary comparison (coherence, informativeness) | Lawrence | Summarization |

### Week 6: Retrieval Implementation
| Task | Owner | Module |
|------|-------|--------|
| Build BM25 index over chunks | Maggie | Retrieval |
| Build FAISS index (Sentence-BERT embeddings) | Maggie | Retrieval |
| Implement retrieval query interface | Maggie | Retrieval |
| Connect summarization → retrieval pipeline | All | Integration |

### Weeks 7-8: UI & Evaluation
| Task | Owner | Module |
|------|-------|--------|
| Build Streamlit paper selection UI | Maggie | UI |
| Build summary + evidence display | Maggie | UI |
| Implement NLI-based grounding checker | Richard | Evaluation |
| Run evaluation on test subset | Richard | Evaluation |
| Compare baseline vs. advanced models | Lawrence | Evaluation |

### Weeks 9-10: Polish & Report
| Task | Owner | Module |
|------|-------|--------|
| Error analysis and debugging | All | - |
| UI polish and refinements | Maggie | UI |
| Final evaluation runs | Richard | Evaluation |
| Write project report | Lawrence | - |
| Prepare demo | All | - |

---

*This document is intended to be a living reference. Update sections as exploration reveals new insights or constraints change.*
