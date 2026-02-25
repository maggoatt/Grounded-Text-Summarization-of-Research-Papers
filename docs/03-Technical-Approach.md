# 3. Technical Approach

This section describes the technical approach for **grounded text summarization** of research papers: we generate summaries from full-text papers, retrieve evidence from the source for each summary sentence, and verify factual consistency using natural language inference (NLI). The system is implemented as a **pipeline** with sequential stages and, within some stages, **alternative methods** (baseline vs. advanced) that can be compared.

---

## 3.1 Pipeline Overview

The pipeline has five main stages. Data flows sequentially from paper selection through summarization, retrieval, and grounding; the summarization and retrieval stages each support two methods (baseline and advanced) that are evaluated in parallel for comparison.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  INPUT: User selects a paper (from preprocessed S2ORC subset via Streamlit UI)   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: PREPROCESSING (offline + per-paper)                                    │
│  • Input:  S2ORC JSON (raw or from data/{corpusid}.json)                        │
│  • Output: Paper with sections list [{section_title, text}, ...]                │
│  • Detail: Section text is extracted from character-offset annotations;          │
│            papers with &lt;4 sections are dropped. No chunking here—chunking is      │
│            done inside summarization (by section) and retrieval (by section).   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: SUMMARIZATION (one of two methods)                                    │
│  • Input:  Full paper (sections)                                                │
│  • Output: Single summary string (then split into sentences for retrieval)      │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────────┐  │
│  │ BASELINE: TextRank (extractive) │  │ ADVANCED: BART (abstractive)         │  │
│  │ • TF-IDF + cosine similarity   │  │ • Encoder–decoder transformer, 406M   │  │
│  │ • PageRank on sentence graph   │  │ • 1024-token context; beam search     │  │
│  │ • Top-k central sentences      │  │ • Section-wise then hierarchical     │  │
│  │ • No neural net; no training   │  │   reduction if &gt;1024 tokens           │  │
│  └─────────────────────────────────┘  └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: EVIDENCE RETRIEVAL (one of two methods, per summary sentence)          │
│  • Input:  Summary sentence (query); source = section-level chunks (title+text)│
│  • Output: Top-k source chunks (+ section title) for that sentence              │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────────┐  │
│  │ BASELINE: BM25 (sparse)         │  │ ADVANCED: Sentence-BERT + FAISS      │  │
│  │ • TF-IDF–style term scoring    │  │ • all-MiniLM-L6-v2 → 384-dim vectors  │  │
│  │ • Exact term overlap            │  │ • FAISS index; cosine similarity      │  │
│  │ • rank_bm25 over tokenized     │  │ • Semantic match (paraphrase-friendly)│  │
│  │   section texts                 │  │ • No training; pre-trained only      │  │
│  └─────────────────────────────────┘  └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: GROUNDING / FACTUAL CONSISTENCY (NLI)                                  │
│  • Input:  (summary_sentence, retrieved_evidence_chunk) pairs                    │
│  • Output: Per-sentence label: entailment / neutral / contradiction + scores    │
│  • Model:  NLI model (e.g. cross-encoder/nli-deberta-v3-base or similar)         │
│  • Use:    Hallucination = contradiction; grounded = entailment; unverifiable =   │
│            neutral; aggregate metrics (e.g. % entailed, % contradicted)          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: PRESENTATION (Streamlit UI)                                             │
│  • Input:  Paper metadata, full summary, per-sentence evidence + NLI verdicts    │
│  • Output: User sees summary; can select a sentence and view supporting evidence │
│            and section context (and optionally confidence / hallucination risk)  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Sequential vs. parallel:** The flow **Input → Preprocessing → Summarization → Retrieval → Grounding → UI** is sequential. The **baseline and advanced methods** for summarization (TextRank vs. BART) and for retrieval (BM25 vs. Sentence-BERT+FAISS) are **alternatives** run for comparison, not simultaneously in one user path (e.g. user picks one summarizer, then one retriever, or we run both and compare in evaluation).

---

## 3.2 Data and Preprocessing

- **Source:** Papers come from the S2ORC corpus (Semantic Scholar Open Research Corpus), obtained via the Semantic Scholar Datasets API as gzipped JSONL shards, then filtered (CS-related keywords), cleaned (fields irrelevant to summarization removed), and formatted into per-paper JSON files in `data/` (see the `data_processing/` scripts).
- **Per-paper input to the pipeline:** A single JSON object with `corpusid`, `title`, `authors`, `sections` (each section: `section_title`, `text`). Section text is derived from S2ORC character-offset annotations (section headers and paragraphs) in `format_cleaned_papers.py`.
- **Output of preprocessing for the rest of the pipeline:** The same structure; no additional chunking is applied at this stage. Summarization uses section text (and, for BART, may split by token count); retrieval uses section-level units (section title + text) as chunks for BM25 and for Sentence-BERT/FAISS.

---

## 3.3 Summarization

We use two approaches: an **extractive baseline** and an **abstractive advanced** method.

**Baseline — TextRank (extractive)**  
- **Type:** Unsupervised, graph-based; no neural network and no training.  
- **Mechanism:** Sentences are represented with TF-IDF vectors; pairwise cosine similarity forms a sentence–sentence graph; PageRank (or equivalent) ranks sentences by “centrality.” The top-k sentences are selected and concatenated in document order to form the summary.  
- **Implementation:** In this project, implemented with `sklearn.feature_extraction.text.TfidfVectorizer`, `sklearn.metrics.pairwise.cosine_similarity`, and `networkx` for the graph and ranking (see `summarization/summarize.py`).  
- **Input:** Full body text (e.g. section texts joined); optionally with section boundaries for ordering. **Output:** One summary string (then split into sentences for retrieval and NLI).

**Advanced — BART (abstractive)**  
- **Type:** Neural sequence-to-sequence model; **encoder–decoder transformer** (406M parameters), pre-trained as a denoising autoencoder and fine-tuned for summarization (e.g. CNN/DailyMail). No project-specific training; we use the pre-trained `facebook/bart-large-cnn` model.  
- **Constraints:** Maximum input length 1024 tokens (BART’s limit). Longer papers are handled by a **hierarchical/sliding-window** strategy: (1) summarize each section independently (each section truncated to 1024 tokens if needed); (2) concatenate section summaries; (3) if the concatenation exceeds 1024 tokens, group summaries into 1024-token chunks and recursively summarize until the result fits in 1024 tokens; (4) run BART once on that final concatenation to produce the paper summary. Generation uses beam search (e.g. 6 beams) with length penalty.  
- **Input:** Paper sections (or their summaries in the multi-step case). **Output:** One summary string (then split into sentences).

---

## 3.4 Evidence Retrieval

For each **summary sentence**, we retrieve a small number of **source chunks** (section-level: section title + text) that can support or refute that sentence. Two methods are implemented for comparison.

**Baseline — BM25 (sparse retrieval)**  
- **Type:** Sparse, lexical retrieval; no neural network.  
- **Mechanism:** Source chunks are tokenized (e.g. lowercase, alphanumeric tokens); a BM25 index (e.g. `rank_bm25.BM25Okapi`) is built over these tokenized chunks. For each summary sentence, the same tokenization is applied and BM25 returns the top-k chunks by score.  
- **Input:** Summary sentence (query string); index built from section-level chunks. **Output:** Top-k chunks plus their section titles for display.

**Advanced — Sentence-BERT + FAISS (dense retrieval)**  
- **Type:** Dense retrieval using a **sentence encoder** (no decoder); no project-specific training.  
- **Mechanism:** We use **Sentence-BERT** (e.g. `all-MiniLM-L6-v2`) to map each chunk and each summary sentence to a fixed-dimensional vector (384-dim). Chunk vectors are stored in a **FAISS** index (e.g. `IndexFlatIP` with L2-normalized vectors for cosine similarity). At query time, the summary sentence is encoded and FAISS returns the top-k nearest chunk indices.  
- **Input:** Same as BM25 (query sentence; chunk index). **Output:** Top-k chunks + section titles.  
- **Advantage over BM25:** Semantic similarity: paraphrased or synonym-heavy sentences can still match the right source chunk.

---

## 3.5 Grounding and Factual Consistency (NLI)

- **Goal:** Decide whether each summary sentence is **supported** by the retrieved evidence (entailment), **contradicted** (hallucination), or **neither** (neutral/unverifiable).  
- **Method:** **Natural language inference (NLI)**. We treat the **evidence chunk** as the premise and the **summary sentence** as the hypothesis, and run a pre-trained NLI model (e.g. **DeBERTa**-based: `cross-encoder/nli-deberta-v3-base` or similar in the notebook; alternatives in the doc include `microsoft/deberta-v3-large-nli` or `facebook/bart-large-mnli`). The model outputs a three-way label (entailment / neutral / contradiction) and optionally probabilities.  
- **Input:** Pairs (evidence passage, summary sentence); passages may be truncated to fit the NLI model’s max length (e.g. 512 tokens). **Output:** Per-pair label and scores; per-sentence aggregate (e.g. best entailment over retrieved chunks). These outputs feed into metrics such as fraction of summary sentences with at least one entailed evidence, or fraction contradicted (hallucination rate).

---

## 3.6 User Interface and Integration

- **Framework:** Streamlit.  
- **Flow:** User selects a paper from the preprocessed set (dropdown keyed by paper ID/title); chooses a summarization model (TextRank or BART); triggers summary generation; sees the full summary and can select individual sentences; for each selected sentence, the UI can show retrieved evidence and section context (and, when wired, NLI verdict and scores).  
- **Implementation:** `app.py` loads papers from `data/*.json`, calls `summarization.summarize` for either `generate_summary_textrank` or `generate_summary` (BART), splits the summary into sentences for display and selection. Evidence retrieval and NLI are implemented in `Evidence_Retrieval.ipynb` and the summarization/evaluation notebook; integration into the Streamlit app for full end-to-end display is in progress.

---

## 3.7 Summary Table

| Component            | Role                    | Input → Output                                                                 | Method / model (brief)                          |
|----------------------|-------------------------|---------------------------------------------------------------------------------|-------------------------------------------------|
| Preprocessing        | Section extraction      | S2ORC/JSON paper → `sections: [{section_title, text}]`                          | Character-offset parsing (`data_processing/`)    |
| Summarization        | Produce summary         | Sections → one summary string                                                  | TextRank (extractive) or BART (abstractive)     |
| Retrieval            | Evidence per sentence   | Summary sentence + chunk index → top-k chunks                                  | BM25 (sparse) or Sentence-BERT + FAISS (dense)  |
| Grounding            | Factual check           | (sentence, chunk) pairs → entailment/neutral/contradiction                     | NLI model (e.g. DeBERTa cross-encoder)          |
| UI                   | Interaction             | User choices + model outputs → display                                         | Streamlit                                       |

This technical approach reuses and extends the design from the project proposal and current codebase, with explicit detail on neural architectures (BART as encoder–decoder transformer; Sentence-BERT as encoder-only; NLI as classification over premise–hypothesis pairs), inputs and outputs per stage, and the placement of baseline vs. advanced methods in the pipeline.

---

## 3.8 Evaluation and Benchmarking APIs

Beyond the core pipeline, we benchmark the **quality of generated summaries** using a set of external or library-based evaluation APIs, implemented in `Summarization_Model_Pipeline.ipynb`.

- **Readability (Textstat)**: We use the `textstat` Python library to compute standard readability scores for each summary, including Flesch Reading Ease, Flesch–Kincaid Grade, Gunning Fog Index, SMOG Index, and Dale–Chall Score. These metrics quantify how easy each summary is to read and allow direct comparison of **TextRank vs. BART** on accessibility.

- **Grammar and style (LanguageTool HTTP API via `language_tool_python`)**: We call the LanguageTool grammar checker through the `language_tool_python` client, which wraps the public HTTP API (`https://languagetool.org/http-api/`). For each summary we count total issues, categorize them (e.g., `TYPOS`), and compute an error rate (errors per word) to compare grammatical correctness across models.

- **Fluency (GPT‑2 perplexity)**: Using Hugging Face `transformers` (`GPT2LMHeadModel` and `GPT2TokenizerFast`), we compute **perplexity** over each summary with a sliding-window strategy (to respect GPT‑2’s context length). Lower perplexity indicates more fluent, language-model–friendly text; we report perplexity for both TextRank and BART summaries.

- **Factual consistency (NLI API via `transformers`)**: As described in §3.5, we load a pre-trained NLI model (e.g. `cross-encoder/nli-deberta-v3-base`) through the `transformers` API. This model provides three-way labels (entailment / neutral / contradiction) and probabilities for each (evidence, sentence) pair, which we aggregate into metrics such as hallucination rate.

- **Content overlap (ROUGE via `rouge-score`)**: With the `rouge-score` library (`rouge_scorer.RougeScorer`), we compute ROUGE‑1, ROUGE‑2, and ROUGE‑L between each generated summary and the paper’s abstract (when available) treated as a reference summary. These scores provide a standard n‑gram overlap benchmark for summarization quality.

Together, these evaluation APIs give a multi-dimensional view of system behavior—readability, grammatical correctness, fluency, overlap with the original abstract, and factual grounding—allowing us to benchmark baseline vs. advanced models and track improvements over time.
