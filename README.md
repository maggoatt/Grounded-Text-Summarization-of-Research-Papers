# Grounded Text Summarization of Research Papers
An interactive interface and simple benchmarking project to generate and analyze various models' grounded summarizations for research papers.

## External Libraries/Packages
### Data Pipeline
- ```python-dotenv```
- ```requests```

### Summarization Models
- ```scikit-learn```
- ```networkx```
- ```transformers```
- ```torch```

### Retrieval
- ```sentence-transformers```
- ```faiss-cpu```
- ```rank-bm25```

### Evaluation
- ```textstat```
- ```language-tool-python```
- ```rouge-score```

### Text Processing
- ```nltk```

### UI
- ```streamlit```

## Use of Publicly Available Code
### Summarization Pipeline
1. Workflow to implement TextRank:

Adapted from: ERRAJI, Yassine (June 19 2025). ["Understanding TextRank: A Deep Dive into Graph-Based Text Summarization and Keyword Extraction"](https://medium.com/@yassineerraji/understanding-textrank-a-deep-dive-into-graph-based-text-summarization-and-keyword-extraction-905d1fb5d266).
Medium Article.
+ Modifications: sentence tokenization, summarization concatenation (2/8 lines) 

## Written Code
1. From ```src/data_processing```:
- ```filter_cs_papers.py```: connects to S2ORC corpus, extracts ```TARGET_PAPERS``` number of CS-related papers via matching keywords in title (257 lines)
- ```clean_existing_papers.py```: keeps relevant info from each paper (metadata, biliographies, sections, text), saving all to a ```.jsonl``` file (113 lines)
- ```format_cleaned_papers.py```: extracts relevant info (section titles and corresponding section text from S2ORC character offsets), saving each paper as ```{corpusid}.json``` for pipeline use (195 lines)
- NOTE: more information can be found in this directory's ```README.md```.

2. From ```src/workflows```:
+ ```app.py```: ```streamlit``` implementation of interactive UI + connection between all workflows, pulls pre-generated summaries, retrievals, and metrics for viewing (281 lines)
+ ```compute_metrics.py```: generates and saves metrics (readability, perplexity, etc.) for evaluating TextRank and BART summaries for UI use (73 lines)

3. From ```src/workflows/summarization```: 
+ ```Summarization_Model_Pipeline.ipynb```: entire implementation + testing of TextRank and BART summarization methods, summarization metrics, and retrieval metrics (~480 lines)
+ ```summarize.py```: helper functions for generating + saving summaries for the UI, takes code from notebook (198 lines)
+ ```metrics.py```: helper functions for generating metrics for ```compute_metrics.py``` to use (94 lines)

4. From ```src/workflows/retrieval```:
+ ```Evidence_Retrieval.ipynb```: BM25 implementation for retrieving top matches (50 lines)
+ ```PaperEmbeddings.ipynb```: creates and saves embeddings for chunks of text from original papers (55 lines)
+ ```bm_retrieval.py```: helper function implementation of BM25 (58 lines)
+ ```faiss_retrieval.py```: loads previously saved embeddings and finds top evidence matches (sentences and section titles) for summaries (40 lines)
+ ```retrieval_save.py```: saves all resulting retrievals for all papers to be used in UI and for metric evaluations (87 lines)

All other directories contain evidence of our presaved data, summaries, retrievals, and metrics/benchmarks.
