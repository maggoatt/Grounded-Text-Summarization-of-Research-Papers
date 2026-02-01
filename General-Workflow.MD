### PREPROCESSING:
Smaller subset of papers from S2ORC corpus

### USER INTERFACE: 
Develop UI to allow user to select one of those papers to summarize 
Future potential: allow user to query a specific paper about specific stuff
Using ```streamlit``` Python library for UI development

### SUMMARIZATION:
Advanced algorithm: Facebook’s BART (abstractive method)
- Need a sliding window to ‘chunk’ paper into token sizes large enough to be processed
- This summarization will be used for the final UI
Baseline algorithm: TextRank to select the top-k most central sentences
No hallucinations, can compare this to the advanced algorithm in terms of:
- Coherence
- Informativeness 
- Hallucination 

### RETRIEVAL:
Allow user to click on each sentence from summary
Retrieve few (1-3) sentences from original 
Advanced algorithm: Sentence-BERT (all-MiniLM-L6-v2 for faster retrieval) + FAISS
- sentence-BERT creates embeddings, FAISS compares embeddings 
- semantic-based 
Baseline algorithm: BM25 (TF-IDF based)
- top-k most relevant chunks based on exact term match
- Display section title of original text sentences to allow user to find OG chunk location
Evaluate the comparisons based on:
- True relevancy
- Hallucination
