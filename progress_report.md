1. Project Summary  
Inspired by concerns around hallucinated AI-generated citations/content in NeurIPS 2025 papers, this project addresses the lack of credibility in AI-generated summarizations of research papers by evaluating large language models on strict text-based summarization. We will develop an interactive, reference-grounded summarization system, which (1) generates research paper abstracts based on the original paper text, (2) allows users to view the original text behind a selected sentence in the generated summary by directly referencing segments of relevant content in the original text, and (3) displays metrics (relevancy, grammar, flow, etc.) regarding the quality of the summarized text. 

2. Team Accomplishments
Retrieved and preprocessed data from the S2ORC corpus into unique .json files, separating raw text into mapped section titles and section paragraphs.
Developed summarization techniques using TextRank (baseline model which extracts top-k sentences directly from text) and Facebook BART (advanced LLM-based model which generates summary), and retrieval techniques (in progress) using BM25 (baseline model) and Sentence-BERT + FAISS (advanced model).
Implemented benchmarking for evaluating summarization models based on readability, grammar/style, and fluency. 
Built user interface with Python Streamlit, which allows users to select a paper, view its metadata, and summarize with either TextRank or Facebook BART with options to view summarization evaluation results (from bullet point 3). 

3. Technical Approach
First, papers are retrieved from the S2ORC Semantic Scholar Datasets API/Corpus as JSONL shards, then filtered (CS-related keywords), cleaned (keeping only metadata and full body text JSON sections), and formatted into per-paper JSON files (exact details in Section 4: Data Sets). Each paper is stored as a JSON file identified by its unique corpus ID. 

The summarization pipeline includes two choices of models, TextRank (extractive baseline model)  and Facebook BART Large-CNN (LLM-based advanced method).

The baseline model, TextRank, is an unsupervised, graph-based model with no training needed. To start, each paragraph from the selected paper is concatenated into one large string, but sentences are still mapped back to their original sections using a dictionary. Sentences are represented with TF-IDF vectors and pairwise cosine similarity is computed to form a sentence-sentence graph. PageRank algorithm is used to score/rank sentences by centrality to the entire paper. The top-k sentences are selected and concatenated to form the summary. Final summaries are stored as {corpus_id}_textrank_summary.txt under the summaries/ directory (which is .gitignored).

The advanced model, Facebook BART Large-CNN, is a pre-trained encoder–decoder transformer (about 400M parameters) containing fine-tuned options for summarization, which we directly used in our pipeline with the facebook/bart-large-cnn model. 

Since BART is an older model, it has a more restrictive maximum input text length of 1024 tokens. Many (if not all) of our papers exceed this limit, with each section containing close to 1000 tokens! We handle this token limit by implementing this section-based sliding-window strategy (inspired by strategies from this forum): 
Summarize each section independently (truncate sections to 1024 tokens if exceeded)
Concatenate section summaries
If the concatenation exceeds 1024 tokens, group summaries into 1024-token chunks and recursively summarize until the result fits in 1024 tokens
Else, summarize the sections once more, with minimum token generation of 200 to avoid summaries that are too short
Please note that our generation parameters use beam search (6 beams) with a smaller length penalty to encourage longer sentence generation
The resulting summarization is stored as {corpus_id}_bart_summary.txt under the summaries/ directory.

For the retrieval (matching summarization sentences to chunks of text/sentences/paragraphs from the original paper) aspect of our project, we used BM25 (Best Match 25) for TF-based (term-frequency) scoring. It calculates its score by balancing term-frequency (how often words from the query appear), inverse document frequency (how rare a word from the query is in all sections), and length normalization (penalization on longer documents).

With BM25, for our data preprocessing, we retrieve the currently selected paper along with its metadata including its title and list of sections. We also made sure to concatenate the section titles to the body text so we wouldn’t miss any important text within the section title. We then used a simple regex tokenizer to normalize case and strip punctuation for BM25 to use. We then indexed the tokenized sections using BM25Okapi from the rank-bm25 library. Given the query sentence (the user selected summary sentence), we tokenized it the same way and calculated the BM25 scores against all sections and returned the top result back to the user along with the section that evidence came from. We observed that BM25 retrieves better when the summary mirrors the original text, but if the wording is only similar, it struggles. 

For the advanced model, we used a pretrained sentence embedding model all-MiniLM-L6-v2 in order to turn the text of the research papers into embeddings. That would create dense vectors for each sentence, which would be matched with cosine similarity. We were recommended for speed to use all-MiniLM-L6-v2 to create and save embeddings of our subset of research papers onto disk and to load them at query time. 

Finally, each sentence from the summary is encoded using the same model and linked to its top matching retrieval chunk and displayed on the interface through user selection. To do this, we use Facebook’s FAISS to take in the summary sentence that the user selected in order to compare that to the embeddings of the research paper. This model is currently being integrated and will be evaluated against BM25 in Week 9.

We display confidence scores on the UI as well. We expect our advanced method to outperform BM25, because our advanced model can handle similar words using embeddings, whereas BM25 sees no relation between similar words because it looks for exact matches. 

We used Python’s Streamlit library to develop the frontend user interface, where the user can interact with the research papers in our database (selection based on dropdown menu with text), generate summaries between the two models, view benchmarking statistics, and retrieve evidence directly from the original text. As of Monday, February 23, our UI looks like this:



See Experiments and Evaluation for our evaluation metrics and models. Some helpful resources we read up on are linked throughout the Technical Approach section above and the Experiments and Evaluation section.

4. Data Sets
We currently use S2ORC: The Semantic Scholar Open Research Corpus, which contains 81.1M full-text English-language academic papers. Each paper is represented as a JSON object consisting of the paper-specific metadata, paper text, and character-offset indexes from the paper text for each section title and section text. The distribution of papers by field of study within the dataset is pictured below.

The papers are obtained from the Semantic Scholars API, which we store as an S2ORC API key in our personal environment files.
To filter, clean, and prepare the data for UI display and summarization, we developed a three-stage workflow:
Step
Script (from data_processing/ dir)
Purpose
1
filter_cs_papers.py
Avoid retrieving too many papers; filters streamed content by latest release and CS keywords in first 5,000 characters of text body, including “computer science", "machine learning", "deep learning", "neural network", "artificial intelligence", etc. Number of extracted papers can be modified using TARGET_PAPERS parameter. 
2
clean_existing_papers.py
From filtered papers (stored as .jsonl file), store only relevant metadata, body text, and character offset (to find section titles and section paragraphs from text).
3
format_cleaned_papers.py
From cleaned papers (stored as .jsonl file), convert character offset information into a list of {section_title: text} sections, skipping papers with <4 sections (sign of poorly stored/managed paper). Store each paper within the data/ directory as {corpus_id}.json. 

Final processed papers are stored as a unique .json file that contains metadata (title, authors, etc.) and a list of section titles with section text.
5. Experiments and Evaluation
To evaluate the relevance, fluency, and other important factors between the various models, we implemented multiple benchmarking APIs into our pipeline. To avoid merge conflicts in GitHub, all benchmarking APIs are being implemented in the Summarization_Model_Pipeline.ipynb file.

For summarization model comparison, we use:

Readability (Textstat) is a Python library to compute a variety of readability scores for each summary, including reading ease (0-100, higher is better/easier), US grade level, and years of education. 

Grammar and style (LanguageTool API) is a python library/package which wraps the API. For each summary we count total errors, categorize them (e.g., TYPOS), and compute  error rate (errors per word) to compare the models’ grammatical correctness.

Fluency (GPT‑2 perplexity) uses Hugging Face transformers (GPT2LMHeadModel and GPT2TokenizerFast). Perplexity is computed over each summary with a sliding window to account for the 1024 token limit (in case TextRank summarization exceeds the limit, as BART is restricted to around 400 tokens). Lower perplexity indicates more fluent/natural text.

While our extraction model benchmarking is still in-progress (as of Monday, Feb. 23), we use the following, with clarifications in response to instructor feedback.

**Content overlap (ROUGE)**  
We use the `rouge-score` library to compute ROUGE-1, ROUGE-2, and ROUGE-L between each generated summary and the paper’s abstract (reference). This gives an n-gram overlap benchmark per model.

**Factual consistency (NLI DeBERTa)**  
We use a pre-trained NLI model (e.g. `cross-encoder/nli-deberta-v3-base`) to get three-way labels (entailment, neutral, contradiction) and probabilities for each **(evidence, sentence)** pair.

- **What is NLI used against?**  
  NLI is run on the **retrieved** evidence. For each summary sentence, the retrieval module (BM25 or Sentence-BERT + FAISS) returns one or more passages from the paper. We take that **retrieved paragraph/chunk as the premise** and the **summary sentence as the hypothesis**, and run the NLI model on that pair. So we are measuring whether the specific sentence is entailed, neutral, or contradicting **with respect to the evidence that we actually retrieved and show to the user**. This tells us both whether the sentence is supported by the source and whether retrieval is returning the right part of the document for that sentence.

- **How we define hallucination**  
  We define **hallucination** strictly as when the NLI label is **contradiction** (the retrieved evidence explicitly contradicts the summary sentence). We do **not** count **neutral** as hallucination in our primary metric: we treat neutral as **unverifiable** (the evidence neither supports nor contradicts the claim). Unverifiable can be due to retrieval returning an irrelevant passage or the summary making a claim not stated in that passage; we report it separately (e.g. “supported / unverifiable / contradicted” breakdown and optionally a “strict” rate that counts only contradictions). So: **hallucination rate = # contradictions / # summary sentences**; we may also report an “unverifiable rate” for transparency.

**Validating retrieval correctness**  
We do not yet have a manually annotated gold set. We plan to validate retrieval as follows:

- **Gold set:** For a subset of papers (e.g. 10–20), we will manually annotate, for each summary sentence, which section(s) or passage(s) in the paper are the “correct” evidence (or mark “no correct evidence” if the sentence is not grounded in the document). We will store this as (paper_id, sentence_index, list of correct section IDs or passage spans).
- **Metrics:** For each (sentence, retrieved chunk) pair, we will check whether the retrieved chunk’s section (or passage) matches any of the gold evidence for that sentence. We can report **hit@1** (top-retrieved chunk is correct), **section match** (retrieved chunk’s section is in the gold set), or a binary “correct section retrieved” rate. We will compare BM25 vs. Sentence-BERT + FAISS on this subset to validate which retrieval method returns the right parts of the document more often.
- **Timeline:** We will create this gold subset and run the comparison as part of our Week 9 evaluation work and include the results in the final report.

We will include all benchmarking statistics (including ROUGE, NLI breakdown, and retrieval validation) into our final UI and report. Here is an example of Textstat computed directly from the Jupyter Notebook file.


6. Software
Please note that the columns are independent of each other (i.e. row contents do not correlate).
Publicly Available Software
Code Written Ourselves
GitHub for version control and collaboration
Code for implementing the summarization using BART and TextRank, which were previously provided libraries/implementations (as described in Technical Approach section), but needed to have arguments specified. For TextRank, top-k sentences were specified. For BART, the chunking/recursive approach was developed independently. Saving all summaries to files were also done independently.
Python Streamlit Library for user interface elements.
Code for designing the interface using Streamlit, including dropdown menus, table styling, model linking, and formatted display.
Facebook’s FAISS
Code for implementing text retrieval using BM25, all-MiniLM-L6-v2, and FAISS
Facebook’s BART (large-cnn) for summarization with an advanced LLM.
The data retrieval code was written independently by Richard Youn. The formatting and extraction of data sections/paragraphs into separate, unique JSON files  were written independently as well. 
TextRank (sklearn and networkx libraries) for summarization with an extractive model.
Code for appending paper sections together and mapping sentences to original sections (for later retrieval), and calling of existing sklearn/networkx TF-IDF vectorization, cosine similarity, and graph ranking.
Best Match 25 (BM25) for retrieval with an extractive model.
Code for backend integration into Streamlit frontend UI was written independently, with Jupyter Notebook contents reformatted as Python functions in summarize.py, which were called in app.py.
Benchmarking APIs listed in Experiments and Evaluation for evaluating various metrics across summarization and retrieval models.
Code for integrating all APIs into the pipeline and running them successfully on each model’s results. Formatting code written for displaying/testing in a readable manner.


7. Challenges Identified 
Challenge Faced
How We Addressed it
When loading the UI, the user would have to wait for all models to be loaded, otherwise we wouldn’t be able to generate summaries or do any text retrieval.
To handle this, our TA suggested that we should just precompute the summaries in order to have an interface that could quickly display results. By doing this, we sacrificed our ability to generate variable results to gain speed for the user experience. One more thing we could implement would be generating multiple summaries per paper in order to have some variability.
Streamlit limits the customization of styling, so we couldn’t do our initial plan of allowing users to hover over sentences to make the evidence retrieval appear.
We went with the simpler idea of just using a radio to display each individual summary sentence so that the user could select one to toggle the evidence retrieval.
BART generation was very tricky, especially given the 1024 token limit. Resulting summarizations were, originally, not as descriptive, too descriptive, truncated, or straight-up hallucinations (rare).
First, each section is independently summarized (and truncated in the rare instance that it exceeds 1024 tokens). If the resulting summaries combined are under 1024 tokens, the concatenation is summarized once more (with a limit of 200-400 generated tokens) and returned. If the resulting concatenated summary still exceeds 1024 tokens, we recursively summarize until we reach a token count under 1024, where the summary is summarized once more then returned. 


8. Updated Milestones
End of Week 8: Finish benchmarking/evaluation APIs. Finish BERT + FAISS retrieval model implementation.
End of Week 9: Implement all APIs and retrieval models into UI. Retrieve ~20-100 papers for the final UI workflow/demo.
End of Week 10: Debug if needed, run and store  summarization/extraction for each paper for quick retrieval. 

9. Individual Student Accomplishments  
Maggie: Implemented formatting of retrieved papers by extracting and mapping section titles to paragraphs, TextRank and Facebook BART summarization pipeline, benchmarking/evaluation API implementations, and helped (15%) with building connection from summarization models to UI. 
Lawrence: Implemented front end code using Streamlit, text retrieval algorithms using BM25, all-MiniLM-L6-v2, and FAISS, reformatted ipynb files to .py to connect interface to functions the front end could use. 
 
Our former teammate, Richard Youn, worked on extracting and filtering papers from the S2ORC corpus, specifically Steps 1 and 2 from the Data Sets section of this project report. 
