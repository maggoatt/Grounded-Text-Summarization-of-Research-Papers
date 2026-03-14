# Quick Guide to the Data Processing Pipeline
To prepare papers from the S2ORC corpus for analysis and summarization, please run the scripts in the ```data_processing``` directory as follows:

1. ```filter_cs_papers.py```: selects latest publications with computer science-related keywords and saves them to ```data/cs_papers.jsonl```.
2. ```clean_existing_papers.py```: removes S2ORC fields unrelated to this project (e.g., bibliography, figures, formulas) and saves to ```data/papers_cleaned.jsonl```.
3. ```format_cleaned_papers.py```: translates character-offset S2ORC annotations into parsable sections and saves each paper (identified via Corpus ID) as an individual ```.json``` file in ```data/```.

All ```.json``` and ```.jsonl``` files should be located locally in your ```data/``` directory, with individual papers labeled as their S2ORC Corpus ID.