from rank_bm25 import BM25Okapi
import json, re
import nltk
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

def tokenize(s: str):
    return re.findall(r"[a-z0-9]+", s.lower())

def build_bm25(paper: dict):
    section_info = []
    raw_texts = []

    for section in paper["sections"]:
        raw_text = f'{section["section_title"]} {section["text"]}'
        section_info.append({
            "corpusId": paper["corpusid"],
            "title": paper["title"],
            "section_title": section["section_title"],
            "text": section["text"]
        })
        raw_texts.append(raw_text)

    tokenized_texts = [tokenize(t) for t in raw_texts]
    bm25 = BM25Okapi(tokenized_texts)
    return bm25, section_info

def find_best_chunk(text, query_tokens, window = 1):
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    sentences = sent_tokenize(text)
    best_index = 0
    best_score = 0
    for i, sent in enumerate(sentences):
        matches = sum(1 for t in tokenize(sent) if t in query_tokens)
        if matches > best_score:
            best_score = matches
            best_idx = i
    
    # grab a window of sentences around the best one
    start = max(0, best_idx - window)
    end = min(len(sentences), best_idx + window + 1)
    return " ".join(sentences[start:end])

def find_evidence(summary_sentence: str, bm25, section_info, top_k=3):
    scores = bm25.get_scores(tokenize(summary_sentence))
    top_idxs = scores.argsort()[::-1][:top_k]
    query_tokens = set(tokenize(summary_sentence))
    
    results = []
    for i in top_idxs:
        section = section_info[int(i)]
        results.append({
            "section_title": section["section_title"],
            "score": float(scores[i]),
            "snippet": find_best_chunk(section["text"], query_tokens),
            "query_tokens": query_tokens
        })
    return results