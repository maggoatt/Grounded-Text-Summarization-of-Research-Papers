from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from pathlib import Path
from bm_retrieval import find_best_chunk, tokenize

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings_dir="paper_embeddings"

def load_paper_embeddings(paper_id):
    # basically the creation of optimized index for searching
    embeddings = np.load(f"{embeddings_dir}/{paper_id}_embeddings.npy").astype("float32")
    with open(f"{embeddings_dir}/{paper_id}_metadata.json") as f:
        metadata = json.load(f)
    
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    return index, metadata

def find_evidence_faiss(summary_sentence: str, index, metadata, top_k=3):
    # taking the query and encoding and comparing it to the embeddings
    query = model.encode([summary_sentence]).astype("float32")
    faiss.normalize_L2(query)
    scores, indices = index.search(query, top_k)

    query_tokens = set(tokenize(summary_sentence))
    
    results = []
    for score, i in zip(scores[0], indices[0]):
        chunk = metadata["chunks"][i]
        chunk = find_best_chunk(chunk, query_tokens, window = 1)
        results.append({
            "chunk": chunk,
            "score": float(score)
        })
    return results