# saves all resulting retrievals for all papers to be used in UI and for metric evaluations
# author: lawrence zhou

import json, re
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from .bm_retrieval import tokenize
from nltk.tokenize import sent_tokenize

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings_dir = Path("../../../paper_embeddings")
summaries_dir = Path("../../../summaries")
data_dir = Path("../../../data")
output_dir = Path("../../../retrievals")
output_dir.mkdir(exist_ok=True)

PAPER_IDS = [
    "202734553", "253098895", "263242894", "267783201",
    "268701241", "269705254", "270157212", "273549360",
    "281843350", "282251113"
]

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def load_faiss(paper_id):
    embeddings = np.load(embeddings_dir / f"{paper_id}_embeddings.npy").astype("float32")
    with open(embeddings_dir / f"{paper_id}_metadata.json") as f:
        metadata = json.load(f)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, metadata

def find_section(chunk_text, paper):
    for section in paper["sections"]:
        if chunk_text[:80] in section["text"]:
            return section["section_title"]
    return "Unknown"

def get_top_sentences_from_chunk(chunk, query_vec, top_n=3):
    """Score each sentence in the chunk by cosine sim to query, return top_n."""
    sentences = sent_tokenize(chunk)
    if not sentences:
        return []
    sent_embeddings = model.encode(sentences).astype("float32")
    faiss.normalize_L2(sent_embeddings)
    # dot product = cosine sim since both are normalized
    sims = sent_embeddings @ query_vec.T  # shape (n_sents,)
    ranked = sorted(zip(sims.flatten(), sentences), reverse=True)
    return [s for _, s in ranked[:top_n]]

for paper_id in PAPER_IDS:
    summary_path = summaries_dir / f"{paper_id}_bart_summary.txt"
    if not summary_path.exists():
        print(f"No BART summary for {paper_id}, skipping")
        continue

    paper = json.loads((data_dir / f"{paper_id}.json").read_text(encoding="utf-8"))
    corpus_id = paper.get("corpusid", paper_id)
    summary_text = summary_path.read_text(encoding="utf-8")
    sentences = split_sentences(summary_text)

    index, metadata = load_faiss(paper_id)

    for sent_num, sentence in enumerate(sentences, start=1):
        query_vec = model.encode([sentence]).astype("float32")
        faiss.normalize_L2(query_vec)

        scores, indices = index.search(query_vec, 3)

        result = {}
        for score, i in zip(scores[0], indices[0]):
            raw_chunk = metadata["chunks"][i]
            section_title = find_section(raw_chunk, paper)
            top_sents = get_top_sentences_from_chunk(raw_chunk, query_vec, top_n=1)
            if top_sents:
                if section_title not in result:
                    result[section_title] = []
                result[section_title].append(top_sents[0])

        out_path = output_dir / f"{corpus_id}_faiss_retrieval_{sent_num}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved {out_path.name}")

print("Done.")