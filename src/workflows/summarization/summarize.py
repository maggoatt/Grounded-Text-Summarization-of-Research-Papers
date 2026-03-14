# helper functions for generating + saving summaries for the UI, takes code from notebook
# author: maggie zhang

# imports

# TextRank
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import json

# BART
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

# General
from nltk.tokenize import sent_tokenize

model_name = "facebook/bart-large-cnn" # (1)
max_token_count = 1024 # BART's actual positional encoding limit

def extract_body_text_and_section_map(paper):
    # get list of sentences and their section titles from paper fir textrank
    body_text = []
    section_map = {}
    for section in paper["sections"]:
        section_title = section["section_title"]
        sentences = sent_tokenize(section["text"])
        for sentence in sentences:
            if sentence:
                section_map[len(body_text)] = section_title
                body_text.append(sentence)
    return body_text, section_map


def generate_summary_textrank(paper, k=3):
    # TextRank 
    body_text, section_map = extract_body_text_and_section_map(paper)

    if not body_text:
        return ""

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(body_text)
    similarity_mtx = cosine_similarity(X)
    graph = nx.from_numpy_array(similarity_mtx)
    scores = nx.pagerank(graph)

    ranked = sorted(
        ((scores[i], s, section_map[i]) for i, s in enumerate(body_text)),
        reverse=True,
    )
    # normalize: no newlines in summary sentences
    sentences = [s.replace("\n", " ").strip() for _, s, _ in ranked[:k]]
    summary = " ".join(sentences)
    return summary


def setup():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bart_model = BartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, bart_model

def get_token_count(tokenizer, text):
    return len(tokenizer.encode(text, truncation=False))

def summarize(text, bart_model, tokenizer, max_new_tokens=500, min_new_tokens=100):
    # summarize a single chunk of text using BART (input auto-truncated to 1024 tokens)
    inputs = tokenizer(text, return_tensors="pt", max_length=max_token_count, truncation=True)

    summary_ids = bart_model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        num_beams=6,
        length_penalty=0.001, # preference for longer vs shorter outputs
        forced_bos_token_id=0
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def reduce_summaries(texts, tokenizer, bart_model, round_num=1):
    """
    Recursively summarize until the combined text fits within 1024 tokens.
    
    1. Summarize each chunk individually
    2. Concatenate the summaries
    3. If still > 1024 tokens, group into chunks and repeat
    4. Once <= 1024 tokens, produce the final summary
    """
    print(f"--- Round {round_num}: summarizing {len(texts)} chunks ---")
    
    chunk_summaries = []
    for i, text in enumerate(texts):
        tc = get_token_count(tokenizer, text)
        summary = summarize(text, bart_model, tokenizer)
        print(f"  chunk {i+1}/{len(texts)}: {tc} tokens -> {get_token_count(tokenizer, summary)} tokens")
        chunk_summaries.append(summary)
    
    # combine all summaries into one text
    combined = " ".join(chunk_summaries)
    combined_tokens = get_token_count(tokenizer, combined)
    print(f"  combined result: {combined_tokens} tokens")
    
    if combined_tokens <= max_token_count:
        # fits within limit, concatenate and return as-is
        print(f"  fits within {max_token_count} tokens, concatenating summaries")
        return combined
    else:
        # still too long, group summaries into 1024-token chunks and recurse
        print(f"  still > {max_token_count} tokens, splitting again...\n")
        groups = []
        current_group = []
        current_tokens = 0
        for s in chunk_summaries:
            s_tokens = get_token_count(tokenizer, s)
            if current_tokens + s_tokens > max_token_count and current_group:
                groups.append(" ".join(current_group))
                current_group = [s]
                current_tokens = s_tokens
            else:
                current_group.append(s)
                current_tokens += s_tokens
        if current_group:
            groups.append(" ".join(current_group))
        
        return reduce_summaries(groups, tokenizer, bart_model, round_num + 1)

def generate_summary(tokenizer, bart_model, paper):
    body_text = []
    section_map = {}
    for section in paper["sections"]: 
        section_title = section["section_title"]
        sentences = sent_tokenize(section["text"])
    
        for sentence in sentences:
            if sentence:
                section_map[len(body_text)] = section_title  # track section of sentence based on index of sentence
                body_text.append(sentence)

    full_body_text = " ".join(body_text) # (2) turn the list of sentences into string
    token_count = get_token_count(tokenizer, full_body_text)
    print(f"total tokens: {token_count}\nmax allowed tokens: {max_token_count}\n")

    if token_count > max_token_count:
        # step 1: summarize each section individually (preserves 1:1 mapping with section titles)
        section_texts = [section["text"] for section in paper["sections"]]
        summaries = []
        print(f"--- Summarizing {len(section_texts)} sections ---")
        for i, text in enumerate(section_texts):
            tc = get_token_count(tokenizer, text)
            s = summarize(text, bart_model, tokenizer)
            print(f"  section {i+1}/{len(section_texts)}: {tc} tokens -> {get_token_count(tokenizer, s)} tokens")
            summaries.append(s)
        
        # step 2: combine section summaries and reduce until it fits in 1024 tokens
        combined = " ".join(summaries)
        combined_tokens = get_token_count(tokenizer, combined)
        print(f"\ncombined section summaries: {combined_tokens} tokens")
        
        if combined_tokens <= max_token_count:
            print(f"combined tokens less than limit, summarize once then return...\n")
            inputs = tokenizer(combined, return_tensors="pt", max_length=max_token_count, truncation=True)
            summary_ids = bart_model.generate(
                inputs["input_ids"],
                max_new_tokens=400,
                min_new_tokens=200,
                num_beams=6,
                length_penalty=0.6, # preference for longer vs shorter outputs
                forced_bos_token_id=0
            )
            summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        else:
            # need more reduction rounds
            print(f"  still > {max_token_count} tokens, entering reduction loop...\n")
            groups = []
            current_group = []
            current_tokens = 0
            for s in summaries:
                s_tokens = get_token_count(tokenizer, s)
                if current_tokens + s_tokens > max_token_count and current_group:
                    groups.append(" ".join(current_group))
                    current_group = [s]
                    current_tokens = s_tokens
                else:
                    current_group.append(s)
                    current_tokens += s_tokens
            if current_group:
                groups.append(" ".join(current_group))
            
            summary_text = reduce_summaries(groups, tokenizer, bart_model, round_num=2)
    else:
        # small enough to summarize directly
        summary_text = summarize(full_body_text, bart_model, tokenizer)
        summaries = [summary_text]

    return summary_text 
    print(f"\n{'='*80}")
    print("FINAL SUMMARY:")
    print(f"{'='*80}")
    print(summary_text)