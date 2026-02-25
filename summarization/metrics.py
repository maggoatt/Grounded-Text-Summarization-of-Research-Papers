# metrics for evaluating generated summaries

import textstat
import language_tool_python
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def readability_scores(text):
    # readability metrics using textstat
    scores = {
        "Flesch Reading Ease": textstat.flesch_reading_ease(text),
        "Flesch-Kincaid Grade": textstat.flesch_kincaid_grade(text),
        "Gunning Fog Index": textstat.gunning_fog(text),
        "SMOG Index": textstat.smog_index(text),
        "Dale-Chall Score": textstat.dale_chall_readability_score(text),
    }
    return scores


def grammar_check(text):
    # grammar/style issues using LanguageTool (simplified return)
    tool = language_tool_python.LanguageTool("en-US")
    matches = tool.check(text)
    word_count = len(text.split())

    categories = {}
    for m in matches:
        cat = m.category
        categories[cat] = categories.get(cat, 0) + 1

    error_rate = len(matches) / word_count if word_count > 0 else 0

    # include a few example issues for potential debugging/inspection
    examples = []
    for m in matches[:3]:
        examples.append({
                "context": m.context,
                "message": m.message,
            })

    return {
        "total_errors": len(matches),
        "word_count": word_count,
        "error_rate": error_rate,
        "categories": categories,
        "examples": examples,
    }


def compute_perplexity(text):
    # perplexity using GPT-2 with a sliding window for long texts
    tokenizer =  GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    encodings = tokenizer(text, return_tensors="pt")
    max_length = model.config.n_positions  # 1024 for GPT-2
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end  # tokens actually scored in this window

        input_ids = encodings.input_ids[:, begin:end]
        target_ids = input_ids.clone()
        # mask out tokens already scored (overlap)
        target_ids[:, :-target_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood * target_len)
        prev_end = end
        if end == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end).item()
    return ppl


def compute_all_metrics(text):
    # convenience aggregator used by the UI
    if not text or not text.strip():
        return {"readability": None, "grammar": None, "perplexity": None}

    return {
        "readability": readability_scores(text),
        "grammar": grammar_check(text),
        "perplexity": compute_perplexity(text),
    }

