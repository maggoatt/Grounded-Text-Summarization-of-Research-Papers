# generates and saves metrics (readability, perplexity, etc.) for evaluating TextRank and BART summaries for UI use
# authors: lawrence zhou and maggie zhang

from pathlib import Path
import json
import csv

from summarization import metrics as m


data_dir = Path("../../../data")
summaries_dir = Path("../../../summaries")
metrics_dir = Path("../../../metrics")

metrics_dir.mkdir(exist_ok=True)


def compute_and_save_metrics():
    # iterate through all papers and save metrics per paper
    for paper_path in data_dir.glob("*.json"):
        with open(paper_path, "r", encoding="utf-8") as f:
            paper = json.load(f)

        cid = paper.get("corpusid") or paper_path.stem

        rows = []
        for model_name, suffix in [("textrank", "_textrank_summary.txt"), ("bart", "_bart_summary.txt")]:
            summary_path = summaries_dir / f"{cid}{suffix}"
            if not summary_path.exists():
                continue

            summary_text = summary_path.read_text(encoding="utf-8")
            metric_values = m.compute_all_metrics(summary_text)

            readability = metric_values.get("readability") or {}
            grammar = metric_values.get("grammar") or {}
            perplexity = metric_values.get("perplexity")

            rows.append(
                {
                    "model": model_name,
                    "flesch_reading_ease": readability.get("Flesch Reading Ease"),
                    "flesch_kincaid_grade": readability.get("Flesch-Kincaid Grade"),
                    "gunning_fog_index": readability.get("Gunning Fog Index"),
                    "smog_index": readability.get("SMOG Index"),
                    "dale_chall_score": readability.get("Dale-Chall Score"),
                    "total_errors": grammar.get("total_errors"),
                    "error_rate": grammar.get("error_rate"),
                    "perplexity": perplexity,
                }
            )

        if not rows:
            continue

        out_path = metrics_dir / f"{cid}_metrics.tsv"
        fieldnames = [
            "model",
            "flesch_reading_ease",
            "flesch_kincaid_grade",
            "gunning_fog_index",
            "smog_index",
            "dale_chall_score",
            "total_errors",
            "error_rate",
            "perplexity",
        ]
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    compute_and_save_metrics()

