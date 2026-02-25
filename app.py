import streamlit as st
from pathlib import Path
import json
import csv
import summarization.summarize as sum
import re


st.set_page_config(page_title="Grounded Text Summarization of Research Papers", layout="wide")


st.markdown(
    "<h1 style='text-align: center;'> Grounded Text Summarization of Research Papers </h1>",
    unsafe_allow_html=True
)


papers_dir = Path("data")
summaries_dir = Path("summaries")
metrics_dir = Path("metrics")

@st.cache_data
def create_index(papers_dir):
    """ Loads all the papers title and ids for the dropdown """
    paper_index = {}
    for file in sorted(papers_dir.glob("*.json")):
        paper_id = file.stem
        paper = json.loads(file.read_text(encoding="utf-8"))
        title = paper.get("title")
        paper_index[paper_id] = {
            "title": paper.get("title", paper_id),
            "authors": paper.get("authors", []),
            "year": paper.get("year", None),
            "subject": paper.get("subject", None),
            "path": str(file)
        }
    return paper_index

def generate_preview(paper: dict, max_chars: int = 800, max_sections: int = 2):
    sections = paper.get("sections")

    chars_used = 0
    text_list = []
    if not sections:
        return "No sections found"
    
    for section in sections:
        text = section.get("text")
        text = " ".join(text.split())

        chars_left = max_chars - chars_used
        if (chars_left <= 0):
            break
        if len(text) > chars_left:
            text = text[:chars_left] + "..."
        text_list.append(text)
        chars_used += len(text)
    return "\n\n".join(text_list)

def load_single_paper(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

paper_index = create_index(papers_dir)

def split_into_sentences(text: str):
    # simple sentence split (good enough for UI; you can improve later)
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return sents


def load_metrics_for_model(paper_id: str, model_display_name: str):
    # load precomputed metrics for the given paper and model from metrics dir
    metrics_path = metrics_dir / f"{paper_id}_metrics.tsv"
    if not metrics_path.exists():
        return None

    model_key = "textrank" if model_display_name == "TextRank" else "bart"

    with metrics_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("model") != model_key:
                continue

            # parse floats safely
            def to_float(value):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            readability = {
                "Flesch Reading Ease": to_float(row.get("flesch_reading_ease")),
                "Flesch-Kincaid Grade": to_float(row.get("flesch_kincaid_grade")),
            }
            grammar = {
                "total_errors": int(row.get("total_errors")) if row.get("total_errors") not in (None, "", "None") else 0,
                "error_rate": to_float(row.get("error_rate")) or 0.0,
            }
            perplexity = to_float(row.get("perplexity"))

            return {
                "readability": readability,
                "grammar": grammar,
                "perplexity": perplexity,
            }

    return None


left, right = st.columns([1, 2], gap="large")

st.markdown("""
<style>
.container {
    border: 1px solid blue;
    border-radius: 8px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

with left:
    st.subheader("Research Paper Selection")

    paper_ids = list(paper_index.keys())

    chosen_id = st.selectbox(
        "Choose a Research Paper",
        paper_ids,
        format_func=lambda paper_id: paper_index[paper_id]["title"],
        key="chosen_paper_id"
    )

    chosen = paper_index[chosen_id]
    paper = load_single_paper(chosen["path"])

    preview = generate_preview(paper, max_chars=800, max_sections=2)
    authors = chosen["authors"]
    authors_str = ", ".join(authors)

    st.markdown(
        f"""
        <div class="container">
            <h4><b>{chosen['title']}</b></h4>
            <div>Author(s): {authors_str}</div>
            <br/>
            <div>{preview}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    # <div>Subject: {chosen['subject']}</div>
    # <div>Year: {chosen['year']}</div>

if "summary_sentences" not in st.session_state:
    st.session_state.summary_sentences = None
if "chosen_sentence" not in st.session_state:
    st.session_state.chosen_sentence = None
if "full_summary_text" not in st.session_state:
    st.session_state.full_summary_text = None

with right:
    st.subheader("Summarization")

    model = st.selectbox("Summarization model", ["TextRank", "Facebook BART"])

    if st.button("Generate Summary", type="primary"):
        if model == "TextRank":
            summary_path = summaries_dir / f"{chosen_id}_textrank_summary.txt"
        else:
            summary_path = summaries_dir / f"{chosen_id}_bart_summary.txt"

        if summary_path.exists():
            summary_text = summary_path.read_text(encoding="utf-8")
            st.session_state.full_summary_text = summary_text
            st.session_state.summary_sentences = split_into_sentences(summary_text)
        else:
            st.session_state.summary_sentences = ["(No summary produced.)"]
            st.session_state.chosen_sentence = st.session_state.summary_sentences[0]
            st.session_state.sentence_radio = st.session_state.chosen_sentence

    # Always show full summary when we have one (persists when user picks sentences)
    if st.session_state.full_summary_text:
        with st.container(border=True):
            st.write(st.session_state.full_summary_text)

    if st.session_state.summary_sentences is None:
        st.markdown(
            """
            <div class="container">
            <div>Generated summary will appear here.</div>
            <br/>
            <div>• Sentence 1 (placeholder)</div>
            <div>• Sentence 2 (placeholder)</div>
            <div>• Sentence 3 (placeholder)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # temporary
        with st.container(border=True):
            st.write("**Summary:**")

            chosen = st.radio(
                "Select a sentence",
                st.session_state.summary_sentences,
                key="sentence_radio",
                label_visibility="collapsed"
            )
            st.session_state.chosen_sentence = chosen
            st.write("**Selected:**", chosen)

            st.write("**Metrics:**")
            metric_values = load_metrics_for_model(chosen_id, model)

            if not metric_values:
                st.write("Metrics will be available after a summary is generated.")
            else:
                readability = metric_values.get("readability")
                grammar = metric_values.get("grammar")
                perplexity = metric_values.get("perplexity")

                if readability:
                    # simplified, user-facing subset of readability metrics
                    fre = readability.get("Flesch Reading Ease")
                    fk = readability.get("Flesch-Kincaid Grade")
                    if fre is not None:
                        st.write(f"- Flesch Reading Ease: {fre:.1f} (higher = easier to read; typical news articles are around 60–70).")
                    if fk is not None:
                        st.write(f"- Flesch-Kincaid Grade: {fk:.1f} (approximate U.S. school grade level needed to understand the summary).")

                if grammar:
                    total_errors = grammar.get("total_errors", 0)
                    error_rate = grammar.get("error_rate", 0.0) * 100
                    st.write(f"- Grammar/style issues: {total_errors} total ({error_rate:.1f} per 100 words; lower = fewer problems).")

                if perplexity is not None:
                    st.write(f"- Fluency (GPT-2 perplexity): {perplexity:.1f} (lower = text is more predictable and natural to a language model).")
