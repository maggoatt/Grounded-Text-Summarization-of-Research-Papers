import streamlit as st
from pathlib import Path
import json
import summarization.summarize as sum
import re
import bm_retrieval as bm_ret
import faiss_retrieval as faiss_ret


st.set_page_config(page_title="Grounded Text Summarization of Research Papers", layout="wide")


st.markdown(
    "<h1 style='text-align: center;'> Grounded Text Summarization of Research Papers </h1>",
    unsafe_allow_html=True
)


papers_dir = Path("data")
summaries_dir = Path("summaries")

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

    if "bm25" not in st.session_state or st.session_state.get("bm25_paper_id") != chosen_id:
        st.session_state.bm25, st.session_state.section_info = bm_ret.build_bm25(paper)
        st.session_state.bm25_paper_id = chosen_id

    if "faiss_index" not in st.session_state or st.session_state.get("faiss_paper_id") != chosen_id:
        st.session_state.faiss_index, st.session_state.faiss_metadata = faiss_ret.load_paper_embeddings(chosen_id)
        st.session_state.faiss_paper_id = chosen_id

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

    model = st.selectbox("Summarization model", ["TextRank", "Sentence Bartholmeow"])

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
            st.subheader("Evidence Retrieval")

            model = st.selectbox("Retrieval model", ["BestMatch25", "all-mini"])

            if st.session_state.chosen_sentence:
                st.divider()
                st.write("**Evidence from paper:**")

                if model == "BestMatch25":
                    results = bm_ret.find_evidence(
                        st.session_state.chosen_sentence,
                        st.session_state.bm25,
                        st.session_state.section_info,
                        top_k=3
                    )
                    for i, r in enumerate(results):
                        with st.expander(f"#{i+1} — From Section '{r['section_title']}' (Score: {r['score']:.2f})"):
                            st.write(r['snippet'] + "...")

                elif model == "all-mini":
                    results = faiss_ret.find_evidence_faiss(
                        st.session_state.chosen_sentence,
                        st.session_state.faiss_index,
                        st.session_state.faiss_metadata,
                        top_k=3
                    )
                    for i, r in enumerate(results):
                        with st.expander(f"#{i+1} — (Score: {r['score']:.2f})"):
                            st.write(r['chunk'])
            st.write("**Selected:**", chosen)

            st.write("**Metrics (placeholder):**")
            st.write("- Confidence: 0.82")
            st.write("- Hallucination risk: Low")
        
