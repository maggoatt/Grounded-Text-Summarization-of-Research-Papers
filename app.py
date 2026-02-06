import streamlit as st


st.set_page_config(page_title="Grounded Text Summarization of Research Papers", layout="wide")


st.markdown(
    "<h1 style='text-align: center;'> Grounded Text Summarization of Research Papers </h1>",
    unsafe_allow_html=True
)

# fake data

PAPERS = [
    {
        "title": "Lawrence is so awesome",
        "authors": "Law",
        "year": 2005,
        "subject": "Computer Science",
        "preview": "Wow law was born and changed the world and everything was so cool"
    },
    {
        "title": "CRISPR Screening in Cancer Research",
        "authors": "C. Researcher, D. Scientist",
        "year": 2022,
        "subject": "Biology",
        "preview": "We perform genome-wide CRISPR screens to identify essential pathways in tumor proliferation. Results highlight key regulators in..."
    },
    {
        "title": "Attention Mechanisms in Vision Transformers",
        "authors": "E. Student, F. Prof",
        "year": 2021,
        "subject": "Computer Science",
        "preview": "Vision Transformers rely on self-attention to model global context. We analyze attention maps and find that..."
    },
]

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

    titles = [paper["title"] for paper in PAPERS]

    chosen_title = st.selectbox(
        "Choose a Research Paper",
        titles
    )

    chosen = None
    # this is fine for now, but prolly build an index when all papers added
    for paper in PAPERS:
        if paper["title"] == chosen_title:
            chosen = paper
            break

    st.markdown(
        f"""
        <div class="container">
            <div><b>{chosen['title']}</b></div>
            <div>{chosen['authors']}</div>
            <div>{chosen['subject']}</div>
            <div>{chosen['year']}</div>
            <br/>
            <div>{chosen['preview']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

if "summary_sentences" not in st.session_state:
    st.session_state.summary_sentences = None
if "chosen_sentence" not in st.session_state:
    st.session_state.chosen_sentence = None

with right:
    st.subheader("Summarization")

    model = st.selectbox("Summarization model", ["TextRank", "Sentence Bartholmeow"])

    if st.button("Generate Summary", type="primary"):
        st.session_state.summary_sentences = [
            "The paper evaluates hallucination using a NIL-based method.",
            "Results show improved factual consistency with the proposed approach.",
            "The method generalizes across multiple benchmarks."
        ]
        st.session_state.chosen_sentence = st.session_state.summary_sentences[0]
        st.session_state.sentence_radio = st.session_state.chosen_sentence


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

            st.write("**Metrics (placeholder):**")
            st.write("- Confidence: 0.82")
            st.write("- Hallucination risk: Low")
