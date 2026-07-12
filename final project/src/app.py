import streamlit as st
from rag import InstiAssist

st.set_page_config(page_title="IITB Insti-Assist", page_icon="🎓", layout="centered")


@st.cache_resource(show_spinner="Loading knowledge base and embedding model...")
def load_assistant():
    return InstiAssist()


GROUNDEDNESS_COLOR = {
    "Strongly grounded": "🟢",
    "Weakly grounded": "🟡",
    "Not grounded": "🔴",
}


def render_highlighted(text: str):
    parts = text.split("[[")
    st.markdown(parts[0].replace("]]", ""), unsafe_allow_html=False)
    for part in parts[1:]:
        if "]]" in part:
            highlighted, rest = part.split("]]", 1)
            st.markdown(f":orange-background[{highlighted}]  \n{rest}")
        else:
            st.markdown(part)


st.title("🎓 IITB Insti-Assist — Academic Assistant")
st.caption(
    "Answers questions about the WnCC Machine Learning Learner's Space 2026 syllabus, "
    "weekly topics, prerequisites, and assignments. Fully local retrieval — no external "
    "API calls. Answers are extracted directly from the course material, with sources "
    "and a groundedness indicator shown below every answer."
)

try:
    assistant = load_assistant()
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Run `python src/ingest.py` from the project root first, then reload this page.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("About this assistant")
    st.markdown(
        "**Scope:** Academic Assistant — WnCC ML Learner's Space syllabus\n\n"
        "**Knowledge base:** Week 1-4 READMEs and assignment briefs only\n\n"
        "**Pipeline:** chunk -> embed (MiniLM) -> FAISS retrieval -> "
        "extractive grounded answer (no external LLM API)\n\n"
        "If the answer isn't in the course material, the assistant will say "
        "\"I don't know\" instead of guessing."
    )
    st.markdown(
        "**Groundedness legend:**\n\n"
        "🟢 Strongly grounded — high-similarity match found\n\n"
        "🟡 Weakly grounded — only a loose match found\n\n"
        "🔴 Not grounded — nothing relevant found, answer withheld"
    )
    top_k = st.slider("Chunks to retrieve (k)", min_value=2, max_value=8, value=4)
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()


def render_turn(turn):
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        badge = GROUNDEDNESS_COLOR.get(turn["groundedness"]["label"], "")
        st.markdown(f"{badge} **{turn['groundedness']['label']}** (top similarity `{turn['groundedness']['top_score']:.3f}`)")
        st.markdown(turn["answer"])
        with st.expander(f"📎 Sources ({len(turn['sources'])})"):
            for s in turn["sources"]:
                st.markdown(f"**{s['title']}** — similarity `{s['score']:.3f}`")
                if s.get("source_url"):
                    st.markdown(f"[View original source]({s['source_url']})")
                highlighted = assistant.highlight_source(s, turn["used_sentences"])
                render_highlighted(highlighted[:500] + ("..." if len(highlighted) > 500 else ""))
                st.divider()


for turn in st.session_state.history:
    render_turn(turn)

question = st.chat_input("Ask about the syllabus, a week, topic, or assignment...")

if question:
    with st.spinner("Retrieving relevant course material..."):
        result = assistant.rag_answer(question, k=top_k)
    render_turn(result)
    st.session_state.history.append(result)
