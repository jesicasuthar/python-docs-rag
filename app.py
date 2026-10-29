
import streamlit as st
from rag import RAGAssistant
from logger import log_interaction

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Python Documentation Assistant",
    page_icon="🐍",
    layout="wide"
)


# ==========================================
# CUSTOM STYLING
# ==========================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .main-title {
            font-size: 2.4rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #777;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .source-card {
            padding: 12px 16px;
            border: 1px solid rgba(128, 128, 128, 0.3);
            border-radius: 10px;
            margin-bottom: 10px;
        }

        .source-label {
            font-weight: 700;
        }

        div[data-testid="stExpander"] {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# LOAD RAG ASSISTANT ONCE
# ==========================================

@st.cache_resource(show_spinner=False)
def load_assistant():
    return RAGAssistant()


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.title("🐍 Python RAG")

    st.markdown(
        """
        Ask questions about Python and receive answers
        grounded in your indexed documentation.
        """
    )

    st.divider()

    st.subheader("How it works")

    st.markdown(
        """
        1. Enter a Python question.
        2. Search relevant documentation.
        3. Generate an answer using the local LLM.
        4. Inspect citations and retrieved context.
        """
    )

    st.divider()

    st.caption("Embedding model")
    st.code("all-MiniLM-L6-v2")

    st.caption("Language model")
    st.code("Qwen2.5-0.5B-Instruct")

    st.caption("Vector database")
    st.code("ChromaDB")

    if st.button("Clear current answer", use_container_width=True):
        st.session_state.pop("last_question", None)
        st.session_state.pop("last_answer", None)
        st.session_state.pop("last_sources", None)
        st.session_state.pop("last_context", None)
        st.rerun()


# ==========================================
# MAIN HEADER
# ==========================================

st.markdown(
    '<div class="main-title">Python Documentation Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask a question. Get a simple answer backed by Python documentation.'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "This assistant uses a local language model and your existing "
    "ChromaDB index. The first load may take a little time."
)


# ==========================================
# QUESTION INPUT
# ==========================================

with st.form("question_form"):
    question = st.text_input(
        "Your question",
        placeholder="e.g. How do I read a CSV file in Python?",
        max_chars=1000
    )

    submitted = st.form_submit_button(
        "🔎 Ask question",
        use_container_width=True,
        type="primary"
    )


# ==========================================
# ==========================================
# PROCESS QUESTION
# ==========================================

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Loading your RAG assistant..."):
                assistant = load_assistant()

            with st.spinner("Searching documentation..."):
                chunks = assistant.retrieve(
                    question.strip(),
                    top_k=3
                )

            if not chunks:
                st.warning(
                    "No relevant documentation was found for this question."
                )

                st.session_state.pop("last_answer", None)
                st.session_state.pop("last_sources", None)
                st.session_state.pop("last_context", None)
                st.session_state.pop("last_question", None)

            else:
                with st.spinner("Generating your answer..."):
                    answer = assistant.generate_answer(
                        question.strip(),
                        chunks
                    )

                context = assistant.build_context(chunks)

                # Log the interaction to JSONL
                try:
                    log_interaction(
                        question=question.strip(),
                        chunks=chunks,
                        answer=answer
                    )
                except Exception as log_error:
                    st.warning(
                        f"Answer generated, but logging failed: {log_error}"
                    )

                # Save the results for display
                st.session_state["last_question"] = question.strip()
                st.session_state["last_answer"] = answer
                st.session_state["last_sources"] = chunks
                st.session_state["last_context"] = context

        except Exception as error:
            st.error(f"Something went wrong: {error}")

            with st.expander("Technical error details"):
                st.exception(error)
# ==========================================
# DISPLAY GENERATED ANSWER
# ==========================================

if "last_answer" in st.session_state:

    st.divider()

    st.subheader("💬 Your question")
    st.write(st.session_state["last_question"])

    st.subheader("✨ Generated answer")

    st.markdown(st.session_state["last_answer"])

    st.caption(
        "Answers are generated locally and may contain errors. "
        "Check the retrieved sources to verify important details."
    )

    # ======================================
    # SOURCE LIST
    # ======================================

    st.divider()
    st.subheader("📚 Retrieved sources")

    sources = st.session_state["last_sources"]

    for chunk in sources:

        label = chunk["label"]
        source_path = chunk["source"]
        distance = chunk.get("distance")

        with st.container(border=True):

            st.markdown(f"**[{label}]**")

            st.code(source_path, language=None)

            if distance is not None:
                st.caption(
                    f"ChromaDB distance: {distance:.4f} "
                    "(lower generally means closer)"
                )

    # ======================================
    # EXPANDABLE RETRIEVED CONTEXT
    # ======================================

    st.divider()
    st.subheader("🔍 Retrieved context")

    st.caption(
        "These are the documentation excerpts passed to the language model."
    )

    for chunk in sources:

        with st.expander(
            f"[{chunk['label']}] — {chunk['source']}"
        ):
            st.markdown("**Retrieved documentation**")

            st.text(chunk["text"])

            st.caption(
                f"Chunk ID: {chunk['id']}"
            )

            if chunk.get("distance") is not None:
                st.caption(
                    f"Retrieval distance: {chunk['distance']:.4f}"
                )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Python Documentation RAG · ChromaDB · Transformers · Streamlit"
)