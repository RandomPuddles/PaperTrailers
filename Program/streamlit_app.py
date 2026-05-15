"""
streamlit_app.py

Streamlit UI for the Gemini-powered AI Meeting Task + Research Assistant.

Required files in the same folder:
- Backend.py
- streamlit_app.py
- .env

Run with:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from Backend import ResearchAssistantBackend, SAMPLE_TRANSCRIPT


# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="AI Meeting Research Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Meeting Task + Research Assistant")

st.write(
    "Paste or upload a meeting transcript. The app will extract tasks, assign them "
    "to team members, recommend arXiv research papers, and export everything to Excel."
)


# ============================================================
# Helper for Streamlit secrets
# ============================================================

def get_streamlit_secret(key: str) -> Optional[str]:
    try:
        return st.secrets.get(key)
    except Exception:
        return None


# ============================================================
# Session state setup
# ============================================================

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "result" not in st.session_state:
    st.session_state.result = None

if "last_error" not in st.session_state:
    st.session_state.last_error = None


# ============================================================
# Cached backend constructor
# ============================================================

@st.cache_resource(show_spinner=False)
def get_backend(
    api_key: Optional[str],
    arxiv_path: Optional[str],
    model: Optional[str],
) -> ResearchAssistantBackend:
    """
    Cached so Streamlit does not recreate the backend on every rerun.

    The backend may download/cache the Kaggle dataset the first time it runs.
    """

    return ResearchAssistantBackend(
        api_key=api_key or None,
        arxiv_jsonl_path=arxiv_path or None,
        model=model or None,
    )


# ============================================================
# Sidebar settings
# ============================================================

with st.sidebar:
    st.header("Settings")

    st.caption(
        "You can either put your Gemini API key in a `.env` file or paste it here."
    )

    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Leave blank if using .env",
    )

    secret_api_key = get_streamlit_secret("GEMINI_API_KEY")

    model_input = st.text_input(
        "Gemini Model",
        value="gemini-2.5-flash",
        help="This should match a model available to your Gemini API account.",
    )

    secret_model = get_streamlit_secret("GEMINI_MODEL")

    st.divider()

    arxiv_path_input = st.text_input(
        "Optional Local arXiv Dataset Path",
        placeholder=r"C:\path\to\arxiv-metadata-oai-snapshot.json",
        help=(
            "Leave this blank to let Backend.py download the Kaggle arXiv dataset "
            "using kagglehub."
        ),
    )

    top_n = st.number_input(
        "Papers per Person",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
    )

    scan_mode = st.selectbox(
        "Dataset Search Size",
        options=[
            "Fast demo: first 100,000 records",
            "Better search: first 500,000 records",
            "Full dataset: all records",
        ],
        index=0,
    )

    if scan_mode.startswith("Fast"):
        max_records = 100_000
    elif scan_mode.startswith("Better"):
        max_records = 500_000
    else:
        max_records = None

    st.caption(
        "The full dataset search may be slower, especially the first time."
    )


# ============================================================
# Transcript input section
# ============================================================

st.subheader("1. Enter Meeting Transcript")

uploaded_file = st.file_uploader(
    "Upload a transcript file",
    type=["txt"],
)

if uploaded_file is not None:
    try:
        uploaded_text = uploaded_file.read().decode("utf-8", errors="replace")
        st.session_state.transcript = uploaded_text
        st.success("Transcript loaded from uploaded file.")
    except Exception as error:
        st.error(f"Could not read uploaded file: {error}")

col_sample, col_clear = st.columns([1, 1])

with col_sample:
    if st.button("Use Sample Transcript"):
        st.session_state.transcript = SAMPLE_TRANSCRIPT
        st.session_state.result = None
        st.session_state.last_error = None

with col_clear:
    if st.button("Clear Transcript"):
        st.session_state.transcript = ""
        st.session_state.result = None
        st.session_state.last_error = None

transcript = st.text_area(
    "Transcript",
    value=st.session_state.transcript,
    height=280,
    placeholder=(
        "Example:\n"
        "Ian: I can build the interface.\n"
        "Maya: I’ll research the drag force equations.\n"
        "Jordan: I can code the simulation logic."
    ),
)

st.session_state.transcript = transcript


# ============================================================
# Run analysis
# ============================================================

st.subheader("2. Run Analysis")

run_button = st.button(
    "Analyze Transcript and Create Excel File",
    type="primary",
)

if run_button:
    st.session_state.result = None
    st.session_state.last_error = None

    if not transcript.strip():
        st.warning("Please enter or upload a transcript first.")
    else:
        progress_box = st.empty()

        def show_progress(message: str) -> None:
            progress_box.info(message)

        try:
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_excel_path = output_dir / f"meeting_research_output_{timestamp}.xlsx"

            final_api_key = api_key_input.strip() or secret_api_key
            final_model = model_input.strip() or secret_model or "gemini-2.5-flash"
            final_arxiv_path = arxiv_path_input.strip() or None

            with st.spinner("Working..."):
                backend = get_backend(
                    api_key=final_api_key,
                    arxiv_path=final_arxiv_path,
                    model=final_model,
                )

                result = backend.process_transcript(
                    transcript=transcript,
                    output_excel_path=str(output_excel_path),
                    top_n_per_person=int(top_n),
                    max_arxiv_records=max_records,
                    progress_callback=show_progress,
                )

            st.session_state.result = result
            progress_box.success("Analysis complete. Excel file created.")

        except Exception as error:
            st.session_state.last_error = str(error)
            progress_box.empty()
            st.error("Something went wrong.")
            st.exception(error)


# ============================================================
# Error display
# ============================================================

if st.session_state.last_error:
    with st.expander("Error Details"):
        st.write(st.session_state.last_error)


# ============================================================
# Results display
# ============================================================

result = st.session_state.result

if result:
    st.subheader("3. Results")

    participants_df = pd.DataFrame(result["participants"])
    tasks_df = pd.DataFrame(result["tasks"])
    assignments_df = pd.DataFrame(result["assignments"])
    papers_df = pd.DataFrame(result["research_recommendations"])

    tab_tasks, tab_assignments, tab_papers, tab_people = st.tabs(
        [
            "Tasks",
            "Assignments",
            "Research Recommendations",
            "Participants",
        ]
    )

    with tab_tasks:
        st.write("Extracted action items from the transcript.")
        if tasks_df.empty:
            st.info("No tasks found.")
        else:
            st.dataframe(
                tasks_df,
                use_container_width=True,
                hide_index=True,
            )

    with tab_assignments:
        st.write("Task ownership based on what each person volunteered to do.")
        if assignments_df.empty:
            st.info("No assignments found.")
        else:
            st.dataframe(
                assignments_df,
                use_container_width=True,
                hide_index=True,
            )

    with tab_papers:
        st.write("Research paper recommendations from the arXiv dataset.")
        if papers_df.empty:
            st.info("No research recommendations found.")
        else:
            st.dataframe(
                papers_df,
                use_container_width=True,
                hide_index=True,
            )

    with tab_people:
        st.write("People detected in the transcript.")
        if participants_df.empty:
            st.info("No participants found.")
        else:
            st.dataframe(
                participants_df,
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("4. Download Excel Output")

    excel_path = Path(result["excel_path"])

    if excel_path.exists():
        with excel_path.open("rb") as excel_file:
            st.download_button(
                label="Download Excel File",
                data=excel_file,
                file_name=excel_path.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )

        st.caption(f"Saved locally at: `{excel_path}`")
    else:
        st.warning("Excel file path was returned, but the file could not be found.")

else:
    st.info("Run the analysis to see tasks, assignments, paper recommendations, and the Excel download.")