"""Guides tab, "Add your own benchmark": how to plug a custom test set in through the Custom/ folder.
The text is the README section between the guide markers, so it is written once."""
import io
import json
import zipfile
from pathlib import Path

import streamlit as st

README = Path(__file__).resolve().parents[1] / "README.md"
START, END = "<!-- dataset-guide:start -->", "<!-- dataset-guide:end -->"

# the dataset of the README example, ready to unzip in the project folder and change
EXAMPLE_NAME = "Company Policies"
EXAMPLE_DOCUMENTS = {
    "vacation.txt": "Every employee is entitled to 26 days of vacation per year. "
                    "Requests must be submitted 15 days in advance.",
    "remote_work.txt": "Remote work is allowed up to 2 days a week. The days are agreed with the team manager.",
    "expenses.txt": "Travel expenses are refunded within 30 days of the end of the trip, with the receipts attached.",
}
EXAMPLE_QUESTIONS = [
    {"id": "pol_001", "query_type": "Simple_Extraction", "question": "How many vacation days is an employee entitled to?",
     "context": "[Doc 1] Every employee is entitled to 26 days of vacation per year.", "gold_answer": "26 days per year"},
    {"id": "pol_002", "query_type": "Multi_Hop",
     "question": "How far in advance must I request vacation, and how many remote work days can I take?",
     "context": "[Doc 1] Requests must be submitted 15 days in advance. [Doc 2] Remote work is allowed up to 2 days a week.",
     "gold_answer": "Vacation must be requested 15 days in advance; remote work is allowed up to 2 days a week."},
    {"id": "pol_003", "query_type": "Unanswerable", "question": "What is the daily meal voucher?",
     "context": "", "gold_answer": "The policies do not contain this information."},
]


def guide_text():
    """The README part between the markers, None if they are missing."""
    text = README.read_text(encoding="utf-8") if README.exists() else ""
    if START not in text or END not in text:
        return None
    return text.split(START, 1)[1].split(END, 1)[0].strip()


def example_zip():
    """Custom/Company Policies/ with questions.jsonl and documents/, as a zip in memory."""
    folder = f"Custom/{EXAMPLE_NAME}"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{folder}/questions.jsonl",
                   "".join(json.dumps({"id": q["id"], "source": EXAMPLE_NAME, **q}, ensure_ascii=False) + "\n" for q in EXAMPLE_QUESTIONS))
        for name, text in EXAMPLE_DOCUMENTS.items():
            z.writestr(f"{folder}/documents/{name}", text + "\n")
    return buffer.getvalue()


def render():
    st.markdown("## How to add your own benchmark")
    st.download_button("Download example", example_zip(), file_name="custom_dataset_example.zip", mime="application/zip",
                       help=f"A working dataset ({len(EXAMPLE_QUESTIONS)} questions, {len(EXAMPLE_DOCUMENTS)} documents): "
                            "unzip it in the project folder, then change it.")
    text = guide_text()
    if text is None:
        st.info("The guide is in README.md, section *Creating Your Own Dataset*.")
    else:
        st.markdown(text)
