import os
import random
import tempfile

import streamlit as st
import google.generativeai as genai
import google.api_core.exceptions as google_exceptions

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# 1. Configure Gemini API Key
# -----------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get(
    "GOOGLE_API_KEY", None
)
if not GOOGLE_API_KEY:
    st.error(
        "⚠️ Missing Google Generative AI Key! "
        "Set GOOGLE_API_KEY in Railway variables or .streamlit/secrets.toml."
    )
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# -----------------------------
# 1b. Model fallback config
# -----------------------------
PRIMARY_MODEL = "gemini-2.0-flash"
FALLBACK_MODELS = [
    "gemini-2.5-flash-lite",
    "gemma-3-1b",
    "gemma-3-2b",
    "gemma-3-4b",
    "gemma-3-12b",
]


def generate_with_fallback(prompt: str) -> str:
    """Try multiple models until one succeeds."""
    last_error = None

    for model_name in [PRIMARY_MODEL] + FALLBACK_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if hasattr(response, "text") and response.text:
                return response.text
            # Some SDK versions use candidates; be defensive.
            try:
                return response.candidates[0].content.parts[0].text
            except Exception:
                continue
        except google_exceptions.ResourceExhausted as e:
            # Quota / rate limit hit for this model → move to next.
            last_error = e
        except Exception as e:
            # Any other model-specific error → remember and try next.
            last_error = e

    # If all models fail, raise the last error.
    raise last_error if last_error else RuntimeError(
        "All models failed with unknown error."
    )


# -----------------------------
# 2. Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="📘 Question Paper Generator",
    page_icon="🧠",
    layout="wide",
)
st.title("📘 AI Question Paper Generator (Google Gemini)")

st.markdown(
    """
Upload your **PDF**, **DOCX**, or **TXT** file.  
The AI will read the content and automatically create a **question paper** with multiple difficulty levels.
"""
)

uploaded_file = st.file_uploader(
    "📂 Upload a document", type=["pdf", "docx", "txt"]
)

if uploaded_file:
    # Save temporarily
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
    ) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # Load document
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "docx":
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()

    # Split large docs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    full_text = " ".join(d.page_content for d in docs)

    st.success("✅ Document loaded successfully!")

    # -----------------------------
    # 3. Question paper settings
    # -----------------------------
    st.subheader("🧩 Select question paper pattern")
    num_two = st.number_input("Number of 2-mark questions", 0, 20, 10)
    num_five = st.number_input("Number of 5-mark questions", 0, 10, 5)
    num_ten = st.number_input("Number of 10-mark questions", 0, 5, 2)

    st.subheader("✏️ Extra instructions (optional)")
    teacher_comment = st.text_area(
        "Tell the AI how to customize the paper "
        '(e.g. "Include one 10‑mark question explaining DBMS in detail" '
        'or "Generate only short 5‑mark questions")',
        value="",
        height=120,
    )

    if st.button("🪄 Generate Question Paper"):
        with st.spinner("Generating your question paper... ⏳"):
            # Random variation seed so each click has slightly different input
            variation_seed = random.randint(1, 1_000_000)

            if teacher_comment.strip():
                requirement_text = (
                    "Follow these teacher instructions when deciding the number of "
                    "questions, marks and style:\n"
                    f"{teacher_comment.strip()}\n"
                )
            else:
                requirement_text = f"""
Generate exactly:
- {num_two} questions of 2 marks
- {num_five} questions of 5 marks
- {num_ten} questions of 10 marks
"""

            prompt = f"""
You are an expert university question paper setter.
You must create different sets of questions each time you are called,
avoiding reusing previous questions as much as possible.

Study material:

--- Document Content Start ---
{full_text[:15000]}
--- Document Content End ---

Requirements:
{requirement_text}

Additional rules:
- Do NOT repeat questions or wording from any previous paper you might have created.
- Vary phrasing, ordering and subtopics so each paper feels different.
- Use clear, exam-style questions only.

Random variation id: {variation_seed}

Format clearly like this:
📘 **Question Paper**
---
**Section A (2 Marks Questions)**
1. ...
2. ...

**Section B (5 Marks Questions)**
1. ...
2. ...

**Section C (10 Marks Questions)**
1. ...
2. ...
"""

            try:
                question_paper = generate_with_fallback(prompt)
            except Exception as e:
                st.error(f"Failed to generate question paper: {e}")
            else:
                st.markdown("## 🧾 Generated Question Paper")
                st.write(question_paper)
                st.download_button(
                    label="💾 Download Question Paper (TXT)",
                    data=question_paper,
                    file_name="question_paper.txt",
                    mime="text/plain",
                )

else:
    st.info("👆 Upload a file to begin.")

