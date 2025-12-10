import os
import random
import tempfile
import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# 1. Configure Gemini API Key
# -----------------------------
st.set_page_config(page_title="📘 Question Paper Generator", page_icon="🧠", layout="wide")

st.sidebar.header("🔐 Gemini API Key Settings")

# Your default (owner) key from env or secrets
DEFAULT_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)

# User-entered key (masked)
user_api_key = st.sidebar.text_input(
    "Enter your Gemini API Key",
    type="password",
    placeholder="Paste your key here",
    help="Optional: If left empty, the app will use the default built‑in API key."
)

# Button to open Google AI Studio API key dashboard (for generating keys)
st.sidebar.link_button(
    "🔑 Get / Manage Gemini API Key",
    "https://aistudio.google.com/app/apikey",
    help="Opens Google AI Studio where you can generate and manage your API keys."
)

# Instructions for teacher / user
st.sidebar.markdown(
    """
**Instructions**

1. Click **"Get / Manage Gemini API Key"** to open Google AI Studio.
2. Sign in with your Google account.
3. Create or copy an existing Gemini API key.
4. Paste the key into the field above.
5. If you do **not** enter anything, the app will use the default API key configured on the server.
"""
)

# Final key selection: user key first, otherwise default
GOOGLE_API_KEY = user_api_key.strip() if user_api_key.strip() else DEFAULT_GOOGLE_API_KEY

if not GOOGLE_API_KEY:
    st.error(
        "⚠️ No Gemini API key available.\n\n"
        "Either:\n"
        "- Set GOOGLE_API_KEY in Railway variables or `.streamlit/secrets.toml`, **or**\n"
        "- Paste a key in the sidebar input."
    )
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)


# -----------------------------
# 2. Streamlit UI
# -----------------------------
st.title("📘 AI Question Paper Generator (Google Gemini)")

st.markdown("""
Upload your **PDF**, **DOCX**, or **TXT** file.  
The AI will read the content and automatically create a **question paper** with multiple difficulty levels.
""")

uploaded_file = st.file_uploader("📂 Upload a document", type=["pdf", "docx", "txt"])

if uploaded_file:
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    full_text = " ".join([d.page_content for d in docs])

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
        "(e.g. \"Include one 10‑mark question explaining DBMS in detail\" "
        "or \"Generate only short 5‑mark questions\" )",
        value="",
        height=120,
    )

    if st.button("🪄 Generate Question Paper"):
        with st.spinner("Generating your question paper... ⏳"):
            model = genai.GenerativeModel("gemini-2.5-flash-lite")

            # Random variation seed so each click has slightly different input
            variation_seed = random.randint(1, 1_000_000)

            # If teacher gives instructions, let them control the pattern
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

            response = model.generate_content(prompt)
            question_paper = response.text

        st.markdown("## 🧾 Generated Question Paper")
        st.write(question_paper)

        st.download_button(
            label="💾 Download Question Paper (TXT)",
            data=question_paper,
            file_name="question_paper.txt",
            mime="text/plain"
        )

else:
    st.info("👆 Upload a file to begin.")
