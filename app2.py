# app.py
import os
import tempfile
import streamlit as st
import google-generativeai as genai
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# 1. Configure Gemini API Key
# -----------------------------
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("⚠️ Missing Google Generative AI Key! Add it to .streamlit/secrets.toml or environment variables.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# -----------------------------
# 2. Streamlit UI
# -----------------------------
st.set_page_config(page_title="📘 Question Paper Generator", page_icon="🧠", layout="wide")
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
    # 3. Choose question paper type
    # -----------------------------
    st.subheader("🧩 Select question paper pattern")
    num_two = st.number_input("Number of 2-mark questions", 0, 20, 10)
    num_five = st.number_input("Number of 5-mark questions", 0, 10, 5)
    num_ten = st.number_input("Number of 10-mark questions", 0, 5, 2)

    if st.button("🪄 Generate Question Paper"):
        with st.spinner("Generating your question paper... ⏳"):
            model = genai.GenerativeModel("gemini-2.0-flash")

            prompt = f"""
You are an expert question paper setter.
Create a question paper based on the following study material:

--- Document Content Start ---
{full_text[:15000]}   # limit for API
--- Document Content End ---

Generate:
- {num_two} questions of 2 marks
- {num_five} questions of 5 marks
- {num_ten} questions of 10 marks

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

        # Download option
        st.download_button(
            label="💾 Download Question Paper (TXT)",
            data=question_paper,
            file_name="question_paper.txt",
            mime="text/plain"
        )

else:
    st.info("👆 Upload a file to begin.")

