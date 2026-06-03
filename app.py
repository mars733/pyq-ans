import streamlit as st
import os
import io
import re
import fitz  # PyMuPDF
import easyocr
from google import genai
import numpy as np
from PIL import Image

# Setup Gemini
def setup_gemini(api_key):
    return genai.Client(api_key=api_key)

def extract_text_digital(uploaded_file):
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

def extract_text_ocr(uploaded_file):
    """
    Renders PDF pages as images and uses EasyOCR to extract text.
    """
    reader = easyocr.Reader(['en'])
    # Ensure we are at the start of the file if it was read before
    uploaded_file.seek(0)
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    full_text = []
    progress_bar = st.progress(0)
    
    for i, page in enumerate(doc):
        # Render page to image (pixmap)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # Higher resolution
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        # OCR the image
        result = reader.readtext(np.array(img), detail=0)
        full_text.append("\n".join(result))
        
        # Update progress
        progress_bar.progress((i + 1) / len(doc))
    
    return "\n\n".join(full_text)

st.set_page_config(page_title="Exam Solver AI", page_icon="🎓")

st.title("🎓 Smart Exam Solver AI")
st.markdown("Upload your exam PDF/Text, pick your questions, and get AI-generated answers with custom lengths.")

# API Key in Sidebar
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
if not api_key:
    st.info("Please enter your Gemini API Key in the sidebar to start. You can get one for free at https://aistudio.google.com/")
    st.stop()

# Initialize Client
client = setup_gemini(api_key)
MODEL_ID = "gemini-2.5-flash"

uploaded_file = st.file_uploader("Upload Exam Paper (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    # Reset session if new file is uploaded
    if "file_name" not in st.session_state or st.session_state.file_name != uploaded_file.name:
        st.session_state.file_name = uploaded_file.name
        if "extracted_text" in st.session_state: del st.session_state.extracted_text
        if "questions" in st.session_state: del st.session_state.questions

    if "extracted_text" not in st.session_state:
        with st.spinner("Analyzing document..."):
            if uploaded_file.type == "application/pdf":
                text = extract_text_digital(uploaded_file)
                if len(text.strip()) < 100:
                    st.warning("Digital extraction yielded little text. Attempting AI-powered OCR for scans...")
                    text = extract_text_ocr(uploaded_file)
            else:
                text = uploaded_file.getvalue().decode("utf-8")
            st.session_state.extracted_text = text

    text = st.session_state.extracted_text
    
    if len(text.strip()) < 20:
        st.error("Could not extract any meaningful text from the document.")
        st.stop()

    # Step 1: Identify Questions
    if "questions" not in st.session_state:
        with st.spinner("Extracting questions from text..."):
            prompt = f"Identify every individual question or prompt in the following exam text. List them as a numbered list. Include the section names if applicable.\n\nTEXT:\n{text}"
            try:
                response = client.models.generate_content(model=MODEL_ID, contents=prompt)
                # Find lines that look like questions
                q_list = [l.strip() for l in response.text.split('\n') if l.strip()]
                st.session_state.questions = q_list
            except Exception as e:
                st.error(f"AI Error: {e}")
                st.stop()

    st.subheader("📋 Select Questions to Answer")
    selected_questions = []
    
    if "questions" in st.session_state:
        for i, q in enumerate(st.session_state.questions):
            # Filter for actual questions (Gemini might add commentary)
            if len(q) < 5: continue
            
            col1, col2 = st.columns([4, 1])
            with col1:
                is_selected = st.checkbox(q, key=f"q_{i}")
            with col2:
                length = st.selectbox("Length", ["Short (50w)", "Medium (200w)", "Long (500w)", "Essay (800w)"], key=f"len_{i}", index=1)
            
            if is_selected:
                selected_questions.append({"question": q, "length": length})

    if st.button("🚀 Generate Detailed Answers") and selected_questions:
        st.divider()
        with st.spinner("Thinking..."):
            results = []
            for item in selected_questions:
                q_prompt = f"Provide a high-quality, exam-standard answer for this question: '{item['question']}'. The answer MUST be approximately {item['length']}. Use clear headings and bullet points where appropriate."
                try:
                    response = client.models.generate_content(model=MODEL_ID, contents=q_prompt)
                    results.append(f"## {item['question']}\n\n{response.text}\n\n")
                    st.markdown(f"## {item['question']}")
                    st.markdown(response.text)
                    st.divider()
                except Exception as e:
                    st.error(f"Error generating answer for '{item['question']}': {e}")
            
            if results:
                final_output = "\n".join(results)
                st.download_button(
                    label="📥 Download Full Answer Sheet (.md)",
                    data=final_output,
                    file_name=f"answers_{uploaded_file.name}.md",
                    mime="text/markdown"
                )
