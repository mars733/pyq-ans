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

# API Key and Model in Sidebar
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
model_options = [
    "gemini-2.5-flash", 
    "gemini-2.0-flash", 
    "gemini-1.5-flash", 
    "gemini-1.5-pro",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite"
]
MODEL_ID = st.sidebar.selectbox("Select Model", model_options, index=0)

if not api_key:
    st.info("Please enter your Gemini API Key in the sidebar to start. You can get one for free at https://aistudio.google.com/")
    st.stop()

# Initialize Client
client = setup_gemini(api_key)

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

    # NEW: Allow user to review/edit OCR text
    with st.expander("🔍 View/Edit Extracted Text", expanded=False):
        st.info("The AI will use the text below to find questions. You can manually fix OCR errors here.")
        text = st.text_area("Extracted Content", value=text, height=300)
        st.session_state.extracted_text = text # Update session state with edits
        st.caption(f"Character Count: {len(text)}")

    # Step 1: Identify Questions
    if "questions" not in st.session_state or st.button("🔄 Re-scan for Questions"):
        with st.spinner("Extracting questions from text..."):
            prompt = f"Identify every individual question or prompt in the following exam text. List them as a numbered list. Include the section names if applicable.\n\nTEXT:\n{text}"
            try:
                response = client.models.generate_content(model=MODEL_ID, contents=prompt)
                # Find lines that look like questions
                q_list = [l.strip() for l in response.text.split('\n') if l.strip()]
                st.session_state.questions = q_list
            except Exception as e:
                st.error(f"AI Error: {e}")
                if "500" in str(e) or "quota" in str(e).lower():
                    st.warning("💡 **Tip:** It looks like this model is busy or out of quota. Try selecting a different model from the sidebar (e.g., Gemini 3.5 or 1.5).")
                st.stop()

    st.subheader("📋 Select Questions to Answer")
    
    if "questions" in st.session_state:
        # Copy All Questions
        all_q_text = "\n".join(st.session_state.questions)
        st.download_button("📋 Copy All Questions (Download txt)", all_q_text, file_name="questions.txt")
        
        col_mode, col_len = st.columns([1, 1])
        with col_mode:
            process_mode = st.radio("Processing Mode", ["One by One (Accurate)", "Batch (Fast/Save Quota)"], index=0)
        with col_len:
            global_length = st.selectbox("Global Length (for Batch/All)", ["50", "100", "150", "200", "250", "300", "400", "500", "800"], index=3)

        selected_questions = []
        word_count_options = ["50", "100", "150", "200", "250", "300", "400", "500", "800"]
        
        for i, q in enumerate(st.session_state.questions):
            if len(q) < 5: continue
            
            col1, col2 = st.columns([4, 1])
            with col1:
                is_selected = st.checkbox(q, key=f"q_{i}")
            with col2:
                length = st.selectbox("Words", word_count_options, key=f"len_{i}", index=3)
            
            if is_selected:
                selected_questions.append({"question": q, "length": length})

    if st.button("🚀 Generate Detailed Answers") and selected_questions:
        st.divider()
        with st.spinner("Thinking..."):
            results = []
            
            if process_mode == "One by One (Accurate)":
                for item in selected_questions:
                    q_prompt = f"Provide a high-quality, exam-standard answer for this question: '{item['question']}'. The answer MUST be approximately {item['length']} words. Use clear headings and bullet points where appropriate."
                    try:
                        response = client.models.generate_content(model=MODEL_ID, contents=q_prompt)
                        ans_text = response.text
                        results.append(f"## {item['question']}\n\n{ans_text}\n\n")
                        st.markdown(f"## {item['question']}")
                        st.code(ans_text, language="markdown") # Easy to copy
                        st.divider()
                    except Exception as e:
                        st.error(f"Error generating answer for '{item['question']}': {e}")
            else:
                # Batch Processing
                batch_q = "\n".join([f"- Question: {item['question']} | Requested Length: {item['length']} words" for item in selected_questions])
                batch_prompt = (
                    f"You are an expert examiner. Provide high-quality, exam-standard answers for the following questions. "
                    f"For each question, adhere strictly to the requested word count. "
                    f"Use clear Markdown headers (##) for each question.\n\n"
                    f"QUESTIONS TO ANSWER:\n{batch_q}"
                )
                try:
                    response = client.models.generate_content(model=MODEL_ID, contents=batch_prompt)
                    ans_text = response.text
                    st.markdown("### 📝 Generated Answers (Batch)")
                    st.markdown(ans_text)
                    st.divider()
                    st.subheader("📋 Copy Batch Result")
                    st.code(ans_text, language="markdown") 
                    results.append(ans_text)
                except Exception as e:
                    st.error(f"Batch Error: {e}")
            
            if results:
                final_output = "\n".join(results)
                st.download_button(
                    label="📥 Download Full Answer Sheet (.md)",
                    data=final_output,
                    file_name=f"answers_{uploaded_file.name}.md",
                    mime="text/markdown"
                )
