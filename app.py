import streamlit as st
from PIL import Image
import google.generativeai as genai
import os
from dotenv import load_dotenv
import pdfplumber

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def extract_data(uploaded_file):
    data = []
    with pdfplumber.open(uploaded_file) as pdf:
        pages = pdf.pages
        for page in pages:
            data.append(page.extract_text())
    return data

def process_with_gemini(text, user_prompt):
    """Process the extracted text with Gemini API."""
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f"Answer the following prompt as a recruiter for a high tech company for a college student at the University of Washington: {text}. For the following resume: {user_prompt}")
    return response.text

def main():
    st.title("OCR and Gemini Text Analysis App")

    # File uploader
    uploaded_file = st.file_uploader("Choose a pdf file", type=["pdf"])

    if uploaded_file is not None:
        # Extract data
        extracted_text = extract_data(uploaded_file)

        # Get User Prompt
        user_prompt = st.text_input("Enter prompt")

        # Perform OCR
        if st.button("Submit"):
            with st.spinner("Processing"):
                analysis = process_with_gemini(extracted_text, user_prompt)
                st.subheader("Gemini Analysis:")
                st.write(analysis)


if __name__ == "__main__":
    main()