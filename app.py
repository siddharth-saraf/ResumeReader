import streamlit as st
import pytesseract
from PIL import Image
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def perform_ocr(image):
    """Perform OCR on the uploaded image."""
    text = pytesseract.image_to_string(image)
    return text

def process_with_gemini(text):
    """Process the extracted text with Gemini API."""
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f"Analyze the following resume extracted from an image highlighting the key skils and giving a short summary and rating and areas for improvement: {text}")
    return response.text

def main():
    st.title("OCR and Gemini Text Analysis App")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform OCR
        if st.button("Perform OCR and Analyze"):
            with st.spinner("Processing..."):
                # OCR
                extracted_text = perform_ocr(image)
                st.subheader("Text Extracted")

                # Gemini Analysis
                analysis = process_with_gemini(extracted_text)
                st.subheader("Gemini Analysis:")
                st.write(analysis)

if __name__ == "__main__":
    main()