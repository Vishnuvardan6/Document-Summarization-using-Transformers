import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
import os

# Load pre-trained models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Helper function to extract text from files
def extract_text_from_file(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = DocxDocument(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

# Streamlit app layout
st.title("Document Summarizer and Question Answering System")

# File upload section
uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    try:
        # Extract text from the uploaded file
        with st.spinner("Extracting text from the document..."):
            document_text = extract_text_from_file(uploaded_file)
        
        # Display the extracted text
        st.subheader("Extracted Document Text:")
        st.text_area("Text Preview", value=document_text[:500] + "...", height=150)

        # Generate summary
        with st.spinner("Generating summary..."):
            summary = summarizer(document_text, max_length=150, min_length=30, do_sample=False)
            summary_text = summary[0]['summary_text']
        
        # Display the summary
        st.subheader("Summary of the Document:")
        st.write(summary_text)

        # Question answering section
        st.subheader("Ask Questions About the Document")
        user_question = st.text_input("Enter your question here:")

        if user_question:
            with st.spinner("Finding the answer..."):
                qa_result = qa_model(question=user_question, context=document_text)
                answer = qa_result['answer']
                confidence = qa_result['score']

            st.subheader("Answer:")
            st.write(f"**{answer}**")
            st.write(f"Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")