import streamlit as st
import pdfplumber
import re
from collections import Counter
from nltk.corpus import stopwords
import nltk
from transformers import pipeline
import matplotlib.pyplot as plt

nltk.download('stopwords')

@st.cache_resource 
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")  # Specify PyTorch

@st.cache_resource  
def load_sentiment_analyzer():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", framework="pt")  # Specify PyTorch

summarizer = load_summarizer()
sentiment_analyzer = load_sentiment_analyzer()

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    text = re.sub(r'([a-zA-Z]+)-\s+', r'\1', text) 
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

def get_top_words(text, top_n=5):
    words = re.findall(r'\b\w+\b', text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(top_n)
    top_words_text = ", ".join([f"{word[0]} ({word[1]})" for word in top_words])
    return top_words, top_words_text

def analyze_sentiment(text):
    max_length = 512
    if len(text) > max_length:
        sentiment_results = sentiment_analyzer(text[:max_length]) 
        sentiment = sentiment_results[0]  
    else:
        sentiment_results = sentiment_analyzer(text)  
        sentiment = sentiment_results[0]  
    return sentiment

def generate_summary(text, summary_type):
    if len(text.strip()) == 0:
        return "The PDF contains no extractable text."

    max_input_length = 1024  
    text = text[:max_input_length]  

    try:
        if summary_type == "Quick Summary":
            summary = summarizer(text, max_length=350, min_length=150, do_sample=False)[0]["summary_text"]
        elif summary_type == "Detailed Summary":
            summary = summarizer(text, max_length=850, min_length=350, do_sample=False)[0]["summary_text"]
        elif summary_type == "Section-Specific Summary":
            sections = text.split("\n\n") 
            section_summaries = []
            for i, section in enumerate(sections):
                if len(section.strip()) > 30:  
                    section_summary = summarizer(section, max_length=160, min_length=50, do_sample=False)[0]["summary_text"]
                    section_summaries.append(f"Section {i+1}: {section_summary}")
            summary = "\n\n".join(section_summaries)
        else:
            summary = "Invalid summary type selected."
    except Exception as e:
        summary = f"An error occurred during summarization: {str(e)}"
    return summary

st.title("PDF Analysis Tool with Summary and Sentiment")
st.write("Upload a PDF file to analyze the most frequent words, sentiment, and generate summaries.")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

summary_type = st.selectbox("Choose Summary Type", ["Quick Summary", "Detailed Summary", "Section-Specific Summary"])

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)
    
    if len(text.strip()) == 0:
        st.error("No readable text found in the uploaded PDF. Please try a different file.")
    else:
        with st.spinner("Calculating top words..."):
            top_words, top_words_text = get_top_words(text)
        st.subheader("Top 5 Words (Excluding Stop Words):")
        st.write(f"The most frequent words are: {top_words_text}")
        
        words, counts = zip(*top_words)
        plt.figure(figsize=(8, 4))
        plt.bar(words, counts, color="skyblue")
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.title("Top 5 Frequent Words")
        st.pyplot(plt)

        with st.spinner("Analyzing sentiment..."):
            sentiment = analyze_sentiment(text)
        st.subheader("Sentiment Analysis:")
        st.write(f"Label: {sentiment['label']}, Score: {sentiment['score']:.2f}")
        
        with st.spinner(f"Generating {summary_type.lower()}..."):
            summary = generate_summary(text, summary_type)
        st.subheader(f"{summary_type}:")
        st.write(summary)