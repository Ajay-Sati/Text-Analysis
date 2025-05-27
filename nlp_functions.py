# nlp_functions.py
# we will try and catch a=block , so that agar code fate to other person dont see out code


from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.util import ngrams
from collections import Counter
import plotly.graph_objects as go
import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import math




# WORD CLOUD.
def show_wordcloud(tokens):
    """
    Generate a WordCloud from a list of tokens.
    """
    try:
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(tokens))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        return plt
    except Exception as e:
        return f"Error generating word cloud: {e}"


# N-GRAM ANALYSIS
def plot_top_ngrams_bar_chart(tokens, gram_n=2, top_n=15):
    try:
        # Step 1: Create n-grams
        ngram_list = list(ngrams(tokens, gram_n))

        # Step 2: Count most common n-grams
        ngram_counts = Counter(ngram_list).most_common(top_n)

        if not ngram_counts:
            raise ValueError("No n-grams found in the given token list.")

        # Step 3: Prepare labels and counts
        labels = []
        counts = []
        for ngram, count in ngram_counts:
            labels.append(' '.join(ngram))
            counts.append(count)

        # Step 4: Gradient colors
        colors = []
        for i in range(len(counts)):
            alpha = 0.3 + i * 0.045
            colors.append(f'rgba(0, 100, 200, {alpha})')  # Blue gradient

        # Step 5: Plotly Bar Chart
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=counts,
                marker_color=colors,
                text=counts,
                textposition='outside'
            )
        ])

        # Step 6: Layout settings
        fig.update_layout(
            title=f"Top {top_n} {gram_n}-grams",
            xaxis_title=f"{gram_n}-grams",
            yaxis_title="Frequency",
            xaxis_tickangle=-45,
            template='plotly_white',
            margin=dict(t=50, b=120),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        print(f"An error occurred: {e}")



# FUNCTION TO CREATE CHUNKS FROM THE TEXT

import pysbd

# Initialize the segmenter
segmenter = pysbd.Segmenter(language="en", clean=False)

def split_text_into_chunks_spacy(text, max_length=500):
    """
    Split long text into sentence-based chunks using pysbd.
    Handles messy, unpunctuated text better than standard tokenizers.
    """
    try:
        sentences = segmenter.segment(text)  # Returns a list of sentences

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence  # Start new chunk

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    except Exception as e:
        return f"Error: {e}"






# EMOTION ANALYSIS
from transformers import AutoTokenizer, pipeline
import pandas as pd  # for tabular output

# Load tokenizer and model
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
emotion_model = pipeline("text-classification", model=model_name, tokenizer=tokenizer, top_k=None)



def detect_emotions(text):
    chunks =split_text_into_chunks_spacy(text)
    emotion_scores = {}

    for chunk in chunks:
        results = emotion_model(chunk)[0]
        for result in results:
            label = result['label']
            score = result['score']
            emotion_scores[label] = emotion_scores.get(label, 0) + score

    # Sort and return top 5 emotions
    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_emotions[:5]

    # Create and return DataFrame
    df = pd.DataFrame(top_5, columns=["Emotion", "Score"])
    return df





#SENTIMENT ANALYSIS


# Load model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, return_all_scores=True)

# Function to get overall sentiment based on average scores
def detect_overall_sentiment_avg(text):
    try:
        sentiment_labels = {
            'LABEL_0': 'Negative',
            'LABEL_1': 'Neutral',
            'LABEL_2': 'Positive'
        }

        chunks = split_text_into_chunks_spacy(text)
        score_totals = {'Negative': 0.0, 'Neutral': 0.0, 'Positive': 0.0}
        chunk_count = len(chunks)

        for chunk in chunks:
            results = sentiment_classifier(chunk)[0]  # return_all_scores=True gives list of dicts
            for res in results:
                label = sentiment_labels[res['label']]
                score_totals[label] += res['score']

        # Calculate average scores
        avg_scores = {label: score_totals[label]/chunk_count for label in score_totals}

        # Select sentiment with highest average score
        overall_sentiment = max(avg_scores, key=avg_scores.get)

        return {
            "overall_sentiment": overall_sentiment,
            "average_scores": avg_scores,
            "total_chunks": chunk_count
        }

    except Exception as e:
        return {"error": str(e)}



# SENTENCE TYPE CLASSIFICATION

# Load zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

labels = [
    "factual",
    "opinion",
    "question",
    "command",
    "emotion",
    "personal experience",
    "suggestion",
    "story",
    "prediction",
    "warning",
    "instruction",
    "definition",
    "narrative",
    "news",
    "argument"
]

def classify_custom(text):
    result = classifier(text, candidate_labels=labels)
    return {
        "text": text,
        "predicted_category": result["labels"][0],
        "score": result["scores"][0],
        "all_categories": list(zip(result["labels"], result["scores"]))
    }

#SUMMARY GENERATION

from transformers import pipeline

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def safe_summarize(text, max_length=300, min_length=100, chunk_size=900):
    """
    A helper to safely summarize text under token limits.
    """
    if len(text.split()) <= chunk_size:
        result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return result[0]['summary_text']
    else:
        chunks = split_text_into_chunks_spacy(text, chunk_size)
        summaries = []
        for i, chunk in enumerate(chunks):
            print(f"summaring chunk {i+1} of {len(chunks)}...")
            result = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summaries.append(result[0]['summary_text'])
            print("Generating final summary from the summarize chunks")
            final_summary= safe_summarize(" ".join(summaries), max_length, min_length, chunk_size) # calling the function in itself , recursive functions.
        return final_summary



