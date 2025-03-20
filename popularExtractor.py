import argparse
from itertools import chain
import os
import json
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
import re

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

def read_file(filePath):
    try:
        constructedPath = ''
        if os.path.isfile(filePath):
            constructedPath = filePath
        elif os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), filePath)):
            constructedPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filePath)
        with open(constructedPath, 'r') as file:
            data = json.load(file)
            # Assuming each JSON file contains a list of titles
            if isinstance(data, list):
                return data
            else:
                print(f"Warning: {constructedPath} does not contain a list. Skipping.")
                return []
    except FileNotFoundError:
        print(f"Error: File {filePath} not found. Skipping.")
        return []
    except json.JSONDecodeError:
        print(f"Error: File {filePath} is not a valid JSON file. Skipping.")
        return []

# Function to perform sentiment analysis
def analyze_sentiment(texts):
    # Initialize sentiment analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis")
    sentiments = sentiment_pipeline(texts)
    return sentiments

# Function to perform topic modeling
def perform_topic_modeling(texts, n_topics=5, n_words=30):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    topics = []
    for idx, topic in enumerate(lda.components_):
        topics.append([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-n_words - 1:-1]])
    return topics


def parseArgs():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Perform topic modeling on Reddit post titles.")
    parser.add_argument(
        "--input", 
        type=str,
        nargs='+',
        required=True, 
        help="Path to the input file containing Reddit post titles (one title per line)."
    )
    parser.add_argument(
        "--num_topics", 
        type=int, 
        default=3, 
        help="Number of topics to model (default: 3)."
    )
    parser.add_argument(
        "--num_top_words", 
        type=int, 
        default=5, 
        help="Number of top words to display per topic (default: 5)."
    )

    # Parse arguments
    args = parser.parse_args()
    return args

# Main function
def main():
    args = parseArgs()
    # print(args.input)
    # exit(0)
    posts = list(chain.from_iterable(map(read_file, args.input)))
    # posts = read_file(args.input)
    # Download NLTK stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    # Perform sentiment analysis
    sentiments = analyze_sentiment(posts)
    negative_posts = []
    # print("Sentiment Analysis Results:")
    for text, sentiment in zip(posts, sentiments):
        if sentiment['label'] == 'NEGATIVE':
            # print(f"Text: {text}\nSentiment: {sentiment}\n")
            negative_posts.append(text)
    
    # Perform topic modeling
    topics = perform_topic_modeling(negative_posts)
    print("Identified Topics:")
    for idx, topic in enumerate(topics):
        print(f"Topic {idx + 1}: {', '.join(topic)}")

if __name__ == "__main__":
    main()