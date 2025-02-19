from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# Sample list of Reddit post titles
titles = [
    "Best budget smartphones in 2023",
    "How to improve your credit score",
    "Top 10 movies to watch this weekend",
    "Tips for saving money on groceries",
    "Latest smartphone releases and reviews",
    "How to invest in the stock market",
    "Best action movies of all time",
    "Ways to reduce your monthly expenses",
    "New tech gadgets you need in 2023",
    "How to build a successful budget"
]

# Preprocessing and vectorization
vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
title_vectors = vectorizer.fit_transform(titles)

# Apply LDA
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(title_vectors)

# Display topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


if __name__ == 'main':
    no_top_words = 5
    display_topics(lda, vectorizer.get_feature_names_out(), no_top_words)