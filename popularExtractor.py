import argparse
from itertools import chain
import json
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


# Display topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))



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

    

if __name__ == '__main__' :
    args = parseArgs()
    titles = list(chain.from_iterable(map(read_file, args.input)))
    # Preprocessing and vectorization
    vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
    title_vectors = vectorizer.fit_transform(titles)
    
    lda = LatentDirichletAllocation(n_components=args.num_topics, random_state=42)
    lda.fit(title_vectors)

    display_topics(lda, vectorizer.get_feature_names_out(), args.num_top_words)