import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movie_data = pd.read_csv("cleaned_movie_data.csv")

# Remove duplicate movies
movie_content = movie_data[['movieId', 'title', 'genres']].drop_duplicates()

# Convert genres into TF-IDF vectors
tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(movie_content['genres'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create index mapping for movie titles
indices = pd.Series(
    movie_content.index,
    index=movie_content['title']
).drop_duplicates()


# Hybrid recommendation function
def hybrid_recommendation(title, top_n=10):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:top_n+1]

    recommendations = []

    for i, score in sim_scores:

        recommendations.append((
            movie_content.iloc[i]['title'],
            score
        ))

    return recommendations


# Streamlit App
st.title("Hybrid Movie Recommendation System")

movie_title = st.selectbox(
    "Select a Movie",
    movie_content['title'].values
)

if st.button("Get Recommendations"):

    recommendations = hybrid_recommendation(movie_title)

    st.subheader("Recommended Movies")

    for movie, score in recommendations:

        st.write(f"{movie} ⭐ {score:.2f}")
