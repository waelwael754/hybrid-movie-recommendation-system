import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split


movie_data = pd.read_csv("cleaned_movie_data.csv")

movie_content = movie_data[['movieId', 'title', 'genres']].drop_duplicates()

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(movie_content['genres'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series( movie_content.index, index=movie_content['title']).drop_duplicates()


reader = Reader(rating_scale=(0.5, 5))

data = Dataset.load_from_df(movie_data[['userId', 'movieId', 'rating']],reader)

trainset, testset = train_test_split(data,test_size=0.2,random_state=42)
model = SVD()
model.fit(trainset)

def hybrid_recommendation(user_id, title, top_n=10):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:top_n+1]

    recommendations = []

    for i, score in sim_scores:

        movie_id = movie_content.iloc[i]['movieId']

        predicted_rating = model.predict(user_id, movie_id).est

        hybrid_score = (0.7 * predicted_rating) + (0.3 * score)

        recommendations.append((
            movie_content.iloc[i]['title'],
            hybrid_score
        ))

    recommendations = sorted(
        recommendations,
        key=lambda x: x[1],
        reverse=True
    )

    return recommendations


# Streamlit App Title
st.title("Hybrid Movie Recommendation System")

user_id = st.number_input("Enter User ID",min_value=1,value=1)

movie_title = st.selectbox("Select a Movie",movie_content['title'].values)
if st.button("Get Recommendations"):

    recommendations = hybrid_recommendation(
        user_id,
        movie_title
    )

    st.subheader("Recommended Movies")

    for movie, score in recommendations:

        st.write(f"{movie} ⭐ {score:.2f}")