
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("recommender_model.keras")

@st.cache_data
def load_assets():
    df_movies = pd.read_csv("movies.csv")
    user_map, movie_map = joblib.load("encodings.pkl")
    return df_movies, user_map, movie_map

model = load_model()
movies_df, user2idx, movie2idx = load_assets()
reverse_movie_map = {v: k for k, v in movie2idx.items()}

st.title("TensorFlow Movie Recommender")
st.write("Select some movies you've liked to get recommendations:")

movie_titles = movies_df.set_index("movieId")["title"].to_dict()
movie_choices = [movie_titles[mid] for mid in movie2idx.keys() if mid in movie_titles]
selected_titles = st.multiselect("Liked movies", sorted(movie_choices))

user_ratings = {}
for title in selected_titles:
    movie_id = [k for k, v in movie_titles.items() if v == title][0]
    user_ratings[movie_id] = 5.0

if st.button("Get Recommendations"):
    if not user_ratings:
        st.warning("Please select at least one movie.")
    else:
        liked_indices = [movie2idx[m] for m in user_ratings if m in movie2idx]
        avg_embedding = tf.reduce_mean(model.layers[2](tf.constant(liked_indices)), axis=0, keepdims=True)
        all_movie_indices = tf.range(len(movie2idx))
        movie_embeddings = model.layers[3](all_movie_indices)
        scores = tf.reduce_sum(avg_embedding * movie_embeddings, axis=1).numpy()
        top_indices = np.argsort(scores)[::-1]

        recommended = []
        for idx in top_indices:
            mid = reverse_movie_map[idx]
            if mid not in user_ratings and mid in movie_titles:
                recommended.append((movie_titles[mid], scores[idx]))
            if len(recommended) >= 10:
                break

        st.subheader("Top Recommendations")
        for title, score in recommended:
            st.write(f"{title} — Score: {score:.3f}")
