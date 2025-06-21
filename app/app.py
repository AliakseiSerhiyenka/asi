import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# === Funkcja: znajdź podobne filmy
def find_similar_movies(features_df: pd.DataFrame, selected_title: str, top_n: int = 3) -> pd.DataFrame:
    if selected_title not in features_df['title'].values:
        return pd.DataFrame({'message': [f"❌ Film '{selected_title}' nie został znaleziony."]})
    title_list = features_df['title'].values
    X = features_df.drop(columns=['title']).values
    idx = np.where(title_list == selected_title)[0][0]
    sim_scores = cosine_similarity([X[idx]], X)[0]
    similar_idx = sim_scores.argsort()[::-1][1:top_n + 1]
    return pd.DataFrame({
        'title': title_list[similar_idx],
        'similarity': sim_scores[similar_idx]
    })


# === UI
st.title("🎬 Znajdź podobne filmy")


@st.cache_data
def load_embeddings():
    return pd.read_parquet("data/05_model_input/movie_embeddings.parquet")


df = load_embeddings()

title = st.selectbox("Wybierz film:", sorted(df['title'].unique()))
top_n = st.slider("Ile podobnych filmów?", 1, 10, 3)

if st.button("🔍 Znajdź podobne"):
    results = find_similar_movies(df, title, top_n)
    st.dataframe(results)
