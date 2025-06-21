import streamlit as st
import pandas as pd
from autogluon.tabular import TabularPredictor

# Ścieżki
MODEL_PATH = "data/06_models/autogluon/"
MOVIES_PATH = "data/01_raw/netflix_titles.csv"


# Wczytaj dane i model
@st.cache_data
def load_data():
    df = pd.read_csv(MOVIES_PATH)
    df["text"] = df["title"].astype(str) + ". " + df["description"].fillna('').astype(str) + ". Genres: " + df[
        "genres"].fillna("[]").astype(str)
    return df


@st.cache_resource
def load_model():
    return TabularPredictor.load(MODEL_PATH)


# Interfejs
st.title("🎬 Movie Recommender – Based on Your Taste")

df = load_data()
predictor = load_model()

# 🎯 Wybór filmu
movie_title = st.selectbox("Wybierz film, który chcesz ocenić:", df["title"].unique())

# 🔮 Predykcja
selected_movie = df[df["title"] == movie_title][["title", "text"]]
if not selected_movie.empty:
    st.subheader("📈 Prognozowana Twoja ocena:")
    pred = predictor.predict(selected_movie[["text"]])
    st.write(f"**{movie_title}** → **{pred.values[0]:.2f} / 10**")

    # 🔝 Top 10 polecanych filmów
    st.subheader("🎥 Podobne filmy, które możesz polubić:")
    df["pred_rating"] = predictor.predict(df[["text"]])
    top_movies = df[df["title"] != movie_title].sort_values("pred_rating", ascending=False).head(10)
    st.table(top_movies[["title", "pred_rating"]])
