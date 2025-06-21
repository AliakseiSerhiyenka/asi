import streamlit as st
import pandas as pd
import random
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

# 🔍 Szukanie po słowie kluczowym
keyword = st.text_input("🔎 Wpisz tytuł lub słowo kluczowe (np. 'Python', 'zombie', 'comedy'):")

if not keyword:
    st.info("Wpisz słowo kluczowe powyżej, aby zobaczyć pasujące filmy.")
else:
    matches = df[df["text"].str.contains(keyword, case=False, na=False)]

    if matches.empty:
        st.warning("Nie znaleziono żadnych pasujących filmów.")
    else:
        st.success(f"Znaleziono {len(matches)} filmów pasujących do: '{keyword}'")

        titles_list = matches["title"].unique()
        selected_title = st.selectbox("🎯 Wybierz film z listy:", titles_list)

        selected_movie = df[df["title"] == selected_title].iloc[0]

        if st.button("🔍 Pokaż rekomendacje"):
            selected_text = pd.DataFrame({"text": [selected_movie["text"]]})
            pred = predictor.predict(selected_text)
            st.subheader("📈 Prognozowana Twoja ocena:")
            st.write(f"**{selected_movie['title']}** → **{pred.values[0]:.2f} / 10**")

            # 🔝 Top 10
            df["pred_rating"] = predictor.predict(df[["text"]])
            top_movies = df[df["title"] != selected_movie["title"]].sort_values("pred_rating", ascending=False).head(10)
            st.subheader("🎥 Top 10 rekomendacji:")
            st.table(top_movies[["title", "pred_rating"]])

            # 🎲 Losowe 3 filmy z pred_rating > 7
            st.subheader("🎁 3 losowe filmy, które możesz też polubić:")
            high_rated = df[(df["title"] != selected_movie["title"]) & (df["pred_rating"] > 7)]
            random_recs = high_rated.sample(min(3, len(high_rated)), random_state=42)
            st.table(random_recs[["title", "pred_rating"]])
