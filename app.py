import streamlit as st
import random


def load_movies():
    filepath = os.path.join(os.path.dirname(__file__), "data.txt")
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def recommend_movies(watched_movies, all_movies, num_recommendations=3):
    unwatched = list(set(all_movies) - set(watched_movies))
    return random.sample(unwatched, min(len(unwatched), num_recommendations))


def main():
    st.title("Filmowy Doradca")
    st.write("Wpisz filmy, które już oglądałeś, a my zaproponujemy Ci coś nowego!")

    all_movies = load_movies()

    watched_movies = st.multiselect("Wybierz filmy, które widziałeś:", all_movies)

    if st.button("Pokaż rekomendacje"):
        if watched_movies:
            recommendations = recommend_movies(watched_movies, all_movies)
            st.subheader("Proponowane filmy:")
            for movie in recommendations:
                st.write(f"- {movie}")
        else:
            st.warning("Wybierz przynajmniej jeden film, aby otrzymać rekomendacje.")


if __name__ == "__main__":
    main()
