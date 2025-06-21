from autogluon.tabular import TabularPredictor
import pandas as pd


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["title"].astype(str) + ". " + df["description"].fillna('').astype(str) + ". Genres: " + df[
        "genres"].fillna('[]').astype(str)
    df = df[["text", "my_rating"]].rename(columns={"my_rating": "target"})
    return df


def train_model(df: pd.DataFrame) -> TabularPredictor:
    predictor = TabularPredictor(
        label="target",
        problem_type="regression",
        path="data/06_models/autogluon/"
    ).fit(df, presets='medium_quality', time_limit=60)
    return predictor


def predict(predictor: TabularPredictor, new_movies: pd.DataFrame) -> pd.DataFrame:
    new_movies = new_movies.copy()
    new_movies["text"] = new_movies["title"].astype(str) + ". " + new_movies["description"].fillna('').astype(
        str) + ". Genres: " + new_movies["genres"].fillna('[]').astype(str)
    preds = predictor.predict(new_movies[["text"]])
    new_movies['will_like'] = preds
    # Sortujemy po ocenie "will_like" malejąco, top 10
    return new_movies[['title', 'will_like']].sort_values(by='will_like', ascending=False).head(10)
