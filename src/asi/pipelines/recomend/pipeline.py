from kedro.pipeline import Pipeline, node
from .nodes import prepare_data, train_model, predict


def create_pipeline(**kwargs):
    return Pipeline(
        [

            node(prepare_data, inputs="my_rated_movies", outputs="prepared_training_data", name="prepare_data_node"),
            node(train_model, inputs="prepared_training_data", outputs="trained_model", name="train_model_node"),
            node(predict, inputs=["trained_model", "netflix_catalog"], outputs="recommendations", name="predict_node"),
            # node(
            #     func=prepare_data,
            #     inputs="my_rated_movies",
            #     outputs="train_data",
            #     name="prepare_data"
            # ),
            # node(
            #     func=extract_embeddings,
            #     inputs=["netflix_titles", "predictor"],
            #     outputs="movie_embeddings",
            #     name="extract_features_node"
            # ),
            # node(
            #     func=find_similar_movies,
            #     inputs=["movie_embeddings", "params:selected_title", "params:top_n"],
            #     outputs="similar_movies",
            #     name="find_similar_movies_node"
            # ),
        ]
    )
