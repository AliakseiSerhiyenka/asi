from kedro.pipeline import Pipeline, node
from .nodes import prepare_data, train_model, predict


def create_pipeline(**kwargs):
    return Pipeline(
        [

            node(prepare_data, inputs="my_rated_movies", outputs="prepared_training_data", name="prepare_data_node"),
            node(train_model, inputs="prepared_training_data", outputs="trained_model", name="train_model_node"),
            node(predict, inputs=["trained_model", "netflix_catalog"], outputs="recommendations", name="predict_node"),
        ]
    )
