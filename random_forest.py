from typing import Dict
import pandas as pd

from decision_tree import DecisionTree


class RandomForest:
    def __init__(
        self, 
        number_of_trees: int, 
        classification_attribute: str, 
        possible_values_for_categorical_attributes: Dict = None,
        random_state: int = 42
    ):
        if number_of_trees < 2:
            raise ValueError('Invalid number of trees. It must be greater or equal to two.')
        self.number_of_trees = number_of_trees
        self.classification_attribute = classification_attribute
        self.possible_values_for_categorical_attributes = possible_values_for_categorical_attributes
        self.random_state = random_state
        self.tree_ensemble = []

    def construct(self, dataset: pd.DataFrame):
        bootstraps = self._bootstrap_aggregation(dataset)
        for b in bootstraps:
            decision_tree = DecisionTree(
                classification_attribute=self.classification_attribute, 
                possible_values_for_categorical_attributes=self.possible_values_for_categorical_attributes
            )
            decision_tree.train(b)
            self.tree_ensemble.append(decision_tree)
    
    def _bootstrap_aggregation(self, dataset: pd.DataFrame):
        number_of_samples = dataset.shape[0]
        bootstraps = []
        for i in range(self.number_of_trees):
            # TODO: Each bootstrap must have all the possible values for an attribute
            bootstraps.append(dataset.sample(n=int(number_of_samples/self.number_of_trees), random_state=(self.random_state * (i + 1)), replace=True))
        return bootstraps # TODO: Check the number of samples for each group

    def predict(self, samples):
        prediction_ensemble = []
        for decision_tree in self.tree_ensemble:
            prediction_ensemble.append(decision_tree.predict(samples))
        predictions = self._get_mode_for_each_train_sample_predictions(prediction_ensemble)
        return predictions

    def _get_mode_for_each_train_sample_predictions(self, prediction_ensemble):
        prediction_ensemble_dataframe = pd.DataFrame(prediction_ensemble)
        predictions = prediction_ensemble_dataframe.mode(axis=0)
        return predictions.iloc[0].values
    