from typing import Dict
from treelib import Node, Tree
import numpy as np
import pandas as pd


class DecisionTreeNode(Node):
    def __init__(self, attribute=None, parent_attribute_value=None, decision=None, information_gain=None):
        if decision is not None:
            super().__init__(tag=str((parent_attribute_value, decision)))
        elif parent_attribute_value is None:
            super().__init__(tag=str((attribute, information_gain)))
        else:
            super().__init__(tag=str(((parent_attribute_value, attribute, information_gain))))
        self.parent_attribute_value = parent_attribute_value
        self.attribute = attribute
        self.decision = decision


class DecisionTree(Tree):
    def __init__(self, classification_attribute, possible_values_for_categorical_attributes: Dict = None):
        super().__init__()
        self.classification_attribute = classification_attribute
        self.possible_values_for_categorical_attributes = possible_values_for_categorical_attributes

    def train(self, dataset):
        if (self.possible_values_for_categorical_attributes is None):
            self.possible_values_for_categorical_attributes = self._get_possible_values_for_categorical_attributes(dataset)
        self.construct(dataset, subset=dataset)
    
    def construct(self, dataset, subset=None, parent=None, parent_attribute_value=None):
        is_pure_subset, classification = self._is_pure_subset(subset)
        if is_pure_subset:
            self.add_node(
                DecisionTreeNode(
                    decision=classification,
                    parent_attribute_value=parent_attribute_value,
                ),
                parent=parent
            )
        elif not self._has_attributes(subset):
            self.add_node(
                DecisionTreeNode(
                    decision=self._get_majority_class(subset),
                    parent_attribute_value=parent_attribute_value
                ),
                parent=parent
            )
        else:
            most_important_attribute, information_gain = self._get_most_important_attribute(subset)
            current_node = DecisionTreeNode(
                attribute=most_important_attribute,
                parent_attribute_value=parent_attribute_value,
                information_gain=information_gain
            )

            if parent:
                self.add_node(current_node, parent=parent)
            else:
                self.add_node(current_node) # Root

            for value in self.possible_values_for_categorical_attributes[most_important_attribute]:
                new_subset = subset.loc[subset[most_important_attribute] == value]
                if not self._has_instances(new_subset): # When the new subset has no instances
                    self.add_node(
                        DecisionTreeNode(
                            decision=self._get_majority_class(subset),
                            parent_attribute_value=value
                        ),
                        parent=current_node
                    )
                else:      
                    self.construct(
                        dataset=dataset,
                        subset=new_subset.drop(columns=[most_important_attribute]),
                        parent=current_node,
                        parent_attribute_value=value
                    )

    def _is_pure_subset(self, dataset):
        sample_classification_value = dataset[self.classification_attribute].iloc[0]
        if (
            dataset.shape[0]
            == dataset.loc[
                dataset[self.classification_attribute] == sample_classification_value
            ].shape[0]
        ):
            return True, sample_classification_value
        return False, None
    
    def _has_attributes(self, dataset) -> bool:
        return dataset.shape[1] > 1

    def _has_instances(self, dataset) -> bool:
        return dataset.shape[0] > 0
    
    def _get_majority_class(self, dataset):
        return dataset[self.classification_attribute].mode().values[0]

    def _get_most_important_attribute(self, dataset):
        attributes = dataset.columns.drop(self.classification_attribute)
        most_important_attribute = attributes[0]
        most_important_attribute_entropy = 1.0
        for attribute in attributes:
            # Using ID3
            grouped_by_values = dataset.groupby(attribute)
            attribute_entropy = 0.0
            for index, subset_values in grouped_by_values:
                entropy = self._calculate_entropy(subset_values)
                attribute_entropy += subset_values.shape[0] / dataset.shape[0] * entropy
            if attribute_entropy < most_important_attribute_entropy:
                most_important_attribute = attribute
                most_important_attribute_entropy = attribute_entropy
        dataset_entropy = self._calculate_entropy(dataset)
        information_gain = round(dataset_entropy - most_important_attribute_entropy, 3)
        return most_important_attribute, information_gain

    def _calculate_entropy(self, subset):
        grouped_by_classification = subset.groupby(self.classification_attribute)
        entropy = 0.0
        for name, subset_grouped_by_classification in grouped_by_classification:
            subset_propability = (
                subset_grouped_by_classification.shape[0] / subset.shape[0]
            )
            entropy += subset_propability * np.log2(1.0 / subset_propability)
        return entropy

    def _get_possible_values_for_categorical_attributes(self, dataset: pd.DataFrame) -> Dict:
        values_for_each_attribute = {}
        for column in dataset:
            values_for_each_attribute[column] = dataset[column].unique()
        return values_for_each_attribute

    def predict(self, instances: pd.DataFrame):
        predictions = []
        number_of_instances = instances.shape[0] # Number of rows in the dataframe
        for i in range(number_of_instances):
            predictions.append(self._walk_to_leaf_node(self.get_node(self.root), instances.iloc[i]))
        return predictions

    def _walk_to_leaf_node(self, node: Node, instance):
        if node.is_leaf():
            return node.decision
        for c in self.children(node.identifier):
            if instance[node.attribute] == c.parent_attribute_value:
                return self._walk_to_leaf_node(c, instance)
