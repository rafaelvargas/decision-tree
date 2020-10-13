from typing import Dict
import math

from treelib import Node, Tree
import numpy as np
import pandas as pd


class DecisionTreeNode(Node):
    def __init__(self, attribute=None, parent_attribute_value=None, division_criterion = None, decision=None, information_gain=None):
        if decision is not None:
            super().__init__(tag=str((parent_attribute_value, decision)))
        elif parent_attribute_value is None:
            super().__init__(tag=str((attribute, information_gain)))
        else:
            super().__init__(tag=str(((parent_attribute_value, attribute, information_gain))))
        self.parent_attribute_value = parent_attribute_value
        self.attribute = attribute
        self.decision = decision
        self.division_criterion = division_criterion


class DecisionTree(Tree):
    def __init__(
        self, 
        classification_attribute: str, 
        attribute_types: Dict[str, str],
        possible_values_for_categorical_attributes: Dict = None,
        use_feature_bagging: bool = False,
        random_state: int = 42
    ):
        super().__init__()
        self.classification_attribute = classification_attribute
        self.attribute_types = attribute_types
        self.possible_values_for_categorical_attributes = possible_values_for_categorical_attributes
        self.use_feature_bagging = use_feature_bagging
        self.random_state = random_state

    def train(self, dataset):
        if (self.possible_values_for_categorical_attributes is None):
            self.possible_values_for_categorical_attributes = self._get_possible_values_for_categorical_attributes(dataset)
        if (self.use_feature_bagging):
            self.feature_bag_size = math.isqrt(dataset.shape[1] - 1)
        else:
            self.feature_bag_size = dataset.shape[1] - 1
        self.construct(dataset, subset=dataset)
    
    def construct(self, dataset, subset=None, parent=None, parent_attribute_value=None, division_criterion=None):
        is_pure_subset, classification = self._is_pure_subset(subset)
        if is_pure_subset:
            self.add_node(
                DecisionTreeNode(
                    decision=classification,
                    parent_attribute_value=parent_attribute_value,
                    division_criterion=division_criterion
                ),
                parent=parent
            )
        elif not self._has_attributes(subset):
            self.add_node(
                DecisionTreeNode(
                    decision=self._get_majority_class(subset),
                    parent_attribute_value=parent_attribute_value,
                    division_criterion=division_criterion
                ),
                parent=parent
            )
        else:
            most_important_attribute, information_gain, splitting_criterion = self._get_most_important_attribute(subset)
            current_node = DecisionTreeNode(
                attribute=most_important_attribute,
                parent_attribute_value=parent_attribute_value,
                information_gain=information_gain,
                division_criterion=division_criterion
            )

            if parent:
                self.add_node(current_node, parent=parent)
            else:
                self.add_node(current_node) # Root

            if (self.attribute_types[most_important_attribute] == 'discrete'):
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
            else:
                # Continuous attribute
                left = subset[subset[most_important_attribute] <= splitting_criterion]
                right = subset[subset[most_important_attribute] > splitting_criterion]

                # Left
                if not self._has_instances(left): # When the new subset has no instances
                    self.add_node(
                        DecisionTreeNode(
                            decision=self._get_majority_class(subset),
                            parent_attribute_value=f'<= {splitting_criterion}',
                            division_criterion=(lambda x : x <= splitting_criterion)
                        ),
                        parent=current_node
                    )
                else:      
                    self.construct(
                        dataset=dataset,
                        subset=left.drop(columns=[most_important_attribute]),
                        parent=current_node,
                        parent_attribute_value=f'<= {splitting_criterion}',
                        division_criterion=(lambda x : x <= splitting_criterion)
                    )
                
                # Right
                if not self._has_instances(right): # When the new subset has no instances
                    self.add_node(
                        DecisionTreeNode(
                            decision=self._get_majority_class(subset),
                            parent_attribute_value=f'> {splitting_criterion}',
                            division_criterion=(lambda x : x > splitting_criterion)
                        ),
                        parent=current_node
                    )
                else:      
                    self.construct(
                        dataset=dataset,
                        subset=right.drop(columns=[most_important_attribute]),
                        parent=current_node,
                        parent_attribute_value=f'> {splitting_criterion}',
                        division_criterion=(lambda x : x > splitting_criterion)
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
        if (self.use_feature_bagging):
            attributes = self._sample_attributes(attributes.to_series())
        most_important_attribute = attributes[0]
        most_important_attribute_entropy = 1.0
        best_splitting_criterion = None
        for attribute in attributes:
            attribute_entropy = 0.0
            splitting_criterion = None
            if (self.attribute_types[attribute] == 'discrete'):
                # Using ID3
                grouped_by_attribute_values = dataset.groupby(attribute)
                for index, subset_values in grouped_by_attribute_values:
                    entropy = self._calculate_entropy(subset_values)
                    attribute_entropy += subset_values.shape[0] / dataset.shape[0] * entropy
            else:
                # Using C4.5
                splitting_criterion, attribute_entropy = self._calculate_entropy_continuous_attributes(dataset, attribute)
            if attribute_entropy < most_important_attribute_entropy:
                most_important_attribute = attribute
                most_important_attribute_entropy = attribute_entropy
                best_splitting_criterion = splitting_criterion
        dataset_entropy = self._calculate_entropy(dataset)
        information_gain = round(dataset_entropy - most_important_attribute_entropy, 3)
        return most_important_attribute, information_gain, best_splitting_criterion

    def _calculate_entropy(self, subset):
        grouped_by_classification = subset.groupby(self.classification_attribute)
        entropy = 0.0
        for name, subset_grouped_by_classification in grouped_by_classification:
            subset_propability = (
                subset_grouped_by_classification.shape[0] / subset.shape[0]
            )
            entropy += subset_propability * np.log2(1.0 / subset_propability)
        return entropy

    def _calculate_entropy_continuous_attributes(self, dataset: pd.DataFrame, attribute: str):
        sorted_instances = dataset.sort_values(attribute)
        best_splitting_criterion = (sorted_instances.iloc[0][attribute] + sorted_instances.iloc[1][attribute]) / 2.0
        
        left = sorted_instances[sorted_instances[attribute] <= best_splitting_criterion]
        right = sorted_instances[sorted_instances[attribute] > best_splitting_criterion]
        left_entropy = self._calculate_entropy(left)
        right_entropy = self._calculate_entropy(right)
        best_split_entropy = ((left.shape[0] / dataset.shape[0]) * left_entropy) + ((right.shape[0] / dataset.shape[0]) * right_entropy)

        for i in range(1, sorted_instances.shape[0] - 1):
            splitting_criterion = (sorted_instances.iloc[i][attribute] + sorted_instances.iloc[i + 1][attribute]) / 2.0
            left = sorted_instances[sorted_instances[attribute] <= splitting_criterion]
            right = sorted_instances[sorted_instances[attribute] > splitting_criterion]
            left_entropy = self._calculate_entropy(left)
            right_entropy = self._calculate_entropy(right)
            current_split_entropy = ((left.shape[0] / dataset.shape[0]) * left_entropy) + ((right.shape[0] / dataset.shape[0]) * right_entropy)
            if (current_split_entropy < best_split_entropy):
                best_split_entropy = current_split_entropy
                best_splitting_criterion = splitting_criterion
        return best_splitting_criterion, best_split_entropy

    def _get_possible_values_for_categorical_attributes(self, dataset: pd.DataFrame) -> Dict:
        values_for_each_attribute = {}
        for column in dataset:
            values_for_each_attribute[column] = dataset[column].unique()
        return values_for_each_attribute
    
    def _sample_attributes(self, attributes: pd.Series):
        number_of_attributes_to_sample = self.feature_bag_size
        if (self.feature_bag_size > len(attributes)):
            number_of_attributes_to_sample = len(attributes)
        sampled_attributes = attributes.sample(
            n=number_of_attributes_to_sample, 
            replace=False, 
            random_state=self.random_state
        )
        return sampled_attributes

    def predict(self, instances: pd.DataFrame):
        predictions = []
        number_of_instances = instances.shape[0] # Number of rows in the dataframe
        for i in range(number_of_instances):
            predictions.append(self._walk_to_leaf_node(self.get_node(self.root), instances.iloc[i]))
        return predictions

    def _walk_to_leaf_node(self, node: Node, instance):
        if (node.is_leaf()):
            return node.decision
        if (self.attribute_types[node.attribute] == 'discrete'):
            for c in self.children(node.identifier):
                if (instance[node.attribute] == c.parent_attribute_value):
                    return self._walk_to_leaf_node(c, instance)
        else:
            for c in self.children(node.identifier):
                if (c.division_criterion(instance[node.attribute])):
                    return self._walk_to_leaf_node(c, instance)