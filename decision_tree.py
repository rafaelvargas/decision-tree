from treelib import Node, Tree
import numpy as np


class DecisionTreeNode(Node):
    def __init__(self, attribute=None, parent_attribute_value=None, decision=None):
        if decision:
            super().__init__(tag=str((parent_attribute_value, decision)))
        elif not parent_attribute_value:
            super().__init__(tag=str(attribute))
        else:
            super().__init__(tag=str((parent_attribute_value, attribute)))
        self.parent_attribute_value = parent_attribute_value
        self.attribute = attribute
        self.decision = decision


class DecisionTree(Tree):
    def __init__(self, classification_attribute):
        super().__init__()
        self.classification_attribute = classification_attribute

    def construct(self, dataset, parent=None, parent_attribute_value=None):
        have_same_classification, classification = self._have_same_classification(
            dataset
        )
        if have_same_classification:
            self.add_node(DecisionTreeNode(decision=classification, parent_attribute_value=parent_attribute_value), parent=parent)
        else:
            most_important_attribute = self._get_most_important_attribute(dataset)
            current_node = DecisionTreeNode(attribute=most_important_attribute, parent_attribute_value=parent_attribute_value)
            attribute_values = self._get_attribute_values(
                dataset, most_important_attribute
            )
            if parent:
                self.add_node(current_node, parent=parent)
            else:
                self.add_node(current_node)
            for value in attribute_values:
                data_subset = dataset.loc[dataset[most_important_attribute] == value]
                self.construct(data_subset.drop(columns=[most_important_attribute]), current_node, parent_attribute_value=value)

    def evaluate(self, instance):
        pass

    def _have_same_classification(self, dataset):
        sample_classification_value = dataset[self.classification_attribute].iloc[0]
        if (
            dataset.shape[0]
            == dataset.loc[
                dataset[self.classification_attribute] == sample_classification_value
            ].shape[0]
        ):
            return True, sample_classification_value
        return False, None

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
        print('Attribute: {} / Entropy: {}'.format(most_important_attribute, most_important_attribute_entropy))
        return most_important_attribute

    def _calculate_entropy(self, subset):
        grouped_by_classification = subset.groupby(self.classification_attribute)
        entropy = 0.0
        for name, subset_grouped_by_classification in grouped_by_classification:
            subset_propability = (
                subset_grouped_by_classification.shape[0] / subset.shape[0]
            )
            entropy += subset_propability * np.log2(1.0 / subset_propability)
        return entropy

    def _get_attribute_values(self, dataset, attribute):
        return dataset[attribute].unique()
