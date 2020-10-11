import numpy as np
import pandas as pd

class KFoldCrossValidator:
    def __init__(self, number_of_folds: int, random_state: int = 42, verbose: bool = False):
        self.number_of_folds = number_of_folds # TODO: Needs to have one or more 
        self.random_state = random_state
        self.verbose = verbose

    def validate(self, classifier, data):
        print(data.shape[0])
        generated_folds = self._generate_folds(data, classifier.classification_attribute)
        results = []
        for k in range(self.number_of_folds):
            if (self.verbose): 
                print(f'Current testing fold: {k + 1}')
            test_fold = generated_folds[k]
            train_data = self._append_train_folds(generated_folds, k)
            classifier.train(train_data)
            predicted_labels = classifier.predict(test_fold.drop(columns=[classifier.classification_attribute]))
            expected_labels = list(test_fold[classifier.classification_attribute])
            results.append(
                (
                    k + 1, 
                    self._calculate_accuracy(expected_labels, predicted_labels), 
                    self._calculate_f1_score(expected_labels, predicted_labels)
                )
            )
        return results

    def _append_train_folds(self, folds, current_test_fold_index: int) -> pd.DataFrame:
        train_data = pd.DataFrame(columns=folds[0].columns)
        for i in range(self.number_of_folds):
            if i != current_test_fold_index:
                train_data = train_data.append(folds[i], ignore_index=True)
        return train_data

    def _generate_folds(self, data: pd.DataFrame, classification_attribute):
        data_grouped_by_class = { group_class: group_data for group_class, group_data in data.groupby(classification_attribute) }
        total_number_of_elements = data.shape[0] # Number of rows
        number_of_elements_by_fold = np.ceil(total_number_of_elements / self.number_of_folds)
        number_of_elements_to_sample_by_class = {}

        for group_class, group_data in data_grouped_by_class.items():
            elements_by_group = group_data.shape[0] # Number of rows in the group
            group_proportion = elements_by_group / total_number_of_elements
            number_of_elements_to_sample = int(np.floor(number_of_elements_by_fold * group_proportion))
            number_of_elements_to_sample_by_class[group_class] = number_of_elements_to_sample
            print(group_class, elements_by_group)

        np.random.seed(self.random_state)
        folds = []

        for i in range(self.number_of_folds):
            fold = pd.DataFrame(columns=data.columns)
            number_of_elements_sampled_by_class = {}
            for group_class, group_data in data_grouped_by_class.items():
                group_number_of_rows = group_data.shape[0]
                
                number_of_elements_to_sample = number_of_elements_to_sample_by_class[group_class]
                if (i == self.number_of_folds - 1):
                    number_of_elements_to_sample = group_number_of_rows

                sampled_indexes = np.random.choice(
                    group_number_of_rows, # Number of rows in the group
                    number_of_elements_to_sample, 
                    replace=False
                )

                number_of_elements_sampled_by_class[group_class] = number_of_elements_to_sample
                    
                for j in sampled_indexes:
                    fold = fold.append(group_data.iloc[j], ignore_index=True)
                data_grouped_by_class[group_class] = data_grouped_by_class[group_class].drop(group_data.index[sampled_indexes])

            folds.append(fold)

            if (self.verbose):
                print(f'Fold {i + 1} generated')
                number_of_elements_sampled = sum([v for v in number_of_elements_sampled_by_class.values()])
                for c, n in number_of_elements_sampled_by_class.items():
                    print(f'Percentage for {c}: {n / number_of_elements_sampled}, n: {n}')

        return folds

    def _calculate_accuracy(self, correct_labels: np.array, predicted_labels: np.array):
        number_of_correct_predictions = 0
        for correct_label, predicted_label in zip(correct_labels, predicted_labels):
            if (predicted_label == correct_label):
                number_of_correct_predictions += 1
        return round(number_of_correct_predictions / len(correct_labels), 3)

    def _calculate_f1_score(self, expected_labels, predicted_labels):
        number_of_true_positives = 0
        number_of_false_positives = 0
        number_of_true_negatives = 0
        number_of_false_negatives = 0
        for correct_label, predicted_label in zip(expected_labels, predicted_labels):
            if (predicted_label == 1):
                if (correct_label == predicted_label): 
                    number_of_true_positives += 1
                else:
                    number_of_false_positives += 1
            else:
                if (correct_label == predicted_label): 
                    number_of_true_negatives += 1
                else:
                    number_of_false_negatives += 1
        
        precision = number_of_true_positives / (number_of_true_positives + number_of_false_positives)
        recall = number_of_true_positives / (number_of_true_positives + number_of_false_negatives)
        return round(2 * (precision * recall) / (precision + recall), 3) 