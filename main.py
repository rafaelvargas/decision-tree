import pandas as pd
from decision_tree import DecisionTree

# Example dataset
example_dataset = pd.read_csv('data/example_dataset.csv', sep=';')

test_dataset = {
    'Tempo': ['Ensolarado', 'Nublado'],
    'Temperatura': ['Quente', 'Quente'],
    'Umidade': ['Alta', 'Alta'],
    'Ventoso': ['Falso', 'Falso'] 
}

train_dataframe = pd.DataFrame(data=example_dataset)
test_dataframe = pd.DataFrame(data=test_dataset)

decision_tree = DecisionTree(classification_attribute='Joga')
decision_tree.train(example_dataset)
predictions = decision_tree.predict(test_dataframe)

print('Example dataset - resultant decision tree:\n')
decision_tree.show()

print('Test dataset:')
print(test_dataframe)

print(f'\nPredictions: {predictions}\n\n')


# Votes dataset
train_dataframe = pd.read_csv('data/house_votes_84.tsv', sep='\t') 

decision_tree = DecisionTree(classification_attribute='target')
decision_tree.train(train_dataframe)

print('Votes dataset - resultant decision tree:\n')
decision_tree.show()