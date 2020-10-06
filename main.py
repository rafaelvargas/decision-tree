import pandas as pd
from decision_tree import DecisionTree

train_dataset = pd.read_csv('test_dataset.csv', sep=';')

test_dataset = {
    'Tempo': ['Ensolarado', 'Nublado'],
    'Temperatura': ['Quente', 'Quente'],
    'Umidade': ['Alta', 'Alta'],
    'Ventoso': ['Falso', 'Falso'] 
}

train_dataframe = pd.DataFrame(data=train_dataset)
test_dataframe = pd.DataFrame(data=test_dataset)

decision_tree = DecisionTree(classification_attribute='Joga')
decision_tree.construct(train_dataset)
predictions = decision_tree.predict(test_dataframe)
decision_tree.show()
print(predictions)
