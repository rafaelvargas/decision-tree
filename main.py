import pandas as pd
from decision_tree import DecisionTree

dataset = {
    'hungry': ['yes', 'yes', 'no', 'yes', 'no'],
    'raining': ['no', 'no', 'no', 'yes', 'no'],
    'classification': ['true', 'true', 'false', 'false', 'false'],
}

test_dataset = {
    'tempo': ['ensolarado', 'ensolarado', 'nublado', 'chuvoso', 'chuvoso', 'chuvoso', 'nublado', 'ensolarado', 'ensolarado', 'chuvoso', 'ensolarado', 'nublado', 'nublado', 'chuvoso'],
    'temperatura': ['quente', 'quente', 'quente', 'amena', 'fria', 'fria', 'fria', 'amena', 'fria', 'amena', 'amena', 'amena', 'quente', 'amena'],
    'umidade': ['alta', 'alta', 'alta', 'alta', 'normal', 'normal', 'normal', 'alta', 'normal', 'normal', 'normal', 'alta', 'normal', 'alta'],
    'ventoso': ['falso', 'verdadeiro', 'falso', 'falso', 'falso', 'verdadeiro', 'verdadeiro', 'falso', 'falso', 'falso', 'verdadeiro', 'verdadeiro', 'falso', 'verdadeiro'],
    'joga': ['nao', 'nao', 'sim', 'sim', 'sim', 'nao', 'sim', 'nao', 'sim', 'sim', 'sim', 'sim', 'sim', 'nao'] 
}

dataframe = pd.DataFrame(data=test_dataset)
decision_tree = DecisionTree('joga')
decision_tree.construct(dataframe)
decision_tree.show()
