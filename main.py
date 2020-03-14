import pandas as pd
from decision_tree import DecisionTree

dataset = {
    'rain': ['no', 'no', 'no', 'yes', 'no'],
    'hungry': ['yes', 'yes', 'no', 'yes', 'no'],
    'classification': ['false', 'false', 'true', 'true', 'true'],
}
dataframe = pd.DataFrame(data=dataset)
decision_tree = DecisionTree('classification')
decision_tree.construct(dataframe)
decision_tree.show()
