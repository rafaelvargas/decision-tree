import pandas as pd
from decision_tree import DecisionTree

dataset = {
    'hungry': ['yes', 'yes', 'no', 'yes', 'no'],
    'raining': ['no', 'no', 'no', 'yes', 'no'],
    'classification': ['true', 'true', 'false', 'false', 'false'],
}
dataframe = pd.DataFrame(data=dataset)
decision_tree = DecisionTree('classification')
decision_tree.construct(dataframe)
decision_tree.show()
