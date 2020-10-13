import pandas as pd
from decision_tree import DecisionTree


def example_dataset():
    # Example dataset
    example_dataset = pd.read_csv('data/example_dataset.csv', sep=';')

    test_dataset = {
        'Tempo': ['Ensolarado', 'Nublado'],
        'Temperatura': ['Quente', 'Quente'],
        'Umidade': ['Alta', 'Alta'],
        'Ventoso': ['Falso', 'Falso'] 
    }

    test_dataframe = pd.DataFrame(data=test_dataset)

    decision_tree = DecisionTree(
        classification_attribute='Joga', 
        attribute_types={
            'Tempo': 'discrete',
            'Temperatura': 'discrete',
            'Umidade': 'discrete',
            'Ventoso': 'discrete' 
        }
    )
    decision_tree.train(example_dataset)
    predictions = decision_tree.predict(test_dataframe)

    print('Example dataset - resultant decision tree:\n')
    decision_tree.show()

    print('Test dataset:')
    print(test_dataframe)

    print(f'\nPredictions: {predictions}\n\n')

def votes_dataset():
    # Votes dataset
    train_dataframe = pd.read_csv('data/house_votes_84.tsv', sep='\t') 

    decision_tree = DecisionTree(
        classification_attribute='target', 
        attribute_types={
            'handicapped-infants' : 'discrete',
            'water-project-cost-sharing' : 'discrete',
            'adoption-of-the-budget-resolution' : 'discrete',
            'physician-fee-freeze' : 'discrete',
            'el-salvador-adi' : 'discrete',
            'religious-groups-in-schools' : 'discrete',
            'anti-satellite-test-ban' : 'discrete',
            'aid-to-nicaraguan-contras'	 : 'discrete',
            'mx-missile' : 'discrete',
            'immigration' : 'discrete',
            'synfuels-corporation-cutback' : 'discrete',
            'education-spending' : 'discrete',
            'superfund-right-to-sue' : 'discrete',
            'crime' : 'discrete',
            'duty-free-exports' : 'discrete',
            'export-administration-act-south-africa' : 'discrete'
        }
    )
    decision_tree.train(train_dataframe)

    print('Votes dataset - resultant decision tree:\n')
    decision_tree.show()

def wine_dataset():
    # Wine dataset
    train_dataframe = pd.read_csv('data/wine_recognition.tsv', sep='\t')

    test_dataframe = pd.DataFrame({    
        '1': [14.23],	
        '2': [1.71],	
        '3': [2.43],	
        '4': [15.6],	
        '5': [127],	
        '6': [2.8],	
        '7': [3.06],	
        '8': [0.28],	
        '9': [2.29],	
        '10': [5.64],	
        '11': [1.04],	
        '12': [3.92],	
        '13': [1065]
    })

    decision_tree = DecisionTree(
        classification_attribute='target', 
        attribute_types={
            '1': 'continuous',	
            '2': 'continuous',	
            '3': 'continuous',	
            '4': 'continuous',	
            '5': 'continuous',	
            '6': 'continuous',	
            '7': 'continuous',	
            '8': 'continuous',	
            '9': 'continuous',	
            '10': 'continuous',	
            '11': 'continuous',	
            '12': 'continuous',	
            '13': 'continuous'
        }
    )
    decision_tree.train(train_dataframe)

    print('Wine dataset - resultant decision tree:\n')
    decision_tree.show()

    print('Predictions:')
    print(decision_tree.predict(test_dataframe))


if __name__ == "__main__":
    example_dataset()