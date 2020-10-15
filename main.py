import pandas as pd
from decision_tree import DecisionTree
import sys


def benchmark_dataset():
    benchmark_dataset = pd.read_csv('data/benchmark_dataset.tsv', sep='\t')
    test_dataset = pd.DataFrame(data={
        'Tempo': ['Ensolarado', 'Nublado'],
        'Temperatura': ['Quente', 'Quente'],
        'Umidade': ['Alta', 'Alta'],
        'Ventoso': ['Falso', 'Falso'] 
    })

    decision_tree = DecisionTree(
        classification_attribute='Joga', 
        attribute_types={
            'Tempo': 'discrete',
            'Temperatura': 'discrete',
            'Umidade': 'discrete',
            'Ventoso': 'discrete' 
        }
    )

    decision_tree.train(benchmark_dataset)
    predictions = decision_tree.predict(test_dataset)

    sys.stdout.write('\x1b[1;34m' + 'Benchmark dataset - resultant decision tree:' + '\x1b[0m' + '\n\n')
    decision_tree.show()

    print('Test dataset:')
    print(test_dataset)

    print(f'\nPredictions: {predictions}\n\n')

def votes_dataset():
    # Votes dataset
    train_dataframe = pd.read_csv('data/house_votes_84.tsv', sep='\t') 
    test_dataset = pd.DataFrame(data={
            'handicapped-infants' : [1],
            'water-project-cost-sharing' : [2],
            'adoption-of-the-budget-resolution' : [1],
            'physician-fee-freeze' : [2],
            'el-salvador-adi' : [2],
            'religious-groups-in-schools' : [2],
            'anti-satellite-test-ban' : [1],
            'aid-to-nicaraguan-contras'	 : [1],
            'mx-missile' : [1],
            'immigration' : [2],
            'synfuels-corporation-cutback' : [0],
            'education-spending' : [2],
            'superfund-right-to-sue' : [2],
            'crime' : [2],
            'duty-free-exports' : [1],
            'export-administration-act-south-africa' : [2]
    })

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

    sys.stdout.write('\x1b[1;34m' + 'Votes dataset - resultant decision tree:' + '\x1b[0m' + '\n\n')
    
    decision_tree.show()

    print('Test dataset:')
    print(test_dataset)

    print(f'\nPredictions: {decision_tree.predict(test_dataset)}\n\n')
    

def wine_dataset():
    # Wine dataset
    wine_dataset = pd.read_csv('data/wine_recognition.tsv', sep='\t')

    test_dataset = pd.DataFrame({    
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
    decision_tree.train(wine_dataset)

    sys.stdout.write('\x1b[1;34m' + 'Wine dataset - resultant decision tree:' + '\x1b[0m' + '\n\n')
    decision_tree.show()

    print('Test dataset:')
    print(test_dataset)
    
    print(f'\nPredictions: {decision_tree.predict(test_dataset)}\n\n')



if __name__ == "__main__":
    benchmark_dataset()
    votes_dataset()
    wine_dataset()