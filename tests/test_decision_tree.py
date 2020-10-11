import json
import pandas as pd

from decision_tree import DecisionTree


def test_resultant_tree():
    example_dataset = pd.read_csv('data/example_dataset.csv', sep=';')
    decision_tree = DecisionTree(classification_attribute='Joga')
    decision_tree.train(example_dataset)
    expected_tree = json.dumps({
        "('Tempo', 0.247)": {
            "children": [
                {
                    "('Chuvoso', 'Ventoso', 0.971)": {
                        "children": [
                            "('Falso', 'Sim')",
                            "('Verdadeiro', 'Nao')"
                        ]
                    }
                },
                {
                    "('Ensolarado', 'Umidade', 0.971)": {
                        "children": [
                            "('Alta', 'Nao')",
                            "('Normal', 'Sim')"
                        ]
                    }
                },
                "('Nublado', 'Sim')"
            ]
        }
    })

    assert decision_tree.to_json() == expected_tree