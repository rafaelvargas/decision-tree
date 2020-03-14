# Decision trees

Implementation of an algorithm for generating decision trees to be used as predictive models.

## Installing the requirements
```
pip3 install -r requirements.txt
```

## Running
```
python3 main.py
```

## Example

Consider the following dataset:

| hungry | raining | willEat |
|--------|---------|---------|
| yes    | no      | true    |
| yes    | no      | true    |
| no     | no      | false   |
| yes    | yes     | false   |
| no     | no      | false   |


The idea is to construct a decision tree based on a training dataset. In this case, it creates a model to predict if a person is going to eat in a restaurant or not. An example of a constructed tree is given right below.

```
hungry
├── ('no', 'false')
└── ('yes', 'raining')
    ├── ('no', 'true')
    └── ('yes', 'false')

```
Note that, when the person is hungry, a new test (or node) is created to check if it is raining. If it is, it predicts the person will not eat at the restaurant, otherwise, it predicts she will.

In the case the person is not hungry, it simply assumes she'll not eat at the restaurant.

