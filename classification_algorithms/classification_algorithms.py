from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from classification_algorithms.KNN import KNN
from classification_algorithms.Random import RandomBaseline

classification_algorithms = {
    
    "dt" : DecisionTreeClassifier, 
    "lr" : LogisticRegression,
    "rf" : RandomForestClassifier,
    "knn" : KNN,
    "random" : RandomBaseline
    
}

def get_classification_algoritm(model_name: str = "dt", **args):
    try:
        return classification_algorithms.get(model_name)(**args)
    except Exception as e:
        print("Model name inserted not correct!")

