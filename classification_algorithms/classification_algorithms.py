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
    """
    Returns the classification algorithm implementation corresponding to the model_name parameters passed.

    Args:
        model_name (str, optional): The name of the classification algorithm to use. Defaults to "dt".

    Raises:
        Exception: If the model_name is not among the pre-defined ones.

    Returns:
        _type_: The implementation of the classification algorithm requested.
    """
    try:
        return classification_algorithms.get(model_name)(**args)
    except:
        raise Exception("Model name inserted not correct! Please insert 'dt', 'lr', 'rf', 'cd' or 'mwd'")

