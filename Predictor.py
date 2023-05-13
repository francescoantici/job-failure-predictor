from typing import Iterable
from encoders.encoders import encoders
from classification_algorithms.classification_algorithms import get_classification_algoritm

class Predictor:
    """
    This class is a proxy the configuration of the prediction algorithm
    """
    
    def __init__(self, encoding: str = "int", classification_algorithm: str = "dt") -> None:
        self.encoder = encoders.get(encoding.lower())
        classification_algorithm = classification_algorithm.lower()
        if classification_algorithm in ["cd", "mwd"]:
            self.classifier = get_classification_algoritm("knn", distance = classification_algorithm)
        elif classification_algorithm in ["random", "majority"]:
            self.classifier = get_classification_algoritm("random", strategy = classification_algorithm)
        else:
            self.classifier = get_classification_algoritm(classification_algorithm)
    
    def fit(self, train_data: Iterable, test_data: Iterable) -> None:
        """_summary_

        Args:
            train_data (Iterable): _description_
            test_data (Iterable): _description_
        """
        pass 
    
    def predict(self, data: Iterable) -> Iterable:
        """_summary_

        Args:
            data (Iterable): _description_

        Returns:
            Iterable: _description_
        """
        pass 
    
    