from typing import Iterable
from encoders.encoders import get_encoding_algorithm
from classification_algorithms.classification_algorithms import get_classification_algoritm

class Predictor:
    """
    This class is a proxy the configuration of the prediction algorithm
    """
    
    def __init__(self, encoding: str = "int", classification_algorithm: str = "dt") -> None:
        self.encoder = get_encoding_algorithm(encoding.lower())
        classification_algorithm = classification_algorithm.lower()
        if classification_algorithm in ["cd", "mwd"]:
            self.classifier = get_classification_algoritm("knn", distance = classification_algorithm)
        elif classification_algorithm in ["random", "majority"]:
            self.classifier = get_classification_algoritm("random", strategy = classification_algorithm)
        else:
            self.classifier = get_classification_algoritm(classification_algorithm)
    
    def fit(self, train_data: Iterable, train_labels: Iterable) -> None:
        """_summary_

        Args:
            train_data (Iterable): The dataframe containing the job features 
            train_labels (Iterable): The dataframe containing the job labels 
            
        """
        x_train_enc = self.encoder.encode(train_data)
        self.classifier = self.classifier.fit(x_train_enc, train_labels)        
        
    
    def predict(self, data: Iterable) -> Iterable:
        """_summary_

        Args:
            data (Iterable): The dataframe containing the job features

        Returns:
            Iterable: The labels predicted with the model of the predictor
        """
        x_test_enc = self.encoder.encode(data)
        return self.classifier.predict(x_test_enc)
    
    
    