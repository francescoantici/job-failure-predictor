from typing import Literal
from sklearn.neighbors import KNeighborsClassifier

class KNN(KNeighborsClassifier):
    
    """
    The implementation of the KNN algorithm. 
    Based on KNeighborsClassifier of scikit-learn
    """
    
    distance_mapping = {
        "cd" : "cosine", 
        "mwd" : "minkowski"
    }
        
    def __init__(self, distance: Literal["mwd", "cd"] = "mwd") -> None:
        
        super().__init__(metric = self.distance_mapping.get(distance))