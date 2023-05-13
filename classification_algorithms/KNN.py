from typing import Callable, Literal
from sklearn._typing import Int
from sklearn.neighbors import KNeighborsClassifier

class KNN(KNeighborsClassifier):
    
    distance_mapping = {
        "cd" : "cosine", 
        "mwd" : "minkowski"
    }
        
    def __init__(self, distance: Literal["mwd", "cd"] = "mwd") -> None:
        
        super().__init__(metric = self.distance_mapping.get(distance))