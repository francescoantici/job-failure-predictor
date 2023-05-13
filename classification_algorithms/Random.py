from typing import Literal
from sklearn._typing import ArrayLike, Int
from sklearn.dummy import DummyClassifier 

class RandomBaseline(DummyClassifier): 
    
    baseline_mapping = {
        "majority" : "prior", 
        "random" : "uniform"
    }
    
    def __init__(self, strategy: Literal["majority", "random"] = "majority") -> None:
        super().__init__(strategy=self.baseline_mapping)
    