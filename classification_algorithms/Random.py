from typing import Literal
from sklearn.dummy import DummyClassifier 

class RandomBaseline(DummyClassifier): 
    
    """
    The implementation of the random baselines.
    Based on the DummyClassifier of scikit-learn
    """
    
    baseline_mapping = {
        "majority" : "prior", 
        "random" : "uniform"
    }
    
    def __init__(self, strategy: Literal["majority", "random"] = "majority") -> None:
        super().__init__(strategy=self.baseline_mapping)
    