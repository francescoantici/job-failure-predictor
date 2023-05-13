from typing import Iterable
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
from sentence_transformers import SentenceTransformer

class IntEncoder:
    """
    An interface for the INT encoder 
    """
    @staticmethod
    def encode(train_data: pd.DataFrame, test_data:  pd.DataFrame) -> tuple:
        """_summary_

        Args:
            train_data (pd.DataFrame): _description_
            test_data (pd.DataFrame): _description_

        Returns:
            tuple: _description_
        """
        for col in train_data.columns:
            _encoder = OrdinalEncoder()
            train_data[col] = _encoder.fit_transform(train_data.col)
            test_data[col] = _encoder.transform(test_data.col)
        return train_data, test_data
        
class SbEncoder:
    """_summary_
    """
    @classmethod
    def encode(cls, train_data: Iterable, test_data: Iterable) -> Iterable:
        """_summary_

        Args:
            train_data (Iterable): _description_
            test_data (Iterable): _description_

        Returns:
            Iterable: _description_
        """
        _encoder = SentenceTransformer("all-MiniLM-L6-v2") 
        encoded_train = _encoder.encode(train_data.apply(cls._convert_entry_to_str, axis = 1).values)
        encoded_test = _encoder.encode(test_data.apply(cls._convert_entry_to_str, axis = 1).values)
        return encoded_train, encoded_test
    
    @staticmethod
    def _convert_entry_to_str(job_entry):
        """_summary_

        Args:
            job_entry (_type_): _description_

        Returns:
            _type_: _description_
        """
        return ",".join([f"{job_entry[k]}" for k in job_entry.index if (job_entry[k] and not (pd.isna(job_entry[k])))])

encoders = {
    "int" : IntEncoder,
    "sb": SbEncoder
}