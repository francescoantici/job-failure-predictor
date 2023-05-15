from typing import Iterable, Literal
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
from sentence_transformers import SentenceTransformer

class IntEncoder:
    """
    The implementation of the INT encoding algorithm. 
    Uses the OrdinalEncoder of scikit-learn
    """
    
    def __init__(self) -> None:
        self.encoder = OrdinalEncoder
    
    def encode(self, data: pd.DataFrame) -> Iterable:
        """_summary_

        Args:
            train_data (Iterable): The dataframe containing the job features to encode

        Returns:
            Iterable: The INT encoding of the dataframes passed as input
        """
        for col in data.columns:
            _encoder = self.encoder()
            data[col] = _encoder.fit_transform(data[col].values.reshape(-1, 1))
        return data
        
class SbEncoder:
    """
    The implementation of the SB encoding algorithm.
    Uses the all-MiniLM-L6-v2 pre-trained model of the SentenceTransformer library
    """
    
    def __init__(self) -> None:
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        
    def encode(self, data: Iterable) -> Iterable:
        """_summary_

        Args:
            train_data (Iterable): The dataframe containing the job features to encode

        Returns:
            Iterable: The SB encoding of the dataframes passed as input
        """
         
        encoded_data= self.encoder.encode(data.apply(self._convert_entry_to_str, axis = 1).values)
        return encoded_data
    
    @staticmethod
    def _convert_entry_to_str(job_entry:Iterable) -> str:
        """Internal function to map the job features to a single string

        Args:
            job_entry (Iterable): The job features as Iterable

        Returns:
            str: The string created from the concatenation of the job features
        """
        return ",".join([f"{job_entry[k]}" for k in job_entry.index if (job_entry[k] and not (pd.isna(job_entry[k])))])

encoders = {
            "int" : IntEncoder,
            "sb": SbEncoder
        }

def get_encoding_algorithm(encoding:str):
    """_summary_
    
    Returns the encoder model corresponding to the encoding parameters passed, sb -> SbEncoder, int -> IntEncoder

    Args:
        encoding (str): _description_

    Raises:
        Exception: If the name of the encoding is not correct

    Returns:
        encoder: The encoder model selected
    """
    try:
        return encoders.get(encoding)()
    except:
        raise Exception("Wrong encoding algorithm inserted, please insert 'sb' or 'int'.")
