import pandas as pd 
from Predictor import Predictor

if __name__ == "__main__":
    
    # Setup setting for the experiment 
    encoding = "sb"
    
    classification_algorithm = "mwd"
        
    test_size = 0.3 
    
    # Load the mockup data
    # The data is a subsample of 100 jobs to test the models 
    # The string features are anonimized by being replaced with random strings 
    # Columns of the subset which contained all null values are removed
    df = pd.read_csv("job_table_mockup.csv")
    
    # Instantiate the predictor interface to automatically load the encoding and classification algorithms
    predictor = Predictor(encoding=encoding, classification_algorithm=classification_algorithm)
        
    # Data split and pre-paration
    # The data is already ordered by submission time
    test_idx_start = int(len(df)*(1 - test_size))
    
    train_df = df.iloc[:test_idx_start]
    
    test_df = df.iloc[test_idx_start:]
    
    train_data = train_df.drop("job_state", axis = 1)
    
    test_data = test_df.drop("job_state", axis = 1)
    
    train_labels = train_df["job_state"].values 
    
    test_labels = test_df["job_state"].values
    
    # Training of the model
    predictor.fit(train_data=train_data, train_labels=train_labels)
    
    # Inference of the model
    preds = predictor.predict(test_data)
        
    
    
    
    
    
    
    
    
    
    
    
    