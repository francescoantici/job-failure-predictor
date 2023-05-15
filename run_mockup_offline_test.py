import pandas as pd 
from Predictor import Predictor

if __name__ == "__main__":
    
    # Load the mockup data
    # The data is a subsample of 100 jobs to test the models 
    # The string features are anonimized by being replaced with random strings 
    # Columns of the subset which contained all null values are removed
    df = pd.read_csv("job_table_mockup.csv")
    
    # Instantiate the predictor interface to automatically load the encoding and classification algorithms
    predictor = Predictor(encoding="sb", classification_algorithm="dt")
    
    # We set the test set size to the 30%
    test_size = 0.3 
    
    # Data split and pre-paration
    # The data is already ordered by submission time
    test_idx_start = int(len(df)*(1 - test_size))
    
    train_df = df.iloc[:test_idx_start]
    
    test_df = df.iloc[test_idx_start:]
    
    x_train = train_df.drop("job_state", axis = 1)
    
    x_test = test_df.drop("job_state", axis = 1)
    
    y_train = train_df["job_state"].values 
    
    y_test = test_df["job_state"].values
    
    # Training of the model
    predictor.fit(x_train=x_train, y_train=y_train)
    
    # Inference of the model
    preds = predictor.predict(x_test)
    
    
    
    
    
    
    
    
    
    
    
    
    