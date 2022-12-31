import pandas as pd

def preprocess(X: pd.DataFrame) -> None:
    """
    Preprocess X in-place.

    This function will be used in "featureEngineer.py" and not "model.py"
    """
    
    # Drop columns with no correlation.
    X.drop(["Name", "PassengerId", "Cabin", "Ticket"], axis=1, inplace=True) 

    # More things to be added following feature engineering