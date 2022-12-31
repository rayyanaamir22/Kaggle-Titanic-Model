# Modules
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

def get_MAE(y_valid: pd.Series, predictions: list[int]) -> float:
    """
    Return the Mean Absolute Error.
    """
    return mean_absolute_error(y_valid, predictions)

def get_CVS(X: pd.DataFrame, y: pd.Series, pipeline: Pipeline, folds=5) -> float:
    """
    (UNFINISHED) Return the Root Mean Squared Error (RMSE) from Cross Validation.

    Preprocess the data, then calculate the cross validation score.
    """
    
    # Define numerical and categorical columns to use
    numericalColumns = [c for c in X.columns if (X[c].dtype in {'int64', 'float64'})]
    categoricalColumns = [c for c in X.columns if (X[c].dtype == "object")]

    # Define preprocessor
    numericalTransformer = SimpleImputer(strategy="median")
    categoricalTransformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('onehot', OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numericalTransformer, numericalColumns),
        ('cat', categoricalTransformer, categoricalColumns)
    ])
    X_transformed = pd.DataFrame(preprocessor.fit_transform(X)) # fit_transform returns an np array
    # The column names are ruined, this doesn't work

    model = pipeline.steps[-1][1] # Access the model from the pipeline
    scores = cross_val_score(model, X_transformed, y, cv=folds, scoring="neg_mean_squared_error")
    score = -1*(scores.mean())
    return np.sqrt(score)