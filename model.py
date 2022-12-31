# Modules
import pandas as pd
from sklearn.model_selection import train_test_split

# Directory
from featureEngineering import featureEngineer
from scoring import get_MAE, get_CVS

# STEP 1: LOAD DATA

# Read in the data
X = pd.read_csv("/Users/rayyanaamir/Desktop/Code/Kaggle/Competitions/Titanic/train.csv")

# Isolate target y, drop y from X
X.dropna(axis=0, subset=["Survived"], inplace=True)
y = X.Survived
X.drop(["Survived"], axis=1, inplace=True)

# STEP 2: Preprocessing

featureEngineer(X)

# Break off training and validation data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=16)

# STEP 3:

# Build model
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

# Feature transformation
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Define numerical and categorical columns to use
numericalColumns = [c for c in X_train.columns if (X_train[c].dtype in {'int64', 'float64'})]
categoricalColumns = [c for c in X_train.columns if (X_train[c].dtype == "object")]

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

# Define pipeline
myPipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(n_estimators=200, learning_rate=0.05))
])

# Fit training data, make predictions, get MAE
myPipeline.fit(X_train, y_train)
predictions = myPipeline.predict(X_valid)

if __name__ == "__main__":
    # Results
    print(f"\nMean Absolute Error: {get_MAE(y_valid, predictions)}\n")
    #print(f"\nCross Validation Score: {get_CVS(X, y, myPipeline)}\n")