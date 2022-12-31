# Discern abstract relationships and modify training data when necessary

# Modules
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Directory
from preprocessing import preprocess

def featureEngineer(X: pd.DataFrame) -> None:
    """
    Make the modifications to X in-place.
    """
    preprocess(X)