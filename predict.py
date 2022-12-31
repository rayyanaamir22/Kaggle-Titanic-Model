# Use the model to predict whether you would survive the Titanic or not.
# This hasn't been tested yet

# Modules
import pandas as pd
import os # For clearing screen
from typing import Union

# Directory
from model import myPipeline as myModel

def outputMessage(result: list[int]) -> bool:
    """
    Prettify the results for logging.
    """
    if (result[0]):
        return f"\nCongrats! You survived!\n"
    return f"\nSorry, you didn't make it :(\n"

def getPassengerData() -> dict[str, Union(str, int, float)]:
    """
    Return a dictionary representing the test subject that can be parsed by the model.
    The user must input the data according to the console prompts.
    """

    # Run a while loop to force input for each field, add each field to the dict testSubject
    testSubject = {}

    # Class
    while True:
        print("Enter seating class (1, 2, 3): ")
        try:
            cls = int(input()) # Potential TypeError
            if (cls in {1, 2, 3}): # Validate
                break
            raise TypeError
        except TypeError:
            os.system("clear")
            print("Invalid input.")
    testSubject["Pclass"] = [cls]

    # Gender
    os.system("clear")
    while True:
        print(f"Live Info: {testSubject}")
        print("Enter sex (M, F):")
        sex = input().upper()
        if (sex=="M"):
            sex = "male"
            break
        elif (sex=="F"):
            sex = "female"
            break
        else:
            os.system("clear")
            print("Invalid input.")
    testSubject["Sex"] = [sex]

    # Age
    os.system("clear")
    while True:
        print(f"Live Info: {testSubject}")
        print("Enter age:")
        try:
            age = int(input())
            if (age>=0): # Validate
                break
            raise TypeError
        except TypeError:
            os.system("clear")
            print("Invalid input.")
    testSubject["Age"] = [age]

    # Siblings and Spouse
    os.system("clear")
    while True:
        print(f"Live Info: {testSubject}")
        print("Enter total amount of siblings and spouse:")
        try:
            sibSp = int(input())
            if (sibSp>=0): # Validate
                break
            raise TypeError
        except TypeError:
            os.system("clear")
            print("Invalid input.")
    testSubject["SibSp"] = [sibSp]

    # Parents and Children
    os.system("clear")
    while True:
        print(f"Live Info: {testSubject}")
        print("Enter total amount of siblings and spouse:")
        try:
            parCh = int(input())
            if (parCh>=0): # Validate
                break
            raise TypeError
        except TypeError:
            os.system("clear")
            print("Invalid input.")
    testSubject["Parch"] = [parCh]

    # Fare
    os.system("clear")
    while True:
        print(f"Live Info: {testSubject}")
        print("Enter fare:")
        try:
            fare = float(input())
            if (fare>=0):
                break
            raise TypeError
        except TypeError:
            os.system("clear")
            print("Invalid input.")
    testSubject["Fare"] = [fare]

    # Embarked
    os.system("clear")
    while True:
        print(f"Live Info: {testSubject}")
        print("Enter where you embarked from: (S for Southampton, C for Cherbourg, Q for Queenstown)")
        embarked = input().upper()
        if (embarked in {"S", "C", "Q"}):
            break
        os.system("clear")
        print("Invalid input.")
    testSubject["Embarked"] = [embarked]  

    return testSubject

if __name__ == "__main__":
    testSubject = pd.DataFrame(getPassengerData())
    result = myModel.predict(testSubject)

    os.system("clear")
    print(f"""
    Titanic Result:
    {outputMessage(result)}
    """)