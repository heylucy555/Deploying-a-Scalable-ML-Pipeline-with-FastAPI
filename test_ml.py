import os
import pandas as pd
import pytest
from ml import data, model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#load data
@pytest.fixture(scope="session")
def df():
    data_path = os.path.join(os.getcwd(), "data", "census.csv")
    return pd.read_csv(data_path)

#test 1: ensure the data is splitting correctly
def test_split(df):
    """
    # ensures the training and test sets have the appropriate size 
    """
    train, test = train_test_split(df, test_size=0.2, random_state=69)
    
    assert len(train) + len(test) == len(df)
    assert len(test) > 0
    assert len(train) > 0

#test 2 ensure the correct classifier is used
def test_clf(df):
    X_train = np.random.rand(100, 25)
    y_train = np.random.randint(2, size=100)

    #Training the model
    model_chk = model.train_model(X_train, y_train)

    #Check model type
    assert isinstance(model_chk, LogisticRegression)

#test 3: datset size
def test_dataset_shape(df):
    """
    # ensure the working dataset is large enough and has the right amount of functions
    """
 
    assert df.shape[0] > 1000
    assert df.shape[1] > 10
