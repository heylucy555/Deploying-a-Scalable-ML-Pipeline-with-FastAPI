import os
import pandas as pd

# TODO: implement the first test. Change the function name and input as needed

t_path = os.getcwd()
data_path = os.path.join(t_path, "data", "census.csv")
print(data_path)
df = pd.read_csv(data_path)


def test_one():
    """
    # ensures the training and test sets have the appropriate size 
    """
    train, test = train_test_split(data, test_size=0.2, random_state=69)
    
    assert len(train) + len(test) == len(df)
    assert abs(len(test) - int(0.2 * len(df))) <= 1


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # make sure the model is using the right classifier
    """
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


# TODO: implement the third test. Change the function name and input as needed
def test_dataset_shape():
    """
    # ensure the working dataset is large enough and has the right amount of functions
    """
    min_columns = 10 

    assert df.shape[0] > 1000
    assert df.shape[1] >= min_columns
