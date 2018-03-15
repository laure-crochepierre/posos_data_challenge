import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
def batch_generator(X, y, batch_size):
    """Primitive batch generator 
    """
    size = len(X)
    X_copy = X.copy()
    y_copy = y.copy()
    indices = np.arange(size)
    np.random.shuffle(indices)
    X_copy = X_copy[indices]
    y_copy = y_copy[indices]
    i = 0
    while True:
        if i + batch_size <= size:
            yield X_copy[i:i + batch_size], y_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            continue


def corpus_from_csv(input_path,output_path):
    df_input = pd.read_csv(input_path, sep=";", index_col=0)
    df_output = pd.read_csv(output_path, sep=";", index_col=0)
    
    y_data = np.array([int(x[0]) for x in df_output.values])
    X_data = np.array([np.array(x[0].split()) for x in df_input.values])
    
    return X_data,y_data

def train_test_validation_split(X, y, test_size=0.2, random_state=42):           
    X_to_split, X_validation, y_to_split, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_to_split, y_to_split, test_size=0.2, random_state=42)
    return X_train, X_test, X_validation, y_train, y_test, y_validation