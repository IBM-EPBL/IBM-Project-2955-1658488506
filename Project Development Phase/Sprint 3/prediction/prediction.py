import numpy as np
import pandas as pd
import pickle
from matplotlib import rcParams
from matplotlib.cm import rainbow
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os
def initialize_dataset():
    global df
    df = pd.read_csv('prediction/heart.csv')

def get_user_data():
    print("get_user_data")
    
def get_data():
    return df

def get_info():
    return df.info()

def describe():
    return df.describe()

def get_head():
    return df.head()

def get_histogram():
    return df.hist()
   
def train_model():
    from sklearn.ensemble import RandomForestRegressor
    
    initialize_dataset()
    
    X = df.drop("target", axis=1)
    y = df["target"]
    
    model = RandomForestRegressor()
    model.fit(X, y)
    print("Train complete")
    pickle.dump(model, open('prediction/model.pkl', 'wb'))
     
def get_prediction(record):
    
    initialize_dataset()
    
    X_test = np.array([record.age, record.sex, record.cp, record.trestbps, record.chol, record.fbs, record.restecg, \
        record.thalach, record.exang, record.oldpeak, record.slope, record.ca, record.thal])
    X_test = X_test.reshape((1,-1))
    pickled_model = pickle.load(open('prediction/model.pkl', 'rb'))
    
    val = pickled_model.predict(X_test)
    
    add_data(X_test, int(val), df)
    #add data to the csv
    #initiate training with new data
    record.target = int(val)
    return record

def add_data(X_test, val, df):
    temp = np.append(X_test, val)
    temp = temp.reshape((1,-1))
    temp = pd.DataFrame(temp, columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'])
    df = df.append(temp)
    open("prediction/heart.csv", "w").close()
    df.to_csv('prediction/heart.csv', mode='a', index=False, header=True)