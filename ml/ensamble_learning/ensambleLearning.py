from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
import pandas as pd
import numpy as np

# dataset link: https://www.kaggle.com/ronitf/heart-disease-uci/downloads/heart-disease-uci.zip/1

def prepare_data():
    original_data = pd.read_csv("./heart.csv")
    data = original_data.values
    data[:, 0] = data_normalization(data[:, 0])
    data[:, 3] = data_normalization(data[:, 3])
    data[:, 4] = data_normalization(data[:, 4])
    data[:, 7] = data_normalization(data[:, 7])

    X_ = data[:, :-1]
    y_ = data[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2, shuffle=False, random_state=0)
    return x_train, x_test, y_train, y_test

def data_normalization(data, method='max-min'):
    if method == 'max-min':
        max_value, min_value = max(data), min(data)
        data = (data - np.repeat(min_value, data.shape[0])) / (max_value - min_value)
        return data

    elif method == 'z-zero':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - np.repeat(mean, data.shape[0])) / std

def build_multi_model(x_train, x_test, y_train, y_test):
    model1 = DecisionTreeClassifier()
    model2 = KNeighborsClassifier(2)
    model3 = LogisticRegression(max_iter=10000,solver='liblinear', tol=1e-8)

    model1.fit(x_train, y_train)
    model2.fit(x_train, y_train)
    model3.fit(x_train, y_train)

    dump(model1, "model1.joblib")
    dump(model2, "model2.joblib")
    dump(model3, "model3.joblib")

    pre1 = model1.predict(x_test)
    pre2 = model2.predict(x_test)
    pre3 = model3.predict(x_test)

    pre = np.zeros(pre1.shape)
    for i in range(0, pre1.shape[0]):
        if pre1[i] + pre2[i] + pre3[i] >= 2:
            pre[i] = 1
        else:
            pre[i] = 0

    print(y_test[0:30], pre3[0:30])
    print(accuracy_score(y_test, pre1))
    print(accuracy_score(y_test, pre2))
    print(accuracy_score(y_test, pre3))
    print(accuracy_score(y_test, pre))

def load_model(x_test, y_test):
    model = load('./model3.joblib')
    result = model.predict(x_test)
    print(accuracy_score(y_test, result))

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = prepare_data()
    build_multi_model(x_train, x_test, y_train, y_test)
    # load_model(x_test, y_test)