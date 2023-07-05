#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
import pickle
import os
from src import logger

warnings.filterwarnings("ignore")

logger_v = logger.setup_logging()

data_path = (
    os.path.abspath(os.path.join(os.path.dirname("__file__"), "."))
    + "/data/Forest_fire.csv"
)

data = pd.read_csv(data_path)
data = np.array(data)


def train():
    X = data[1:, 1:-1]
    y = data[1:, -1]
    y = y.astype("int")
    X = X.astype("int")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    log_reg = LogisticRegression()

    log_reg.fit(X_train, y_train)

    inputt = [int(x) for x in "45 32 60".split(" ")]
    final = [np.array(inputt)]

    b = log_reg.predict_proba(final)

    logger_v.info(classification_report(log_reg.predict(X_test), y_test))

    pickle.dump(log_reg, open("model.pkl", "wb"))


if __name__ == "__main__":
    train()
