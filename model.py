import time
time.sleep(10)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Generate synthetic dataset
np.random.seed(42)

data_size = 500

build_time = np.random.normal(10, 2, data_size)
failure_rate = np.random.normal(0.2, 0.1, data_size)
commit_freq = np.random.normal(5, 2, data_size)
deploy_delay = np.random.normal(3, 1, data_size)

# Simple anomaly condition
anomaly = (build_time > 14) | (failure_rate > 0.4) | (deploy_delay > 5)
anomaly = anomaly.astype(int)

df = pd.DataFrame({
    "build_time": build_time,
    "failure_rate": failure_rate,
    "commit_freq": commit_freq,
    "deploy_delay": deploy_delay,
    "anomaly": anomaly
})

X = df.drop("anomaly", axis=1)
y = df["anomaly"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, "anomaly_model.pkl")

print("Model trained and saved successfully.")
