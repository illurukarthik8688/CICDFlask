from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import pandas as pd

# Load the dataset
df = pd.read_csv('your_data.csv')

# Drop nulls
df = df.dropna()

# Encode categorical features
categorical_columns = ["stage_name", "job_name", "task_name", "pipeline_id"]

encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
df["status"] = target_encoder.fit_transform(df["status"])

X = df[categorical_columns]
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print(classification_report(y_test, pred))
