import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pickle
import os

df = pd.read_csv('bank.csv')

# CHECK dataset target column
print(df.columns)

# Change target based on actual dataset
TARGET_COLUMN = 'deposit'   # <--- CHANGE HERE

label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

os.makedirs('model', exist_ok=True)
pickle.dump(model, open('model/xgb_model.pkl', 'wb'))
pickle.dump(label_encoders, open('model/label_encoders.pkl', 'wb'))
pickle.dump(list(X.columns), open('model/columns.pkl', 'wb'))

print("✅ Correct Model Training Completed!")
print(f"Training Accuracy: {model.score(X_train, y_train)}")
print(f"Testing Accuracy: {model.score(X_test, y_test)}")