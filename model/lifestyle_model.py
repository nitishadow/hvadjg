# model/lifestyle_model.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score

# Load and preprocess the dataset
df = pd.read_csv('Lifestyle Data.csv')
df = pd.get_dummies(df, columns=['Gender', 'Stress_Level'], drop_first=True)
X = df.drop('Healthy_Lifestyle_Score', axis=1).values
y = df['Healthy_Lifestyle_Score'].values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train the RandomForest model
def train_model():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    print(f'Cross-validated R2 Score: {np.mean(r2_scores) * 100:.2f}%')

    model.fit(X, y)
    return model

# Initialize the model and scaler once
rf_model = train_model()

# Prediction function
def predict_lifestyle_score(data):
    user_df = pd.DataFrame(data, index=[0])
    user_input = scaler.transform(user_df.values)
    prediction = rf_model.predict(user_input)
    return prediction[0]

