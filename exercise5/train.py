# save_model.py
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

data = pd.read_csv("./Housing.csv")

X, y = data.drop(["price"], axis=1), data["price"]

# Define categorical and numerical features
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

# Build a preprocessor for the pipeline using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

X = preprocessor.fit_transform(X)
# Create the regression pipeline
model = SVR()

# Train the model
model.fit(X, y)

print(X[0:3])

with open("app/model.pkl", "wb") as f:
    pickle.dump(model, f)
    
with open("app/processor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)
