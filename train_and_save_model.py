import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Ensure the CSV file is accessible
file_path = os.path.join(os.path.dirname(__file__), "MASTER_DATA.csv")

def load_data(file_path):
    # Load the original CSV file and remove the specified columns
    Data = pd.read_csv(file_path).drop(columns=["Subtest"])
    # Replace 'No Dyslexia' with 0, 'Mild' with 1, and 'Severe' with 2
    Data.replace({'No Dyslexia': 0, 'Mild': 1, 'Severe': 2}, inplace=True)
    # Separate the features (X) and the target (y)
    X = Data.iloc[0:25].values.T.astype(float)  # Transpose X to ensure rows are samples and columns are features
    y = Data.iloc[25].values.astype(int)  # Ensure y is an array of integers
    return X, y

def create_pipeline():
    # Define the pipeline with scaling, normalization, PCA, and the classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),       # Standardize features by removing the mean and scaling to unit variance
        ('normalizer', Normalizer()),       # Normalize samples individually to unit norm
        ('pca', PCA()),                     # PCA for dimensionality reduction
        ('classifier', RandomForestClassifier(random_state=42))  # RandomForest classifier
    ])
    return pipeline

def train_model(X, y):
    # Split the data into training and testing sets using stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline = create_pipeline()

    # Define the parameter grid
    param_grid = {
        'pca__n_components': [2, 3, 4],
        'classifier__n_estimators': [10, 50, 100]
    }

    # Create the GridSearchCV object with recall as the scoring metric and stratified k-fold cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(n_splits=5), scoring='recall_macro', verbose=1)

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Return the best estimator
    return grid_search.best_estimator_

def save_model(model, filepath):
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def main():
    X, y = load_data(file_path)
    best_pipeline = train_model(X, y)
    save_model(best_pipeline, "best_model.pkl")

if __name__ == "__main__":
    main()
