import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# --- Configuration ---
DATA_PATH = 'Heart_Disease_Prediction.csv'
MODEL_PATH = 'heart_disease_model.pkl'
SCALER_PATH = 'scaler.pkl'

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None

def preprocess_data(df):
    """
    Cleans and preprocesses the data.
    - Encodes the target variable.
    - One-hot encodes categorical features.
    """
    df = df.copy()
    
    # Encode target: Presence -> 1, Absence -> 0
    if 'Heart Disease' in df.columns:
        df['Heart Disease'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
    
    # Categorical columns to encode
    cat_cols = ['Chest pain type', 'EKG results', 'Slope of ST', 'Thallium']
    
    # One-hot encoding
    df_processed = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    return df_processed

def train_model(X_train, y_train):
    """Trains a Logistic Regression model."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and saves the confusion matrix plot."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix: Heart Disease Prediction')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to 'confusion_matrix.png'")

def main():
    # 1. Load Data
    df = load_data(DATA_PATH)
    if df is None:
        return
    
    # 2. Preprocessing
    df_clean = preprocess_data(df)
    
    # 3. Split Data
    X = df_clean.drop('Heart Disease', axis=1)
    y = df_clean['Heart Disease']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train Model
    print("Training model...")
    model = train_model(X_train_scaled, y_train)
    
    # 6. Evaluate
    evaluate_model(model, X_test_scaled, y_test)
    
    # 7. Save Artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")

if __name__ == "__main__":
    main()